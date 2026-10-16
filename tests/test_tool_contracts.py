import inspect
import json
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

from deepseek_client import DeepSeekClient
from mcp_client import McpManager
from team_tools import TeamTool
from tests.support import make_settings
from tool_runtime import ExecutionContext, ToolResult, validate
from tools import TOOL_NAMES, WORKER_EXCLUDED, CodingTools, ToolPolicy
from websearch import WebSearchClient


def sample_arguments(schema):
    if "enum" in schema:
        return schema["enum"][0]
    kind = schema["type"]
    if kind == "object":
        return {
            name: sample_arguments(schema["properties"][name])
            for name in schema.get("required", [])
        }
    if kind == "array":
        return [sample_arguments(schema["items"]) for _ in range(schema.get("minItems", 1))]
    if kind == "string":
        return "sample" * max(1, schema.get("minLength", 0))
    if kind == "integer":
        return schema.get("minimum", 1)
    if kind == "boolean":
        return False
    raise AssertionError(f"Unsupported tool schema type: {kind}")


class ToolContractTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        settings = make_settings(
            api_key="test",
            base_url="https://example.invalid",
            model="test",
            session_file=self.workspace / "session.json",
            max_context_chars=120000,
            keep_recent_turns=6,
            max_task_seconds=3600,
            max_subagents=2,
            workspace=self.workspace,
        )
        self.tools = CodingTools(self.workspace)
        self.tools.attach_websearch(WebSearchClient(settings))
        manager = McpManager(self.workspace / "unused.json", self.workspace, servers={})
        self.addCleanup(manager.close)
        self.tools.attach_mcp(manager)
        self.tools.attach_user_input(lambda question, options: {"answer": "choice"})
        self.tools.attach_reasoning(DeepSeekClient(settings), max_workers=2)
        self.tools.registry["team"] = TeamTool(None, "main").spec()

    def test_declared_names_cover_every_registered_and_optional_tool(self):
        self.assertEqual(set(self.tools.registry), TOOL_NAMES)
        definitions = self.tools.definitions
        names = [item["function"]["name"] for item in definitions]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(set(names), TOOL_NAMES)
        self.assertLessEqual(WORKER_EXCLUDED, TOOL_NAMES)
        for name, spec in self.tools.registry.items():
            with self.subTest(name=name):
                self.assertEqual(name, spec.name)
                self.assertEqual(spec.definition()["type"], "function")
                self.assertIs(
                    spec.definition()["function"]["parameters"]["additionalProperties"], False
                )

    def test_schemas_and_handler_signatures_agree(self):
        for name, spec in self.tools.registry.items():
            with self.subTest(name=name):
                signature = inspect.signature(spec.handler)
                parameters = signature.parameters
                self.assertIn("_context", parameters)
                self.assertIs(parameters["_context"].kind, inspect.Parameter.KEYWORD_ONLY)
                public = {key: value for key, value in parameters.items() if key != "_context"}
                self.assertEqual(set(spec.properties), set(public))
                required = {
                    key for key, value in public.items() if value.default is inspect.Parameter.empty
                }
                self.assertEqual(set(spec.required), required)
                arguments = sample_arguments(spec.schema)
                validate(arguments, spec.schema)
                signature.bind(**arguments, _context=ExecutionContext(time.monotonic() + 1))
                with self.assertRaises(ValueError):
                    validate({**arguments, "unexpected_argument": True}, spec.schema)

    def test_read_only_classification_does_not_expose_mutating_tools_by_default(self):
        expected = {"read", "grep", "find", "ls", "websearch"}
        actual = {name for name, spec in self.tools.registry.items() if spec.read_only}
        self.assertEqual(actual, expected)
        self.assertEqual(
            {item["function"]["name"] for item in self.tools.worker_definitions}, expected
        )

    def test_advertised_permissions_match_actual_dispatch_for_parents_and_workers(self):
        policies = (
            ToolPolicy(),
            ToolPolicy(allow=frozenset()),
            ToolPolicy(allow=TOOL_NAMES),
            ToolPolicy(deny=TOOL_NAMES),
            ToolPolicy(
                allow=frozenset({"read", "write", "mcp", "delegate"}), deny=frozenset({"read"})
            ),
        )
        invoked = []
        for name, spec in tuple(self.tools.registry.items()):

            def handler(*, _tool_name=name, _context, **arguments):
                invoked.append(_tool_name)
                return ToolResult(data={"tool": _tool_name})

            self.tools.registry[name] = replace(spec, handler=handler)
        for policy in policies:
            self.tools.policy = policy
            for granted in (frozenset(), TOOL_NAMES):
                self.tools.worker_allow = granted
                for worker in (False, True):
                    definitions = (
                        self.tools.worker_definitions if worker else self.tools.definitions
                    )
                    advertised = {item["function"]["name"] for item in definitions}
                    for name, spec in self.tools.registry.items():
                        expected = policy.permits(name) and (
                            not worker
                            or (name not in WORKER_EXCLUDED and (spec.read_only or name in granted))
                        )
                        if name == "monitor":
                            expected = expected and policy.permits("exec")
                        with self.subTest(policy=policy, granted=granted, worker=worker, name=name):
                            invoked.clear()
                            call = {
                                "id": f"call-{name}",
                                "type": "function",
                                "function": {
                                    "name": name,
                                    "arguments": json.dumps(sample_arguments(spec.schema)),
                                },
                            }
                            output = (
                                self.tools.execute_workspace(
                                    call, ExecutionContext(time.monotonic() + 1)
                                )
                                if worker
                                else self.tools.execute(call)
                            )
                            result = json.loads(output)
                            self.assertEqual(name in advertised, expected)
                            self.assertEqual(invoked, [name] if expected else [])
                            self.assertEqual(result["status"], "ok" if expected else "error")
                            if not expected:
                                self.assertEqual(result["error"]["code"], "unavailable_tool")

    def test_reserved_worker_tools_cannot_be_granted_explicitly(self):
        self.tools.worker_allow = TOOL_NAMES
        advertised = {item["function"]["name"] for item in self.tools.worker_definitions}
        self.assertTrue(advertised.isdisjoint(WORKER_EXCLUDED))
        for name in WORKER_EXCLUDED:
            self.assertFalse(self.tools.tool_available(name, worker=True))

    def test_detached_optional_tools_are_neither_advertised_nor_executable(self):
        self.tools.worker_allow = TOOL_NAMES
        for name in ("websearch", "mcp", "ask_user", "delegate"):
            self.tools.detach(name)
            self.assertFalse(self.tools.tool_available(name))
            self.assertFalse(self.tools.tool_available(name, worker=True))
            for definitions in (self.tools.definitions, self.tools.worker_definitions):
                self.assertNotIn(name, {item["function"]["name"] for item in definitions})
            result = json.loads(
                self.tools.execute(
                    {
                        "id": name,
                        "type": "function",
                        "function": {"name": name, "arguments": "{}"},
                    }
                )
            )
            self.assertEqual(result["error"]["code"], "unavailable_tool")


if __name__ == "__main__":
    unittest.main()
