import json
import shlex
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from mcp.types import CallToolResult

from conversation import Conversation
from mcp_client import McpManager
from task_runtime import TaskProgress
from tests.support import make_settings
from tool_runtime import MAX_RESULT_CHARS, ArtifactStore, ExecutionContext, ToolResult
from tools import CodingTools
from ui import Console


def call(name, arguments):
    return {
        "id": "reliability-call",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class ToolReliabilityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.tools = CodingTools(self.root)
        self.addCleanup(self.tools.processes.close)

    def execute(self, name, arguments, **options):
        return json.loads(self.tools.execute(call(name, arguments), **options))

    def test_returned_cursor_round_trips_for_every_paginated_tool(self):
        for index in range(5):
            (self.root / f"file-{index}.txt").write_text(f"needle {index}\n")
        (self.root / "long.txt").write_text("中文" * 4000 + "\nlast\n")
        queries = {
            "read": {"path": "long.txt", "limit": 1},
            "grep": {"pattern": "needle", "limit": 1},
            "find": {"pattern": "file-*.txt", "limit": 1},
            "ls": {"limit": 1},
        }
        for name, query in queries.items():
            with self.subTest(tool=name):
                first = self.execute(name, query)
                self.assertEqual(first["status"], "ok")
                self.assertIsNotNone(first["next_cursor"])
                second = self.execute(name, {**query, "next_cursor": first["next_cursor"]})
                merged = self.execute(name, {**query, **first["next_cursor"]})
                self.assertEqual(second["status"], "ok", second)
                self.assertEqual(second["data"], merged["data"])
                self.assertEqual(second["next_cursor"], merged["next_cursor"])

    def test_long_line_cursor_reads_all_content_without_gaps(self):
        content = "中文" * 9000 + "\nlast line\n"
        (self.root / "long.txt").write_text(content)
        arguments, chunks = {"path": "long.txt", "limit": 1}, []
        for _ in range(10):
            result = self.execute("read", arguments)
            self.assertEqual(result["status"], "ok", result)
            chunks.append(result["data"]["content"])
            if result["next_cursor"] is None:
                break
            arguments["next_cursor"] = result["next_cursor"]
        self.assertEqual("".join(chunks), content)

    def test_double_encoded_cursor_is_decoded_but_invalid_positions_are_rejected(self):
        (self.root / "file.txt").write_text("one\ntwo\nthree\n")
        cursor = {"offset": 2, "column": 0}
        result = self.execute("read", {"path": "file.txt", "next_cursor": json.dumps(cursor)})
        self.assertEqual(result["data"]["content"], "two\nthree\n")
        for extra in [
            {"offset": 1, "next_cursor": cursor},
            {"next_cursor": {"offset": 0}},
            {"next_cursor": {"offset": True}},
            {"next_cursor": {"offset": 2, "unexpected": 1}},
            {"next_cursor": "not JSON"},
        ]:
            with self.subTest(extra=extra):
                result = self.execute("read", {"path": "file.txt", **extra})
                self.assertEqual(result["error"]["code"], "invalid_arguments")
                self.assertIn("input_schema", result["data"])

    def test_url_read_has_actionable_error(self):
        result = self.execute("read", {"path": "https://example.invalid/page"})
        self.assertEqual(result["error"]["code"], "invalid_arguments")
        self.assertIn("URL", result["error"]["message"])

    def test_binary_read_says_how_to_extract_text(self):
        (self.root / "paper.pdf").write_bytes(b"%PDF-1.4\n\xe2\x28\xa1\x00\xff")
        result = self.execute("read", {"path": "paper.pdf"})
        self.assertEqual(result["error"]["code"], "not_text")
        self.assertIn("pdftotext", result["error"]["message"])

    def test_oversized_regex_reports_a_recovery_method(self):
        (self.root / "sample.txt").write_text("needle\n")
        result = self.execute("grep", {"pattern": "^.{0,1000000}needle"})
        self.assertEqual(result["error"]["code"], "regex_too_large")
        self.assertIn("literal=true", result["data"]["guidance"])
        recovered = self.execute("grep", {"pattern": "needle", "literal": True})
        self.assertEqual(recovered["status"], "ok")
        self.assertEqual(len(recovered["data"]["matches"]), 1)

    def test_foreground_progress_is_emitted_before_process_exit(self):
        events = []
        release = self.root / "release"
        program = (
            "import time; from pathlib import Path; print('READY'); "
            "p=Path('release'); deadline=time.monotonic()+4\n"
            "while not p.exists() and time.monotonic()<deadline: time.sleep(.02)\n"
            "assert p.exists(); print('DONE')"
        )

        def progress(event):
            events.append(event)
            if "READY" in event["output"]:
                self.assertNotIn("DONE", event["output"])
                release.touch()

        command = f"{shlex.quote(sys.executable)} -c {shlex.quote(program)}"
        result = self.execute("exec", {"command": command}, on_progress=progress)
        self.assertEqual(result["status"], "ok", result)
        self.assertTrue(events)
        self.assertEqual(events[0]["phase"], "tool_progress")
        self.assertGreater(events[0]["received_bytes"], 0)
        self.assertIn("DONE", result["data"]["output"])

    def test_progress_callback_failure_does_not_fail_or_repeat_the_command(self):
        def broken_progress(event):
            raise RuntimeError("display unavailable")

        program = "import time; from pathlib import Path; Path('effect').write_text('once'); time.sleep(1.1)"
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote(program)}"
        result = self.execute("exec", {"command": command}, on_progress=broken_progress)
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual((self.root / "effect").read_text(), "once")

    def test_conversation_routes_running_command_progress_to_its_caller(self):
        program = "import time; print('READY'); time.sleep(1.1); print('DONE')"
        command = f"{shlex.quote(sys.executable)} -c {shlex.quote(program)}"
        settings = make_settings(
            api_key="test",
            base_url="https://example.invalid",
            model="test",
            session_file=self.root / "session.json",
            workspace=self.root,
            max_context_chars=120000,
            keep_recent_turns=6,
            max_task_seconds=60,
            max_subagents=1,
        )
        client = SimpleNamespace(
            complete=Mock(
                side_effect=[
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [call("exec", {"command": command})],
                    },
                    {"role": "assistant", "content": "Inspection complete"},
                ]
            )
        )
        conversation = Conversation(settings, client, self.tools)
        events = []
        self.assertEqual(
            conversation.ask("Inspect command output", on_progress=events.append),
            "Inspection complete",
        )
        progress = [event for event in events if event["phase"] == "tool_progress"]
        self.assertTrue(progress)
        self.assertIn("READY", progress[0]["output"])
        self.assertEqual(progress[0]["trace"]["tool_call_id"], "reliability-call")
        self.assertEqual(conversation.tool_runs[0]["state"], "completed")

    def test_large_result_keeps_cursor_and_error_retryability(self):
        result = ToolResult(
            status="error",
            data={"large": "x" * (MAX_RESULT_CHARS + 1000)},
            error={"code": "IO_ERROR", "message": "Cannot save file", "retryable": False},
            next_cursor={"offset": 2},
        )
        rendered = json.loads(ArtifactStore(self.root).render(result))
        self.assertEqual(rendered["error"], result.error)
        self.assertEqual(rendered["next_cursor"], result.next_cursor)
        self.assertTrue(Path(rendered["artifact_path"]).exists())

    def test_same_error_across_different_arguments_prompts_a_method_change(self):
        progress = TaskProgress(self.root)
        for index in range(3):
            progress.observe(
                call("grep", {"pattern": f"^.{{0,1000000}}needle{index}"}),
                {"status": "error", "error": {"code": "regex_too_large", "message": "too large"}},
            )
        self.assertIn("repeated_failures", progress.guidance())
        self.assertIn("change the method", progress.guidance())
        progress.observe(call("grep", {"pattern": "needle", "literal": True}), {"status": "ok"})
        self.assertNotIn("repeated_failures", progress.guidance())


class McpReliabilityTests(unittest.TestCase):
    def setUp(self):
        self.schema = {
            "type": "object",
            "properties": {"index": {"type": "integer", "minimum": 1}},
            "required": ["index"],
            "additionalProperties": False,
        }
        self.metadata = {"name": "action", "inputSchema": self.schema}
        self.connection = Mock()
        self.connection.request.return_value = CallToolResult(
            content=[], structured_content={"ok": True}
        )
        self.manager = object.__new__(McpManager)
        self.manager.servers = {"test": SimpleNamespace(timeout=30)}
        self.manager._find_tool = Mock(return_value=("test", self.metadata))
        self.manager._connection = Mock(return_value=self.connection)

    def execute(self, arguments):
        return self.manager.call(
            "test/action", arguments, None, ExecutionContext(time.monotonic() + 5)
        )

    def test_wrong_parameter_names_and_types_never_reach_the_server(self):
        for arguments in [{"number": 70}, {"index": "70"}, {"index": True}, {"index": 0}]:
            with self.subTest(arguments=arguments):
                result = self.execute(arguments)
                self.assertEqual(result.error["code"], "invalid_arguments")
                self.assertEqual(result.data["input_schema"], self.schema)
        self.connection.request.assert_not_called()
        self.assertEqual(self.execute({"index": 70}).status, "ok")
        self.connection.request.assert_called_once()

    def test_nested_local_references_are_validated_without_rewriting_arguments(self):
        self.metadata["inputSchema"] = {
            "$defs": {"index": {"type": "integer", "minimum": 1}},
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": {"index": {"$ref": "#/$defs/index"}},
                    "required": ["index"],
                }
            },
            "required": ["nested"],
        }
        result = self.execute({"nested": {"index": "1"}})
        self.assertEqual(result.error["code"], "invalid_arguments")
        self.assertIn("arguments.nested.index", result.error["message"])
        self.connection.request.assert_not_called()
        arguments = {"nested": {"index": 1}}
        self.assertEqual(self.execute(arguments).status, "ok")
        self.assertEqual(self.connection.request.call_args.args[1]["arguments"], arguments)

    def test_remote_schema_references_do_not_start_network_requests_or_tool_calls(self):
        self.metadata["inputSchema"] = {"$ref": "https://example.invalid/schema"}
        result = self.execute({"index": 1})
        self.assertEqual(result.error["code"], "invalid_tool_schema")
        self.connection.request.assert_not_called()

    def test_original_error_code_message_and_retryability_survive(self):
        error = {
            "code": "CONFIRMATION_REQUIRED",
            "message": "Requires explicit confirmation",
            "retryable": False,
        }
        self.connection.request.return_value = CallToolResult(
            content=[],
            structured_content={"error": error},
            is_error=True,
        )
        result = self.execute({"index": 1})
        self.assertEqual(result.error, error)
        self.connection.request.assert_called_once()
        self.assertEqual(self.connection.request.call_args.args[1]["arguments"], {"index": 1})

    def test_text_only_errors_are_visible(self):
        self.connection.request.return_value = CallToolResult(
            content=[{"type": "text", "text": "Could not save file: parent directory missing"}],
            is_error=True,
        )
        result = self.execute({"index": 1})
        self.assertEqual(result.error["code"], "mcp_tool_error")
        self.assertIn("parent directory missing", result.error["message"])


class ProgressDisplayTests(unittest.TestCase):
    def test_command_progress_updates_display_without_publishing_an_answer(self):
        console = Console()
        console.interactive = True
        console._live = Mock()
        console.progress(
            {
                "phase": "tool_progress",
                "tool": "exec",
                "elapsed_seconds": 31,
                "received_bytes": 2048,
                "output": "old\n\x1b[2JREADY",
            }
        )
        self.assertIn("READY", console._task_activity)
        self.assertIn("2.0 KB", console._task_activity)
        self.assertNotIn("\x1b", console._task_activity)
        self.assertEqual(console._response_content, [])
        console._live.refresh.assert_called_once()

    def test_tool_argument_generation_is_visible_and_reset_for_the_next_request(self):
        console = Console()
        console.interactive = True
        console._live = Mock()
        console.progress({"phase": "tool_call_delta", "tool": "delegate", "argument_chars": 1234})
        self.assertIn("生成工具参数", console._task_activity)
        self.assertIn("1,234", console._task_activity)
        console.progress({"phase": "receiving", "received_bytes": 4096})
        self.assertIn("生成工具参数", console._task_activity)
        self.assertEqual(console._preview_content, [])
        console.progress({"phase": "response_reset"})
        self.assertFalse(console._tool_parameter_progress)

    def test_noninteractive_command_progress_is_rate_limited(self):
        console = Console()
        console.interactive = False
        with (
            patch.object(console, "info") as info,
            patch("ui.time.monotonic", side_effect=[0, 1, 31]),
        ):
            for _ in range(3):
                console.progress({"phase": "tool_progress", "tool": "exec", "output": "READY"})
        self.assertEqual(info.call_count, 2)


if __name__ == "__main__":
    unittest.main()
