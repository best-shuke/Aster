import json
import tempfile
import time
import unittest
from dataclasses import replace
from datetime import date
from itertools import pairwise
from pathlib import Path
from unittest.mock import patch

from config import environment_context, system_prompt
from context_management import compact_messages
from conversation import Conversation
from deepseek_client import CompletionError
from reasoning import SubagentRunner
from task_acceptance import TaskContract
from task_runtime import TaskRuntime
from tests.support import make_settings
from tool_runtime import ExecutionContext
from tools import CodingTools


class Client:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, messages, **kwargs):
        self.requests.append({"messages": json.loads(json.dumps(messages)), **kwargs})
        if kwargs.get("purpose") == "summary":
            return {"role": "assistant", "content": "A short summary without failure details."}
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def answer(text="done"):
    return {"role": "assistant", "content": text}


def call(identity, path):
    return {
        "id": identity,
        "type": "function",
        "function": {"name": "read", "arguments": json.dumps({"path": path})},
    }


class PromptCacheTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.settings = make_settings(
            api_key="test",
            base_url="https://example.invalid",
            model="fake",
            session_file=self.root / "session.json",
            workspace=self.root,
            max_context_chars=120000,
            keep_recent_turns=6,
            max_task_seconds=60,
            max_subagents=1,
        )

    def assert_prefixes(self, requests):
        for previous, current in pairwise(requests):
            before, after = previous["messages"], current["messages"]
            self.assertEqual(before, after[: len(before)])
            self.assertEqual(previous.get("tools"), current.get("tools"))

    def test_tool_rounds_repairs_and_checkpoints_keep_the_whole_previous_input(self):
        (self.root / "ok.txt").write_text("verified")
        client = Client(
            {"role": "assistant", "tool_calls": [call("1", "absent.txt"), call("2", "ok.txt")]},
            answer("<｜｜DSML｜｜ calls>not executed</｜｜DSML｜｜ calls>"),
            CompletionError(
                "generation_budget", "checkpoint", partial_message={"reasoning": "partial evidence"}
            ),
            {"role": "assistant", "tool_calls": [call("3", "ok.txt")]},
            answer(),
        )
        conversation = Conversation(self.settings, client, CodingTools(self.root))
        self.assertEqual(conversation.ask("inspect"), "done")
        self.assert_prefixes(client.requests)
        # A note must not split the assistant/tool-result group.
        messages = client.requests[1]["messages"]
        assistant = next(i for i, m in enumerate(messages) if m.get("tool_calls"))
        self.assertEqual(
            [m["tool_call_id"] for m in messages[assistant + 1 : assistant + 3]], ["1", "2"]
        )
        self.assertIn("unresolved_failures", messages[-1]["content"])
        self.assertEqual(
            json.loads(self.settings.session_file.read_text())["messages"], conversation.messages
        )

    def test_new_turn_date_and_acceptance_survive_restart_without_changing_prefix(self):
        first = Client(answer("first"))
        conversation = Conversation(self.settings, first, None)
        with patch("config.date") as today:
            today.today.return_value = date(2026, 9, 24)
            conversation.ask(
                "first request", task_contract=TaskContract("answer", description="first criteria")
            )
        second = Client(answer("second"))
        restored = Conversation(self.settings, second, None)
        with patch("config.date") as today:
            today.today.return_value = date(2026, 9, 25)
            restored.ask(
                "second request",
                task_contract=TaskContract("answer", description="second criteria"),
            )
        self.assert_prefixes([first.requests[0], second.requests[0]])
        self.assertIn("2026-09-25", second.requests[0]["messages"][-1]["content"])
        self.assertIn("second criteria", second.requests[0]["messages"][-1]["content"])
        self.assertNotIn("first criteria", first.requests[0]["messages"][0]["content"])

    def test_workspace_does_not_change_shared_system_prompt(self):
        self.assertEqual(
            system_prompt(self.root, self.settings.prompt_file),
            system_prompt(self.root / "other", self.settings.prompt_file),
        )

    def test_successful_tool_results_do_not_duplicate_observation_summaries(self):
        (self.root / "ok.txt").write_text("verified")
        client = Client({"role": "assistant", "tool_calls": [call("1", "ok.txt")]}, answer())
        Conversation(self.settings, client, CodingTools(self.root)).ask("inspect")
        new = client.requests[1]["messages"][len(client.requests[0]["messages"]) :]
        self.assertEqual([m["role"] for m in new], ["assistant", "tool"])
        self.assert_prefixes(client.requests)

    def test_terminal_checkpoint_and_new_task_preserve_prefix_without_executing_partial_calls(self):
        client = Client(
            CompletionError(
                "task_deadline",
                "stopped",
                partial_message={
                    "role": "assistant",
                    "content": "unfinished draft",
                    "tool_calls": [call("unexecuted", "absent.txt")],
                },
            ),
            answer("resumed"),
        )
        conversation = Conversation(self.settings, client, CodingTools(self.root))
        with self.assertRaises(CompletionError):
            conversation.ask("Inspect source")
        self.assertEqual(conversation.tool_runs, [])
        self.assertIn("Stopped generation", conversation.messages[-1]["content"])
        self.assertEqual(conversation.ask("Continue using retained evidence"), "resumed")
        self.assertEqual(conversation.tool_runs, [])
        self.assert_prefixes(client.requests)

    def test_cleared_warning_is_appended_and_unchanged_warning_is_not_repeated(self):
        client = Client(answer(), answer(), answer())
        runtime = TaskRuntime(client, self.settings, time.monotonic() + 60)
        messages = [{"role": "user", "content": "task"}]
        system = {"role": "system", "content": "fixed"}
        failure = {"status": "error", "error": {"message": "not found"}, "data": {}}
        runtime.progress.observe(call("1", "missing"), failure)
        runtime.generate(messages, system, [])
        runtime.generate(messages, system, [])
        self.assertEqual(client.requests[0]["messages"], client.requests[1]["messages"])
        runtime.progress.observe(call("2", "missing"), {"status": "ok", "data": {}})
        runtime.generate(messages, system, [])
        self.assertIn("cleared", client.requests[-1]["messages"][-1]["content"])
        self.assert_prefixes(client.requests)

    def test_finalization_is_appended_once_without_rewriting_history(self):
        client = Client(answer(), answer(), answer())
        with patch("task_runtime.time.monotonic", return_value=0):
            runtime = TaskRuntime(client, self.settings, 100)
        messages = [{"role": "user", "content": "task"}]
        system = {"role": "system", "content": "fixed"}
        for now in (1, 90, 91):
            with patch("task_runtime.time.monotonic", return_value=now):
                messages.append(runtime.generate(messages, system, []))
        self.assert_prefixes(client.requests)
        self.assertEqual([r["purpose"] for r in client.requests], ["solve", "final", "final"])
        self.assertEqual(sum("Synthesis phase:" in (m.get("content") or "") for m in messages), 1)

    def test_compaction_preserves_task_context_and_then_stops_rewriting_summary(self):
        settings = replace(self.settings, max_context_chars=30000)
        client = Client(answer(), answer())
        runtime = TaskRuntime(client, settings, time.monotonic() + 60)
        runtime.progress.observe(
            call("1", "absent"), {"status": "error", "error": {"message": "missing"}, "data": {}}
        )
        context = {
            "role": "system",
            "content": environment_context(self.root) + "\nAcceptance: verify actual files",
        }
        messages = [
            {"role": "user", "content": "old"},
            answer("x" * 40000),
            {"role": "user", "content": "current"},
            context,
            answer("y" * 40000),
        ]
        system = {"role": "system", "content": "fixed"}
        messages.append(runtime.generate(messages, system, []))
        runtime.generate(messages, system, [])
        solves = [r for r in client.requests if r["purpose"] == "solve"]
        self.assert_prefixes(solves)
        self.assertIn(context, solves[0]["messages"])
        self.assertIn("unresolved_failures", solves[0]["messages"][-1]["content"])
        self.assertLess(
            len(json.dumps(solves[0]["messages"], ensure_ascii=False)), settings.max_context_chars
        )

    def test_large_recovery_notes_are_compactable_not_pinned_as_task_context(self):
        messages = [
            {"role": "user", "content": "original"},
            {"role": "system", "content": environment_context(self.root)},
            {"role": "system", "content": "<unfinished_work>" + "x" * 50000},
        ]
        changed = compact_messages(
            messages,
            {"role": "system", "content": "fixed"},
            [],
            Client(),
            replace(self.settings, max_context_chars=30000),
            time.monotonic() + 60,
        )
        self.assertTrue(changed)
        self.assertIn({"role": "user", "content": "original"}, messages)
        self.assertLess(len(json.dumps(messages)), 30000)

    def test_warning_resolution_and_synthesis_survive_compaction_together(self):
        client = Client(answer(), answer())
        settings = replace(self.settings, max_context_chars=30000)
        with patch("task_runtime.time.monotonic", return_value=0):
            runtime = TaskRuntime(client, settings, 100)
        messages = [{"role": "user", "content": "original"}]
        system = {"role": "system", "content": "fixed"}
        runtime.progress.observe(
            call("1", "missing"), {"status": "error", "error": {"message": "missing"}, "data": {}}
        )
        with patch("task_runtime.time.monotonic", return_value=1):
            runtime.generate(messages, system, [])
        messages.append(answer("x" * 50000))
        runtime.progress.observe(call("2", "missing"), {"status": "ok", "data": {}})
        with patch("task_runtime.time.monotonic", return_value=90):
            runtime.generate(messages, system, [])
        last = client.requests[-1]["messages"][-1]["content"]
        self.assertIn("Synthesis phase:", last)
        self.assertIn("cleared", last)
        self.assertEqual(client.requests[-1]["purpose"], "final")

    def test_workers_put_shared_reference_before_distinct_assignments(self):
        client = Client(answer(), answer())
        runner = SubagentRunner(client, None, [], self.root)
        for task in ("inspect API", "inspect tests"):
            result = runner.run(
                task,
                "Shared evidence\n" * 100,
                "report",
                ExecutionContext(time.monotonic() + 60),
                mode="reason",
            )
            self.assertEqual(result.status, "ok")
        a, b = [r["messages"][1]["content"].split("Assignment:\n")[0] for r in client.requests]
        self.assertEqual(a, b)
        self.assertIn("Shared evidence", a)


if __name__ == "__main__":
    unittest.main()
