import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from execution_trace import TraceContext
from tests.support import make_settings
from tool_runtime import ToolResult
from tools import CodingTools
from worker_checkpoint import WorkerCheckpoint


def tool_call(name, arguments, identity="read-1"):
    return {
        "id": identity,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class WorkerCheckpointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.trace = TraceContext.task("session").agent("A1").request()
        self.system = {"role": "system", "content": "fixed worker instructions"}
        self.settings = make_settings(
            api_key="test",
            base_url="https://example.invalid",
            model="fixture",
            workspace=self.root,
            session_file=self.root / "session.json",
            max_context_chars=120000,
            keep_recent_turns=6,
            max_task_seconds=None,
            max_subagents=20,
        )

    def test_compaction_retains_actual_tool_observations_and_executed_identities(self):
        checkpoint = WorkerCheckpoint(self.root)
        call = tool_call("read", {"path": "source.txt"})
        output = json.dumps(ToolResult(data={"content": "actual source evidence"}).as_dict())
        checkpoint.tool_started(call)
        checkpoint.tool_completed(call, output, self.trace.tool(call["id"]))
        checkpoint.capture(
            self.system,
            [{"role": "system", "content": "compacted summary"}],
            {"checks": [{"tool": "read", "status": "ok"}]},
            self.trace,
        )
        result = checkpoint.failure("deadline_exceeded", "deadline")
        state = json.loads(Path(result.data["checkpoint_path"]).read_text())
        self.assertEqual(state["completed_tool_call_ids"], [call["id"]])
        self.assertIn("actual source evidence", state["tool_observations"][0]["result"])
        self.assertEqual(result.data["tool_calls"], 1)
        self.assertEqual(state["system"], self.system)

    def test_inflight_operation_is_unknown_and_not_queued_for_replay(self):
        checkpoint = WorkerCheckpoint(self.root)
        started = tool_call("write", {"path": "effect.txt", "content": "effect"}, "started")
        pending = tool_call("read", {"path": "next.txt"}, "pending")
        checkpoint.capture(
            self.system,
            [{"role": "assistant", "tool_calls": [started, pending]}],
            {},
            self.trace,
        )
        checkpoint.tool_started(started)
        result = checkpoint.failure("deadline_exceeded", "deadline")
        state = json.loads(Path(result.data["checkpoint_path"]).read_text())
        self.assertEqual(state["unknown_tool_call_ids"], ["started"])
        self.assertEqual(state["pending_tool_call_ids"], ["pending"])
        self.assertEqual(result.status, "timeout")

    def test_unknown_tool_result_and_malformed_receipt_remain_unknown(self):
        for output in [json.dumps({"status": "unknown"}), "invalid JSON"]:
            with self.subTest(output=output):
                checkpoint = WorkerCheckpoint(self.root)
                call = tool_call("exec", {"command": "fixture"})
                checkpoint.tool_completed(call, output, self.trace.tool(call["id"]))
                result = checkpoint.failure("deadline_exceeded", "deadline")
                state = json.loads(Path(result.data["checkpoint_path"]).read_text())
                self.assertEqual(state["unknown_tool_call_ids"], [call["id"]])

    def test_retry_keeps_the_latest_partial_and_marks_it_unverified(self):
        checkpoint = WorkerCheckpoint(self.root)
        checkpoint.begin_generation()
        checkpoint.progress({"phase": "response_delta", "content": "old attempt"})
        checkpoint.progress({"phase": "response_reset"})
        checkpoint.progress({"phase": "response_delta", "content": "latest draft"})
        result = checkpoint.failure("deadline_exceeded", "deadline")
        state = json.loads(Path(result.data["checkpoint_path"]).read_text())
        self.assertEqual(state["unfinished_response"]["content"], "latest draft")
        self.assertFalse(state["unfinished_response_verified"])
        self.assertFalse(result.data["report_verified"])

    def test_partial_message_survives_preview_reset_and_late_updates_cannot_replace_it(self):
        checkpoint = WorkerCheckpoint(self.root)
        checkpoint.progress({"phase": "response_delta", "content": "stream fragment"})
        checkpoint.progress({"phase": "response_reset"})
        checkpoint.preserve_partial({"content": "complete interrupted response"})
        result = checkpoint.failure("deadline_exceeded", "deadline")
        path = Path(result.data["checkpoint_path"])
        original = path.read_text()
        checkpoint.capture(self.system, [], {"report": "late accepted answer"}, self.trace)
        checkpoint.progress({"phase": "response_delta", "content": "late fragment"})
        checkpoint.preserve_partial({"content": "late replacement"})
        again = checkpoint.failure("deadline_exceeded", "deadline")
        self.assertEqual(Path(again.data["checkpoint_path"]).read_text(), original)
        self.assertIn("complete interrupted response", original)
        self.assertNotIn("late accepted answer", original)

    def test_failed_checkpoint_write_retains_state_and_can_be_retried(self):
        checkpoint = WorkerCheckpoint(self.root)
        checkpoint.capture(self.system, [{"role": "user", "content": "assignment"}], {}, self.trace)
        checkpoint.progress({"phase": "response_delta", "content": "retained draft"})
        with patch("worker_checkpoint.atomic_write_text", side_effect=OSError("disk fixture")):
            result = checkpoint.failure("deadline_exceeded", "deadline")
        self.assertEqual(
            result.data["checkpoint"]["unfinished_response"]["content"], "retained draft"
        )
        self.assertEqual(result.data["checkpoint_error"], "disk fixture")
        restored = checkpoint.failure("deadline_exceeded", "deadline")
        self.assertTrue(Path(restored.data["checkpoint_path"]).is_file())

    def test_empty_error_partial_does_not_erase_streamed_progress(self):
        checkpoint = WorkerCheckpoint(self.root)
        checkpoint.progress({"phase": "response_delta", "content": "retained streamed draft"})
        for empty in ({}, {"content": None}, {"reasoning_content": ""}):
            checkpoint.preserve_partial(empty)
        result = checkpoint.failure("deadline_exceeded", "deadline")
        state = json.loads(Path(result.data["checkpoint_path"]).read_text())
        self.assertEqual(state["unfinished_response"]["content"], "retained streamed draft")

    def test_worker_checkpoint_keeps_internal_continuation_history_and_latest_partial(self):
        from deepseek_client import CompletionError
        from reasoning import SubagentRunner
        from tool_runtime import ExecutionContext

        requests = []

        class Client:
            def complete(inner, messages, **kwargs):
                requests.append(json.loads(json.dumps(messages)))
                if len(requests) == 1:
                    raise CompletionError(
                        "output_limit",
                        "fixture limit",
                        partial_message={"content": "first finding"},
                    )
                raise CompletionError(
                    "task_deadline", "fixture deadline", partial_message={"content": "next finding"}
                )

        client = Client()
        client.settings = self.settings
        runner = SubagentRunner(client, lambda *args: None, [], self.root)
        result = runner.run("Inspect", "context", "report", ExecutionContext(time.monotonic() + 60))
        state = json.loads(Path(result.data["checkpoint_path"]).read_text())
        self.assertEqual(len(requests), 2)
        self.assertEqual(result.status, "timeout")
        self.assertIn("first finding", json.dumps(state["messages"]))
        self.assertEqual(state["unfinished_response"]["content"], "next finding")

    def test_shared_deadline_preserves_completed_reads_and_unfinished_stream_without_late_tools(
        self,
    ):
        entered, release, finished = threading.Event(), threading.Event(), threading.Event()
        self.addCleanup(release.set)
        (self.root / "source.txt").write_text("verified source evidence")
        requests, executed = [], []

        class Client:
            def complete(inner, messages, **kwargs):
                requests.append(messages)
                if len(requests) == 1:
                    return {
                        "role": "assistant",
                        "tool_calls": [tool_call("read", {"path": "source.txt"})],
                    }
                kwargs["on_progress"]({"phase": "response_delta", "content": "draft finding"})
                entered.set()
                release.wait(2)
                finished.set()
                return {
                    "role": "assistant",
                    "tool_calls": [tool_call("write", {"path": "late.txt", "content": "late"})],
                }

        tools = CodingTools(self.root)
        self.addCleanup(tools.processes.close)
        tools.attach_reasoning(Client())
        runner = tools.registry["delegate"].handler.__self__.runner
        execute = runner.execute_workspace

        def track(call, context):
            executed.append(call["function"]["name"])
            return execute(call, context)

        runner.execute_workspace = track
        started = time.monotonic()
        result = json.loads(
            tools.execute(
                tool_call(
                    "delegate", {"tasks": [{"task": "Inspect source", "expected_output": "report"}]}
                ),
                deadline=started + 0.4,
            )
        )
        try:
            self.assertTrue(entered.is_set())
            self.assertLess(time.monotonic() - started, 1)
            self.assertEqual(result["status"], "timeout")
            report = result["data"]["reports"][0]
            self.assertEqual(report["data"]["tool_calls"], 1)
            self.assertEqual(report["data"]["model_requests"], 2)
            self.assertEqual(report["data"]["report"], "draft finding")
            self.assertEqual(report["data"]["checks"][0]["status"], "ok")
            path = Path(report["data"]["checkpoint_path"])
            state = json.loads(path.read_text())
            self.assertIn("verified source evidence", state["tool_observations"][0]["result"])
            self.assertEqual(state["completed_tool_call_ids"], ["read-1"])
            original = path.read_text()
        finally:
            release.set()
        self.assertTrue(finished.wait(1))
        self.assertEqual(executed, ["read"])
        self.assertFalse((self.root / "late.txt").exists())
        self.assertEqual(path.read_text(), original)

    def test_shared_cancellation_keeps_native_partial_when_progress_callback_received_nothing(self):
        from deepseek_client import CompletionError
        from reasoning import SubagentRunner
        from tool_runtime import ExecutionContext

        execution = ExecutionContext(time.monotonic() + 60)
        requests = []

        class Client:
            def complete(inner, messages, **kwargs):
                requests.append(messages)
                execution.cancelled.set()
                raise CompletionError(
                    "incomplete_stream",
                    "fixture interruption",
                    partial_message={"content": "native interrupted finding"},
                )

        client = Client()
        client.settings = self.settings
        runner = SubagentRunner(client, lambda *args: None, [], self.root)
        result = runner.run("Inspect", "context", "report", execution)
        state = json.loads(Path(result.data["checkpoint_path"]).read_text())
        self.assertEqual(len(requests), 1)
        self.assertEqual(result.status, "timeout")
        self.assertEqual(result.data["report"], "native interrupted finding")
        self.assertIn("native interrupted finding", json.dumps(state["messages"]))
        self.assertEqual(state["unfinished_response"]["content"], "native interrupted finding")

    def test_successful_worker_creates_no_recovery_files_and_keeps_request_prefix(self):
        from reasoning import SubagentRunner
        from tool_runtime import ExecutionContext

        requests = []

        class Client:
            def complete(inner, messages, **kwargs):
                requests.append(json.loads(json.dumps(messages)))
                if len(requests) == 1:
                    return {
                        "role": "assistant",
                        "tool_calls": [tool_call("read", {"path": "source.txt"})],
                    }
                return {"role": "assistant", "content": "full report"}

        client = Client()
        client.settings = self.settings
        output = json.dumps(ToolResult(data={"content": "source"}).as_dict())
        runner = SubagentRunner(
            client, lambda *args: output, [{"function": {"name": "read"}}], self.root
        )
        result = runner.run(
            "Inspect", "context", "full report", ExecutionContext(time.monotonic() + 10)
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["report"], "full report")
        self.assertEqual(requests[0], requests[1][: len(requests[0])])
        self.assertEqual(
            list((self.root / ".agent-artifacts").glob("*.worker-checkpoint.json")), []
        )
        self.assertNotIn("checkpoint_path", result.data)


if __name__ == "__main__":
    unittest.main()
