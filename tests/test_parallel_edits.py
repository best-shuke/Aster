import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from file_changes import FileTransaction
from tool_runtime import ExecutionContext
from tools import CodingTools
from ui import Console


def tool_call(name, arguments):
    return {
        "id": f"call-{name}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class ParallelEditingClient:
    def __init__(self, path, replacements, failure=None, timeout=None, reverse_completion=False):
        self.path = path
        self.original = path.read_text()
        self.replacements = replacements
        self.barrier = threading.Barrier(len(replacements))
        self.failure = failure
        self.timeout = timeout
        self.reverse_completion = reverse_completion
        self.second_finished = threading.Event()

    def complete(self, messages, tools=None, **kwargs):
        assignment = messages[1]["content"].split("Assignment:\n", 1)[1].splitlines()[0]
        index = int(assignment.split()[-1])
        completed = sum(message["role"] == "tool" for message in messages)
        old_text, new_text = self.replacements[index]
        if completed == 0:
            operation = tool_call("read", {"path": self.path.name})
        elif completed == 1:
            self.barrier.wait(timeout=5)
            operation = tool_call(
                "edit",
                {"path": self.path.name, "edits": [{"oldText": old_text, "newText": new_text}]},
            )
        elif completed == 2:
            operation = tool_call("read", {"path": self.path.name})
        else:
            draft = json.loads(messages[-1]["content"])
            if new_text not in draft["data"]["content"]:
                raise AssertionError("worker could not read its own draft")
            if self.path.read_text() != self.original:
                raise AssertionError("draft was applied before all workers finished")
            if index == self.failure:
                raise RuntimeError("simulated worker failure")
            if index == self.timeout:
                raise TimeoutError("simulated model timeout")
            if self.reverse_completion:
                if index == 0:
                    self.second_finished.wait(timeout=5)
                else:
                    self.second_finished.set()
            return {"role": "assistant", "content": f"draft {index} ready"}
        return {"role": "assistant", "content": None, "tool_calls": [operation]}


class ParallelEditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.path = self.root / "shared.txt"
        self.path.write_text("alpha\nbeta\n")
        self.tools = CodingTools(self.root)
        self.tools.worker_allow = frozenset({"write", "edit"})

    def delegate(self, client):
        self.tools.attach_reasoning(client)
        tasks = [
            {"task": f"change {index}", "expected_output": "Describe the staged changes"}
            for index in range(len(client.replacements))
        ]
        result = json.loads(self.tools.execute(tool_call("delegate", {"tasks": tasks})))
        if result["truncated"] and result["artifact_path"]:
            result = json.loads(Path(result["artifact_path"]).read_text())
        return result

    def test_merge_progress_has_clear_status_labels(self):
        self.assertEqual(Console._agent_activity({"activity": "awaiting_merge"}), "等待草稿合并")
        self.assertEqual(Console._agent_activity({"activity": "merging"}), "正在合并草稿")

    def test_two_workers_edit_different_parts_of_the_same_file(self):
        result = self.delegate(
            ParallelEditingClient(self.path, [("alpha", "new alpha"), ("beta", "new beta")])
        )
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(self.path.read_text(), "new alpha\nnew beta\n")
        for report in result["data"]["reports"]:
            self.assertEqual(report["data"]["file_changes"]["status"], "merged")

    def test_conflicting_workers_merge_in_task_order_not_completion_order(self):
        result = self.delegate(
            ParallelEditingClient(
                self.path,
                [("alpha", "first"), ("alpha", "second")],
                reverse_completion=True,
            )
        )
        self.assertEqual(result["status"], "partial", result)
        self.assertEqual(self.path.read_text(), "first\nbeta\n")
        reports = result["data"]["reports"]
        self.assertEqual(reports[0]["status"], "ok")
        self.assertEqual(reports[1]["error"]["code"], "merge_conflict")
        self.assertEqual(reports[1]["data"]["file_changes"]["status"], "conflict")
        self.assertTrue(Path(reports[1]["data"]["file_changes"]["artifact_path"]).is_file())

    def test_failed_worker_draft_is_not_applied(self):
        result = self.delegate(
            ParallelEditingClient(
                self.path, [("alpha", "new alpha"), ("beta", "failed beta")], failure=1
            )
        )
        self.assertEqual(result["status"], "partial", result)
        self.assertEqual(self.path.read_text(), "new alpha\nbeta\n")
        report = result["data"]["reports"][1]
        self.assertEqual(report["data"]["file_changes"]["status"], "not_applied")

    def test_merge_failures_preserve_drafts_and_report_original_errors(self):
        failures = (
            (TimeoutError, "timeout", "changes_not_applied", "not_applied"),
            (OSError, "error", "change_commit_failed", "failed"),
        )
        for error, status, code, change_status in failures:
            with self.subTest(error=error.__name__):
                events = []
                self.tools.on_task_update = events.append
                with patch.object(FileTransaction, "commit", side_effect=error("merge failed")):
                    result = self.delegate(
                        ParallelEditingClient(self.path, [("alpha", "new alpha")])
                    )
                report = result["data"]["reports"][0]
                self.assertEqual(result["status"], status)
                self.assertEqual(report["status"], status)
                self.assertEqual(report["error"], {"code": code, "message": "merge failed"})
                changes = report["data"]["file_changes"]
                self.assertEqual(changes["status"], change_status)
                self.assertTrue(Path(changes["artifact_path"]).is_file())
                self.assertEqual(self.path.read_text(), "alpha\nbeta\n")
                self.assertEqual(events[-1]["activity"], "finished")
                self.assertEqual(events[-1]["state"], status)

    def test_merge_progress_failure_does_not_prevent_authorized_draft_merge(self):
        events = []

        def fail_on_merge(event):
            events.append(event)
            if event.get("activity") == "merging":
                raise OSError("progress failed")

        self.tools.on_task_update = fail_on_merge
        result = self.delegate(ParallelEditingClient(self.path, [("alpha", "new alpha")]))
        report = result["data"]["reports"][0]
        self.assertEqual(result["status"], "ok")
        self.assertIsNone(report["error"])
        self.assertEqual(report["data"]["file_changes"]["status"], "merged")
        self.assertEqual(self.path.read_text(), "new alpha\nbeta\n")
        self.assertEqual(events[-1]["state"], "completed")

    def test_timed_out_worker_draft_is_not_applied(self):
        result = self.delegate(
            ParallelEditingClient(
                self.path, [("alpha", "new alpha"), ("beta", "timed out beta")], timeout=1
            )
        )
        self.assertEqual(result["status"], "partial", result)
        self.assertEqual(self.path.read_text(), "new alpha\nbeta\n")
        self.assertEqual(result["data"]["reports"][1]["status"], "timeout")

    def test_twenty_workers_edit_one_file_without_losing_any_changes(self):
        lines = [f"original_{index:02d}\n" for index in range(20)]
        self.path.write_text("".join(lines))
        result = self.delegate(
            ParallelEditingClient(
                self.path,
                [(f"original_{index:02d}", f"updated_{index:02d}") for index in range(20)],
            )
        )
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(
            self.path.read_text(), "".join(f"updated_{index:02d}\n" for index in range(20))
        )
        self.assertEqual(len(result["data"]["reports"]), 20)

    def test_staged_changes_block_exec_against_the_old_shared_workspace(self):
        self.tools.worker_allow = frozenset({"write", "exec"})
        context = ExecutionContext(time.monotonic() + 30)
        with self.tools.worker_session(context) as changes:
            staged = json.loads(
                self.tools.execute_workspace(
                    tool_call("write", {"path": "draft.txt", "content": "private"}), context
                )
            )
            self.assertTrue(staged["data"]["staged"])
            blocked = json.loads(
                self.tools.execute_workspace(
                    tool_call("exec", {"command": "touch bypass"}), context
                )
            )
            self.assertEqual(blocked["error"]["code"], "staged_changes_pending")
            self.assertFalse((self.root / "draft.txt").exists())
            self.assertFalse((self.root / "bypass").exists())
            changes.commit(context.remaining)
        self.assertEqual((self.root / "draft.txt").read_text(), "private")

    def test_worker_must_read_existing_file_before_full_overwrite(self):
        context = ExecutionContext(time.monotonic() + 30)
        with self.tools.worker_session(context) as changes:
            operation = tool_call("write", {"path": self.path.name, "content": "replacement"})
            for _ in range(2):
                blocked = json.loads(self.tools.execute_workspace(operation, context))
                self.assertEqual(blocked["error"]["code"], "read_required")
            self.tools.execute_workspace(tool_call("read", {"path": self.path.name}), context)
            accepted = json.loads(self.tools.execute_workspace(operation, context))
            self.assertTrue(accepted["data"]["staged"])
            changes.commit(context.remaining)
        self.assertEqual(self.path.read_text(), "replacement")

    def test_worker_session_keeps_parent_read_versions_separate(self):
        self.tools.execute(tool_call("read", {"path": self.path.name}))
        parent_versions = dict(self.tools.file_versions)
        context = ExecutionContext(time.monotonic() + 30)
        with self.tools.worker_session(context):
            self.assertEqual(self.tools.file_versions, {})
            blocked = self.tools.write(self.path.name, "replacement", _context=context)
            self.assertEqual(blocked.error["code"], "read_required")
            self.tools.read(self.path.name, _context=context)
            self.assertTrue(self.tools.file_versions)
        self.assertEqual(self.tools.file_versions, parent_versions)

    def test_batch_cancellation_never_commits_completed_or_late_drafts(self):
        ready, release, returned = threading.Event(), threading.Event(), threading.Event()

        class StallingClient(ParallelEditingClient):
            def complete(self, messages, tools=None, **kwargs):
                assignment = messages[1]["content"].split("Assignment:\n", 1)[1].splitlines()[0]
                completed = sum(message["role"] == "tool" for message in messages)
                if assignment.endswith("1") and completed == 3:
                    ready.set()
                    release.wait(timeout=5)
                    returned.set()
                return super().complete(messages, tools, **kwargs)

        client = StallingClient(self.path, [("alpha", "new alpha"), ("beta", "new beta")])
        self.tools.attach_reasoning(client)
        context = ExecutionContext(time.monotonic() + 30)
        results = []
        tasks = [
            {"task": f"change {index}", "expected_output": "Describe the draft"}
            for index in range(2)
        ]
        delegate = self.tools.registry["delegate"].handler
        coordinator = threading.Thread(
            target=lambda: results.append(delegate(tasks, _context=context))
        )
        coordinator.start()
        try:
            self.assertTrue(ready.wait(timeout=5))
            context.cancelled.set()
            coordinator.join(timeout=5)
            self.assertFalse(coordinator.is_alive())
            self.assertNotEqual(results[0].status, "ok")
            self.assertEqual(self.path.read_text(), "alpha\nbeta\n")
            self.assertTrue(list((self.root / ".agent-artifacts").glob("*.changes.json")))
        finally:
            release.set()
            coordinator.join(timeout=5)
        self.assertTrue(returned.wait(timeout=5))
        self.assertEqual(self.path.read_text(), "alpha\nbeta\n")
