import io
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from conversation import Conversation
from tests.support import make_settings
from tests.test_conversation import FakeClient
from tests.test_tools import call, python_command
from tool_runtime import PAGE_CHARS, ExecutionContext
from tools import CodingTools, ToolPolicy
from ui import Console


class ProcessMonitorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name).resolve()
        self.tools = CodingTools(self.workspace)
        self.addCleanup(self.tools.close)

    def execute(self, name, arguments, **kwargs):
        return json.loads(self.tools.execute(call(name, arguments), **kwargs))

    def start(self, program, **kwargs):
        result = self.execute(
            "exec", {"command": python_command(program), "background": True, **kwargs}
        )
        self.assertEqual(result["status"], "ok", result)
        return result["data"]["session_id"]

    def monitor(self, session_id, action="status", **kwargs):
        return self.execute("monitor", {"session_id": session_id, "action": action, **kwargs})

    def test_scan_completion_continues_the_same_process(self):
        session_id = self.start(
            "import os, time\nfrom pathlib import Path\n"
            "print('WAIT_SCAN', os.getpid())\n"
            "while not Path('scanned').exists(): time.sleep(0.02)\n"
            "print('SIGNED_IN', os.getpid())\n"
            "Path('continued').write_text(str(os.getpid()))\n"
        )
        ready = self.monitor(session_id, "wait", wait_for="WAIT_SCAN", wait_seconds=2)
        self.assertTrue(ready["data"]["matched"], ready)
        self.assertEqual(ready["data"]["state"], "running")
        pid = int(ready["data"]["output"].splitlines()[0].split()[1])
        # A scan is an external state change, with no new model turn or process.
        (self.workspace / "scanned").touch()
        detected = self.monitor(session_id, "wait", wait_for="SIGNED_IN", wait_seconds=2)
        self.assertTrue(detected["data"]["matched"], detected)
        finished = self.monitor(session_id, "wait", wait_seconds=2)
        self.assertEqual(finished["data"]["state"], "completed", finished)
        self.assertEqual((self.workspace / "continued").read_text(), str(pid))
        self.assertIn(f"SIGNED_IN {pid}", finished["data"]["output"])

    def test_interrupted_monitor_preserves_the_process_and_stdin(self):
        session_id = self.start("print('WAIT_SCAN'); answer = input(); print('SIGNED_IN', answer)")
        self.monitor(session_id, "wait", wait_for="WAIT_SCAN", wait_seconds=2)
        session = self.tools.processes.lookup(session_id)
        with (
            patch.object(session.done, "wait", side_effect=KeyboardInterrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            self.monitor(session_id, "wait", wait_for="SIGNED_IN")
        self.assertIsNone(session.process.poll())
        self.monitor(session_id, "input", input="scanned\n")
        finished = self.monitor(session_id, "wait", wait_seconds=2)
        self.assertEqual(finished["data"]["exit_code"], 0, finished)
        self.assertIn("SIGNED_IN scanned", finished["data"]["output"])

    def test_short_wait_and_turn_deadline_do_not_kill_the_process(self):
        session_id = self.start("print('READY'); input(); print('DONE')")
        ready = self.monitor(session_id, "wait", wait_for="READY", wait_seconds=2)
        self.assertTrue(ready["data"]["matched"], ready)
        waiting = self.monitor(session_id, "wait", wait_for="DONE", wait_seconds=0)
        self.assertEqual(waiting["data"]["event"], "waiting")
        deadline = time.monotonic() + 0.08
        expired = self.execute(
            "monitor", {"action": "wait", "session_id": session_id}, deadline=deadline
        )
        self.assertEqual(expired["status"], "timeout")
        self.assertIsNone(self.tools.processes.lookup(session_id).process.poll())
        self.monitor(session_id, "input", input="\n")
        self.assertEqual(
            self.monitor(session_id, "wait", wait_seconds=2)["data"]["state"], "completed"
        )

    def test_background_launch_outlives_the_launching_turn_deadline(self):
        launched = self.execute(
            "exec",
            {"command": python_command("input(); print('DONE')"), "background": True},
            deadline=time.monotonic() + 0.1,
        )
        session_id = launched["data"]["session_id"]
        time.sleep(0.15)
        self.assertEqual(self.monitor(session_id)["data"]["state"], "running")
        self.monitor(session_id, "input", input="\n")
        self.assertEqual(
            self.monitor(session_id, "wait", wait_seconds=2)["data"]["state"], "completed"
        )

    def test_independent_timeout_and_nonzero_exit_are_reported(self):
        session_id = self.start("input()", timeout=1)
        result = self.monitor(session_id, "wait", wait_seconds=3)
        self.assertEqual(result["status"], "timeout", result)
        self.assertEqual(result["data"]["state"], "timed_out")
        self.assertIsNotNone(self.tools.processes.lookup(session_id).process.poll())
        failed = self.start("print('FAILED'); raise SystemExit(7)")
        result = self.monitor(failed, "wait", wait_seconds=2)
        self.assertEqual(result["status"], "error", result)
        self.assertEqual(result["data"]["exit_code"], 7)
        self.assertIn("FAILED", Path(result["artifact_path"]).read_text())

    def test_stop_and_close_clean_up_the_process_group(self):
        child = "import time; from pathlib import Path; time.sleep(0.4); Path('leaked').touch()"
        session_id = self.start(
            "import subprocess, sys\n"
            f"subprocess.Popen([sys.executable, '-c', {child!r}])\n"
            "print('READY'); input()"
        )
        self.monitor(session_id, "wait", wait_for="READY", wait_seconds=2)
        stopped = self.monitor(session_id, "stop")
        self.assertEqual(stopped["data"]["state"], "stopped")
        active = self.start("print('READY'); input()")
        self.monitor(active, "wait", wait_for="READY", wait_seconds=2)
        self.tools.close()
        self.assertEqual(self.tools.processes.lookup(active).state, "stopped")
        time.sleep(0.5)
        self.assertFalse((self.workspace / "leaked").exists())
        self.assertTrue(
            all(session.done.is_set() for session in self.tools.processes.sessions.values())
        )

    def test_output_pagination_and_marker_across_chunks(self):
        prefix = "x" * (PAGE_CHARS - 1) + "扫码完成"
        padding = 65533 - len(prefix.encode("utf-8"))
        expected = prefix + "x" * padding + "SIGNED_IN尾\n"
        session_id = self.start(
            f"print('x' * {PAGE_CHARS - 1} + '扫码完成' + 'x' * {padding} + 'SIGNED_IN尾')"
        )
        matched = self.monitor(session_id, "wait", wait_for="SIGNED_IN", wait_seconds=2)
        self.assertTrue(matched["data"]["matched"], matched)
        self.monitor(session_id, "wait", wait_seconds=2)
        offset = 0
        chunks = []
        while True:
            result = self.monitor(session_id, offset=offset)
            chunks.append(result["data"]["output"])
            offset = result["data"]["next_offset"]
            if not result["truncated"]:
                break
        self.assertEqual("".join(chunks), expected)
        self.assertEqual(offset, len(expected.encode("utf-8")))

    def test_control_characters_keep_the_session_id_and_correct_cursor(self):
        session_id = self.start("print('\\0' * 6000)")
        self.monitor(session_id, "wait", wait_seconds=2)
        offset = 0
        output = ""
        while True:
            result = self.monitor(session_id, offset=offset)
            self.assertEqual(result["data"]["session_id"], session_id)
            output += result["data"]["output"]
            offset = result["data"]["next_offset"]
            if not result["truncated"]:
                break
        self.assertEqual(output, "\0" * 6000 + "\n")

    def test_marker_split_between_waits_is_detected_with_the_new_output_cursor(self):
        session_id = self.start("print('PART_', end='', flush=True); input(); print('DONE')")
        first = self.monitor(session_id, "wait", wait_for="PART_", wait_seconds=2)
        offset = first["data"]["next_offset"]
        waiting = self.monitor(
            session_id, "wait", wait_for="PART_DONE", offset=offset, wait_seconds=0
        )
        self.assertFalse(waiting["data"]["matched"])
        self.monitor(session_id, "input", input="\n", offset=offset)
        matched = self.monitor(
            session_id, "wait", wait_for="PART_DONE", offset=offset, wait_seconds=2
        )
        self.assertTrue(matched["data"]["matched"], matched)
        repeated = self.monitor(
            session_id,
            "wait",
            wait_for="PART_DONE",
            offset=matched["data"]["next_offset"],
            wait_seconds=0,
        )
        self.assertFalse(repeated["data"]["matched"], repeated)

    def test_monitor_ui_distinguishes_waiting_from_completion(self):
        for state, event, label in (
            ("waiting", "waiting", "等待中"),
            ("running", "waiting", "等待中"),
            ("running", "matched", "条件满足"),
        ):
            with (
                self.subTest(state=state, event=event),
                patch("sys.stdout", io.StringIO()) as output,
            ):
                Console().tool_result(
                    "monitor", {"status": "ok", "data": {"state": state, "event": event}}
                )
            self.assertIn(label, output.getvalue())

    def test_file_conditions_detect_external_changes_and_reuse_the_baseline(self):
        path = self.workspace / "task-state"
        waiting = self.execute(
            "monitor",
            {
                "action": "watch",
                "path": str(path),
                "condition": "changed",
                "wait_seconds": 0,
            },
        )
        self.assertEqual(waiting["data"]["event"], "waiting")
        baseline = waiting["data"]["baseline"]

        def external_work():
            time.sleep(0.05)
            path.write_text("USER_ACTION_COMPLETE")

        writer = threading.Thread(target=external_work)
        writer.start()
        try:
            changed = self.execute(
                "monitor",
                {
                    "action": "watch",
                    "path": str(path),
                    "condition": "changed",
                    "baseline": baseline,
                    "wait_seconds": 2,
                },
            )
        finally:
            writer.join()
        self.assertEqual(changed["data"]["state"], "ready", changed)
        for condition in ("exists", "contains"):
            result = self.execute(
                "monitor",
                {
                    "action": "watch",
                    "path": str(path),
                    "condition": condition,
                    "wait_for": "USER_ACTION_COMPLETE",
                    "wait_seconds": 0,
                },
            )
            self.assertTrue(result["data"]["matched"], result)
        fresh = self.execute(
            "monitor",
            {
                "action": "watch",
                "path": str(path),
                "condition": "changed",
                "wait_seconds": 0,
            },
        )
        self.assertFalse(fresh["data"]["matched"])
        path.unlink()
        removed = self.execute(
            "monitor",
            {
                "action": "watch",
                "path": str(path),
                "condition": "changed",
                "baseline": fresh["data"]["baseline"],
                "wait_seconds": 0,
            },
        )
        self.assertTrue(removed["data"]["matched"], removed)

    def test_invalid_file_watch_and_missing_session_arguments(self):
        for arguments in (
            {"action": "watch"},
            {"action": "watch", "path": "state", "condition": "contains"},
            {"action": "status"},
            {"action": "wait", "session_id": "missing", "wait_seconds": 61},
        ):
            result = self.execute("monitor", arguments)
            self.assertEqual(result["error"]["code"], "invalid_arguments", result)

    def test_nonblocking_file_watch_checks_all_existing_content_and_rejects_fifos(self):
        path = self.workspace / "task.log"
        path.write_text("x" * 65533 + "TASK_DONE")
        result = self.execute(
            "monitor",
            {
                "action": "watch",
                "path": str(path),
                "condition": "contains",
                "wait_for": "TASK_DONE",
                "wait_seconds": 0,
            },
        )
        self.assertTrue(result["data"]["matched"], result)
        fifo = self.workspace / "fifo"
        os.mkfifo(fifo)
        result = self.execute(
            "monitor",
            {
                "action": "watch",
                "path": str(fifo),
                "condition": "contains",
                "wait_for": "TASK_DONE",
                "wait_seconds": 0,
            },
        )
        self.assertEqual(result["error"]["code"], "invalid_arguments", result)

    def test_invalid_offset_does_not_send_input_or_stop(self):
        session_id = self.start("print('READY'); answer = input(); print(answer)")
        self.monitor(session_id, "wait", wait_for="READY", wait_seconds=2)
        for action in ("input", "stop"):
            result = self.monitor(session_id, action, offset=99999, input="wrong\n")
            self.assertEqual(result["error"]["code"], "invalid_arguments")
            self.assertIsNone(self.tools.processes.lookup(session_id).process.poll())
        self.monitor(session_id, "input", input="correct\n")
        result = self.monitor(session_id, "wait", wait_seconds=2)
        self.assertIn("correct", result["data"]["output"])
        self.assertNotIn("wrong", result["data"]["output"])

    def test_listing_fork_and_unknown_sessions(self):
        session_id = self.start("input()")
        snapshot = self.tools.fork()
        self.assertIs(snapshot.processes, self.tools.processes)
        listing = json.loads(snapshot.execute(call("monitor", {"action": "list"})))
        self.assertEqual([item["session_id"] for item in listing["data"]["sessions"]], [session_id])
        self.assertEqual(self.monitor("missing")["error"]["code"], "unknown_session")

    def test_monitor_permissions_and_worker_background_restriction(self):
        self.tools.policy = ToolPolicy(allow=frozenset({"monitor"}))
        self.assertFalse(self.tools.tool_available("monitor"))
        self.assertEqual(
            self.execute("monitor", {"action": "list"})["error"]["code"], "unavailable_tool"
        )
        self.tools.policy = ToolPolicy()
        self.tools.worker_allow = frozenset({"monitor", "exec"})
        self.assertFalse(self.tools.tool_available("monitor", worker=True))
        result = json.loads(
            self.tools.execute_workspace(
                call("exec", {"command": "touch unexpected", "background": True}),
                ExecutionContext(time.monotonic() + 2),
            )
        )
        self.assertEqual(result["error"]["code"], "unavailable_tool")
        self.assertFalse((self.workspace / "unexpected").exists())

    def test_conversation_continues_after_scan_without_restarting_the_command(self):
        responses = [
            {
                "role": "assistant",
                "content": "启动扫码程序。",
                "tool_calls": [
                    call(
                        "exec",
                        {
                            "command": python_command(
                                "print('WAIT_SCAN'); input(); print('SIGNED_IN')"
                            ),
                            "background": True,
                        },
                    )
                ],
            },
        ]
        tools = self.tools

        class LoginClient(FakeClient):
            def complete(self, messages, **kwargs):
                if not self.responses:
                    result = json.loads(messages[-1]["content"])
                    session_id = result["data"]["session_id"]
                    if (
                        messages[-2].get("tool_calls", [{}])[0].get("function", {}).get("name")
                        == "exec"
                    ):
                        # Simulate the human completing the scan in the browser.
                        tools.monitor(
                            "input",
                            session_id,
                            input="\n",
                            _context=ExecutionContext(time.monotonic() + 2),
                        )
                        self.responses.append(
                            {
                                "role": "assistant",
                                "content": "等待扫码完成。",
                                "tool_calls": [
                                    call(
                                        "monitor",
                                        {
                                            "action": "wait",
                                            "session_id": session_id,
                                            "wait_for": "SIGNED_IN",
                                            "wait_seconds": 2,
                                        },
                                    )
                                ],
                            }
                        )
                    else:
                        assert result["data"]["matched"]
                        self.responses.append(
                            {"role": "assistant", "content": "扫码已完成，已继续原流程。"}
                        )
                return super().complete(messages, **kwargs)

        client = LoginClient(*responses)
        settings = make_settings(
            workspace=self.workspace,
            session_file=self.workspace / "session.json",
            api_key="test",
            base_url="https://example.invalid",
            model="test",
            max_context_chars=120000,
            keep_recent_turns=6,
            max_task_seconds=10,
            max_subagents=2,
        )
        conversation = Conversation(settings, client, tools)
        self.addCleanup(conversation.close)
        answer = conversation.ask("启动登录程序，等待扫码后继续。")
        self.assertIn("已继续", answer)
        self.assertEqual(len(tools.processes.sessions), 1)
        self.assertEqual(len(client.requests), 3)

    def test_user_input_pause_then_next_turn_reuses_the_existing_process(self):
        def pause_user_input(question, options):
            raise KeyboardInterrupt

        self.tools.attach_user_input(pause_user_input)
        client = FakeClient(
            {
                "role": "assistant",
                "content": "启动任务。",
                "tool_calls": [
                    call(
                        "exec",
                        {
                            "command": python_command("print('READY'); input(); print('DONE')"),
                            "background": True,
                        },
                    )
                ],
            },
            {
                "role": "assistant",
                "content": "等待用户操作。",
                "tool_calls": [
                    call(
                        "ask_user",
                        {
                            "question": "操作是否完成？",
                            "options": [
                                {"label": "完成", "description": "检查结果并继续。"},
                                {"label": "还没有", "description": "继续等待。"},
                            ],
                        },
                    )
                ],
            },
        )
        settings = make_settings(
            workspace=self.workspace,
            session_file=self.workspace / "session.json",
            api_key="test",
            base_url="https://example.invalid",
            model="test",
            max_context_chars=120000,
            keep_recent_turns=6,
            max_task_seconds=10,
            max_subagents=2,
        )
        conversation = Conversation(settings, client, self.tools)
        self.addCleanup(conversation.close)
        with self.assertRaises(KeyboardInterrupt):
            conversation.ask("启动任务，等待用户操作后继续。")
        session = next(iter(self.tools.processes.sessions.values()))
        self.assertIsNone(session.process.poll())
        self.monitor(session.session_id, "input", input="\n")
        client.responses.extend(
            [
                {
                    "role": "assistant",
                    "content": "检查原任务。",
                    "tool_calls": [
                        call(
                            "monitor",
                            {
                                "action": "wait",
                                "session_id": session.session_id,
                                "wait_seconds": 2,
                            },
                        )
                    ],
                },
                {"role": "assistant", "content": "操作完成，已继续并完成任务。"},
            ]
        )
        self.assertIn("已继续", conversation.ask("操作完成，请继续。"))
        self.assertEqual(len(self.tools.processes.sessions), 1)
        self.assertEqual(session.state, "completed")


if __name__ == "__main__":
    unittest.main()
