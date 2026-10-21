import io
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import main
from conversation import Conversation
from deepseek_client import CompletionError, DeepSeekClient
from sessions import SessionConflictError, SessionStore
from tests.support import make_settings
from tests.test_conversation import FakeClient
from tool_runtime import ExecutionContext, validate, validate_tool_calls
from tools import CodingTools
from ui import Console


def call(name, arguments, identity="fixes-call"):
    return {
        "id": identity,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class EnvFileTests(unittest.TestCase):
    def test_common_env_file_syntax_is_read_as_written(self):
        from config import load_env

        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / ".env"
            path.write_text(
                "\ufeffDEEPSEEK_API_KEY=sk-first\n"
                "export DEEPSEEK_MODEL=deepseek-chat  # default model\n"
                "DEEPSEEK_BASE_URL='https://api.example.com/v1' # quoted\n"
                'DEEPSEEK_PROMPT_FILE="prompt #1.md"\n'
                "DEEPSEEK_SESSION_FILE=a#b.json\n"
                "# DEEPSEEK_MAX_SUBAGENTS=3\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {}, clear=True):
                load_env(path)
                loaded = dict(os.environ)
        self.assertEqual(
            loaded,
            {
                "DEEPSEEK_API_KEY": "sk-first",
                "DEEPSEEK_MODEL": "deepseek-chat",
                "DEEPSEEK_BASE_URL": "https://api.example.com/v1",
                "DEEPSEEK_PROMPT_FILE": "prompt #1.md",
                "DEEPSEEK_SESSION_FILE": "a#b.json",
            },
        )


class HarnessFixTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.settings = make_settings(
            api_key="test-placeholder",
            base_url="https://example.invalid",
            model="test",
            session_file=self.root / "session.json",
            max_context_chars=120000,
            keep_recent_turns=6,
            max_task_seconds=60,
            max_subagents=1,
            workspace=self.root,
        )
        self.tools = CodingTools(self.root)

    def execute(self, name, arguments):
        return json.loads(self.tools.execute(call(name, arguments)))

    def test_write_and_edit_failures_do_not_damage_the_original(self):
        target = self.root / "app.py"
        for name, arguments in (
            ("write", {"path": "app.py", "content": "replacement\n"}),
            (
                "edit",
                {"path": "app.py", "edits": [{"oldText": "original", "newText": "replacement"}]},
            ),
        ):
            for failure in ("file_state.os.fsync", "pathlib.Path.replace"):
                with self.subTest(name=name, failure=failure):
                    target.write_text("original\r\n", newline="")
                    self.execute("read", {"path": "app.py"})
                    with patch(failure, side_effect=OSError("simulated disk full")):
                        result = self.execute(name, arguments)
                    self.assertEqual(result["status"], "error")
                    self.assertEqual(target.read_bytes(), b"original\r\n")
                    self.assertEqual(list(self.root.glob("*.tmp")), [])

    def test_failed_new_write_does_not_leave_a_partial_target(self):
        with patch("file_state.os.fsync", side_effect=OSError("disk full")):
            result = self.execute("write", {"path": "new.py", "content": "new"})
        self.assertEqual(result["status"], "error")
        self.assertFalse((self.root / "new.py").exists())

    def test_atomic_edit_preserves_executable_mode_and_newlines(self):
        target = self.root / "app.py"
        target.write_bytes(b"original\r\n")
        target.chmod(0o755)
        result = self.execute(
            "edit", {"path": "app.py", "edits": [{"oldText": "original", "newText": "replacement"}]}
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(target.stat().st_mode & 0o777, 0o755)
        self.assertEqual(target.read_bytes(), b"replacement\r\n")

    def test_stale_read_cannot_overwrite_a_user_edit(self):
        target = self.root / "app.py"
        for name, arguments in (
            ("write", {"path": "app.py", "content": "agent replacement"}),
            (
                "edit",
                {"path": "app.py", "edits": [{"oldText": "original", "newText": "replacement"}]},
            ),
        ):
            with self.subTest(name=name):
                target.write_text("original\n")
                self.execute("read", {"path": "app.py"})
                target.write_text("original\nuser change\n")
                result = self.execute(name, arguments)
                self.assertEqual(result["error"]["code"], "file_changed")
                self.assertEqual(target.read_text(), "original\nuser change\n")
                self.execute("read", {"path": "app.py"})
                self.assertEqual(self.execute(name, arguments)["status"], "ok")

    def test_worker_read_does_not_replace_the_parents_read_version(self):
        target = self.root / "app.py"
        target.write_text("original")
        self.execute("read", {"path": "app.py"})
        target.write_text("user edit")
        worker = threading.Thread(target=lambda: self.execute("read", {"path": "app.py"}))
        worker.start()
        worker.join()
        result = self.execute("write", {"path": "app.py", "content": "replacement"})
        self.assertEqual(result["error"]["code"], "file_changed")
        self.assertEqual(target.read_text(), "user edit")

    def test_worker_execution_on_the_same_thread_keeps_parent_versions_separate(self):
        target = self.root / "app.py"
        target.write_text("original")
        self.execute("read", {"path": "app.py"})
        target.write_text("user edit")
        self.tools.execute_workspace(
            call("read", {"path": "app.py"}), ExecutionContext(time.monotonic() + 10)
        )
        result = self.execute("write", {"path": "app.py", "content": "replacement"})
        self.assertEqual(result["error"]["code"], "file_changed")
        self.assertEqual(target.read_text(), "user edit")

    def test_file_changed_during_atomic_write_is_preserved(self):
        target = self.root / "app.py"
        target.write_text("original")
        original_fsync = os.fsync

        def concurrent_edit(descriptor):
            original_fsync(descriptor)
            target.write_text("user edit")

        with patch("file_state.os.fsync", side_effect=concurrent_edit):
            result = self.execute(
                "edit",
                {"path": "app.py", "edits": [{"oldText": "original", "newText": "replacement"}]},
            )
        self.assertEqual(result["error"]["code"], "file_changed")
        self.assertEqual(target.read_text(), "user edit")

    def test_two_instances_cannot_overwrite_the_same_session(self):
        Conversation(self.settings, FakeClient({"role": "assistant", "content": "seed"}), None).ask(
            "seed"
        )
        first = Conversation(
            self.settings, FakeClient({"role": "assistant", "content": "first"}), None
        )
        second = Conversation(self.settings, FakeClient(), None)
        first.ask("from first")
        active = self.settings.session_file.read_bytes()
        snapshot = first.sessions.snapshot_path(first.session_id).read_bytes()
        with self.assertRaises(SessionConflictError):
            second.ask("from second")
        with self.assertRaises(SessionConflictError):
            second.clear()
        self.assertEqual(self.settings.session_file.read_bytes(), active)
        self.assertEqual(first.sessions.snapshot_path(first.session_id).read_bytes(), snapshot)

    def test_two_processes_use_a_locked_compare_and_swap(self):
        store = SessionStore(self.settings.session_file)
        store.save(
            {
                "version": 1,
                "session_id": "a" * 32,
                "messages": [{"role": "user", "content": "seed"}],
            }
        )
        program = """
import sys
from pathlib import Path
from sessions import SessionConflictError, SessionStore
store = SessionStore(Path(sys.argv[1]))
data = store.read()
data['messages'].append({'role': 'user', 'content': sys.argv[2]})
print('ready', flush=True)
sys.stdin.readline()
try:
    store.save(data)
except SessionConflictError:
    raise SystemExit(2)
"""
        processes = []
        try:
            for name in ("first", "second"):
                process = subprocess.Popen(
                    [sys.executable, "-c", program, str(self.settings.session_file), name],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                processes.append(process)
                self.assertEqual(process.stdout.readline().strip(), "ready")
            for process in processes:
                process.stdin.write("go\n")
                process.stdin.flush()
            for process in processes:
                _, errors = process.communicate(timeout=10)
                self.assertEqual(errors, "")
            self.assertEqual(sorted(process.returncode for process in processes), [0, 2])
            data = SessionStore(self.settings.session_file).read()
            self.assertEqual(len(data["messages"]), 2)
            self.assertEqual(store.read(store.snapshot_path("a" * 32)), data)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                process.communicate()

    def test_conflicting_empty_session_cannot_delete_another_instances_new_session(self):
        stale = Conversation(self.settings, FakeClient(), None)
        fresh = Conversation(
            self.settings, FakeClient({"role": "assistant", "content": "answer"}), None
        )
        fresh.ask("new question")
        with self.assertRaises(SessionConflictError):
            stale.clear()
        self.assertEqual(
            SessionStore(self.settings.session_file).read()["messages"][0]["content"],
            "new question",
        )

    def test_blocked_task_is_not_displayed_as_completed(self):
        client = FakeClient(
            {"role": "assistant", "tool_calls": [call("exec", {"command": "false"})]},
            {"role": "assistant", "content": "无法完成，测试失败。"},
        )
        conversation = Conversation(self.settings, client, self.tools)
        output = io.StringIO()
        with patch("sys.stdout", output):
            console = Console()
            main.ask(conversation, console, "修复代码")
        self.assertEqual(conversation.last_outcome["status"], "blocked")
        self.assertEqual(console._task_state, "blocked")
        self.assertIn("受阻", output.getvalue())
        self.assertNotIn("✓ 完成", output.getvalue())

    def test_conversation_rejects_completion_after_an_unverified_shell_edit(self):
        program = "from pathlib import Path; Path('app.py').write_text('answer = 42\\n')"
        write_command = f"{shlex.quote(sys.executable)} -c {shlex.quote(program)}"
        verification = "from app import answer; assert answer == 42"
        check_command = f"{shlex.quote(sys.executable)} -c {shlex.quote(verification)}"
        client = FakeClient(
            {"role": "assistant", "tool_calls": [call("exec", {"command": write_command})]},
            {"role": "assistant", "content": "已修改。"},
            {"role": "assistant", "tool_calls": [call("exec", {"command": check_command})]},
            {"role": "assistant", "content": "已修改并验证。"},
        )
        conversation = Conversation(self.settings, client, self.tools)
        published = []
        answer = conversation.ask("修复代码", on_content=published.append)
        self.assertEqual(answer, "已修改并验证。")
        self.assertEqual(published, [answer])
        self.assertEqual(len(client.requests), 4)
        self.assertGreater(conversation.last_outcome["code_revision"], 0)
        self.assertEqual(
            conversation.last_outcome["code_revision"],
            conversation.last_outcome["verified_revision"],
        )

    def test_an_uncounted_check_is_rerun_alone_instead_of_rejecting_the_answer(self):
        verification = "from app import answer; assert answer == 42"
        check = f"{shlex.quote(sys.executable)} -c {shlex.quote(verification)}"
        client = FakeClient(
            {"role": "assistant", "tool_calls": [call("write", {"path": "app.py", "content": "answer = 42\n"})]},
            {"role": "assistant", "tool_calls": [call("exec", {"command": f"{check} 2>&1 | tail -5"}, "piped")]},
            {"role": "assistant", "content": "已修改并验证。"},
        )
        conversation = Conversation(self.settings, client, self.tools)
        answer = conversation.ask("修复代码")
        self.assertEqual(answer, "已修改并验证。")
        self.assertEqual(len(client.requests), 3)
        self.assertEqual(conversation.last_outcome["status"], "completed")
        reruns = [m for m in conversation.messages if "Runtime reran" in (m.get("content") or "")]
        self.assertEqual(len(reruns), 1)
        self.assertIn(f"`{check} 2>&1` (cwd .): passed", reruns[0]["content"])

    def test_a_rerun_that_fails_rejects_the_answer_with_its_output(self):
        verification = "from app import answer; assert answer == 42, answer"
        check = f"{shlex.quote(sys.executable)} -c {shlex.quote(verification)}"
        client = FakeClient(
            {"role": "assistant", "tool_calls": [call("write", {"path": "app.py", "content": "answer = 41\n"})]},
            {"role": "assistant", "tool_calls": [call("exec", {"command": f"{check}; echo done"}, "hidden")]},
            {"role": "assistant", "content": "已修改并验证。"},
            {"role": "assistant", "tool_calls": [call("write", {"path": "app.py", "content": "answer = 42  # fixed\n"}, "fix")]},
            {"role": "assistant", "tool_calls": [call("exec", {"command": check}, "rerun")]},
            {"role": "assistant", "content": "已修复，检查通过。"},
        )
        conversation = Conversation(self.settings, client, self.tools)
        answer = conversation.ask("修复代码")
        self.assertEqual(answer, "已修复，检查通过。")
        self.assertEqual(len(client.requests), 6)
        notes = "\n".join(
            m.get("content") or "" for m in conversation.messages if m["role"] == "system"
        )
        self.assertIn("error, exit 1", notes)
        self.assertIn("AssertionError: 41", notes)
        self.assertIn("Verification failed and remains unresolved. Rerun the failed check", notes)
        self.assertEqual(conversation.last_outcome["status"], "completed")

    def test_ui_can_restore_and_finish_incomplete_or_blocked_tasks(self):
        for status in ("blocked", "incomplete"):
            with self.subTest(status=status), patch("sys.stdout", io.StringIO()):
                console = Console()
                console.restore_task("task", None, completed=False, status=status)
                self.assertEqual(console._task_state, status)
                console.task_start("task")
                console.task_end(False, status=status)
                self.assertEqual(console._task_state, status)

    def test_one_shot_exit_code_matches_task_outcome(self):
        for status, expected in (("completed", 0), ("blocked", 1), ("incomplete", 1)):
            conversation = Conversation(self.settings, FakeClient(), None)
            conversation.last_outcome = {"status": status}
            with (
                self.subTest(status=status),
                patch("main.load_settings", return_value=replace(self.settings, mcp_enabled=False)),
                patch("main.Conversation", return_value=conversation),
                patch("main.Console"),
                patch("main.ask"),
                patch("sys.argv", ["agent", "task"]),
            ):
                self.assertEqual(main.main(), expected)

    def test_invalid_tool_calls_are_rejected_before_any_write(self):
        for identity in ("", None, 12):
            malformed = call("write", {"path": "unexpected.py", "content": "bad"}, identity)
            client = FakeClient(
                {"role": "assistant", "tool_calls": [malformed]},
                {"role": "assistant", "content": "ordinary answer"},
            )
            with self.subTest(identity=identity):
                conversation = Conversation(self.settings, client, self.tools)
                self.assertEqual(conversation.ask("question"), "ordinary answer")
                self.assertFalse((self.root / "unexpected.py").exists())
                self.assertEqual(conversation.tool_runs, [])

    def test_duplicate_tool_ids_reject_the_entire_batch_before_writes(self):
        invocation = call("write", {"path": "unexpected.py", "content": "bad"})
        client = FakeClient(
            {"role": "assistant", "tool_calls": [invocation, deepcopy(invocation)]},
            {"role": "assistant", "content": "ordinary answer"},
        )
        conversation = Conversation(self.settings, client, self.tools)
        conversation.ask("question")
        self.assertFalse((self.root / "unexpected.py").exists())
        self.assertEqual(conversation.tool_runs, [])

    def test_repeated_malformed_calls_fail_without_executing_tools(self):
        malformed = {
            "role": "assistant",
            "tool_calls": [call("write", {"path": "unexpected.py", "content": "bad"}, "")],
        }
        client = FakeClient(deepcopy(malformed), deepcopy(malformed), deepcopy(malformed))
        conversation = Conversation(self.settings, client, self.tools)
        with self.assertRaises(CompletionError) as caught:
            conversation.ask("question")
        self.assertEqual(caught.exception.code, "invalid_final_response")
        self.assertFalse((self.root / "unexpected.py").exists())
        self.assertEqual(conversation.tool_runs, [])
        self.assertEqual(conversation.last_outcome["status"], "incomplete")

    def test_client_validates_complete_tool_call_structure(self):
        valid = call("ls", {})
        malformed_calls = [
            None,
            "bad",
            [None],
            [dict(valid, id="")],
            [dict(valid, id="  ")],
            [valid, deepcopy(valid)],
            [dict(valid, type="other")],
            [dict(valid, function={"name": "", "arguments": "{}"})],
            [dict(valid, function={"name": "ls", "arguments": {}})],
        ]
        for calls in malformed_calls:
            with self.subTest(calls=calls), self.assertRaises(ValueError):
                validate_tool_calls(calls)
            if calls is not None:
                with self.subTest(calls=calls), self.assertRaises(CompletionError):
                    DeepSeekClient._validate_completion(
                        {"content": "answer", "tool_calls": calls}, "stop"
                    )
        DeepSeekClient._validate_completion({"content": None, "tool_calls": [valid]}, "tool_calls")

    def test_declared_string_length_limits_are_enforced_recursively(self):
        schema = self.tools.registry["update_plan"].schema
        valid = {"title": "x" * 120, "steps": [{"step": "y" * 240, "status": "pending"}]}
        validate(valid, schema)
        for location, limit in (("title", 121), ("step", 241)):
            invalid = deepcopy(valid)
            if location == "title":
                invalid["title"] = "x" * limit
            else:
                invalid["steps"][0]["step"] = "y" * limit
            with self.subTest(location=location), self.assertRaises(ValueError):
                validate(invalid, schema)
