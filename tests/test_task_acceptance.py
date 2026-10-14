import json
import os
import shlex
import sys
import tempfile
import unittest
from pathlib import Path

from task_acceptance import TaskAcceptance, TaskContract
from tools import CodingTools


def call(name, arguments):
    return {
        "id": "acceptance-call",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class TaskAcceptanceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.workspace = Path(self.temporary.name).resolve()
        self.tools = CodingTools(self.workspace)
        self.addCleanup(self.tools.close)
        self.acceptance = TaskAcceptance(
            "修复代码", TaskContract("change"), workspace=self.workspace
        )

    def execute(self, name, arguments):
        invocation = call(name, arguments)
        self.acceptance.before_tool(invocation)
        result = json.loads(self.tools.execute(invocation))
        self.acceptance.observe(invocation, result)
        return result

    def command(self, program):
        return {"command": f"{shlex.quote(sys.executable)} -c {shlex.quote(program)}"}

    def test_failed_latest_check_invalidates_previous_success(self):
        self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
        arguments = self.command("from app import answer; assert answer == 42")
        self.execute("exec", arguments)
        self.assertIsNone(self.acceptance.validate("已完成并验证。"))
        failed = call("exec", arguments)
        self.acceptance.before_tool(failed)
        self.acceptance.observe(failed, {"status": "error", "error": {"code": "nonzero_exit"}})
        self.assertIn("unresolved", self.acceptance.validate("已完成并验证。"))
        self.assertEqual(self.acceptance.status, "in_progress")
        self.execute("exec", self.command("assert True"))
        self.assertIn("unresolved", self.acceptance.validate("已完成并验证。"))
        self.execute("exec", arguments)
        self.assertIsNone(self.acceptance.validate("已完成并验证。"))

    def test_background_verification_requires_observing_the_final_result(self):
        self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
        arguments = {**self.command("input(); assert True"), "background": True}
        started = self.execute("exec", arguments)
        session_id = started["data"]["session_id"]
        self.assertIn("still running", self.acceptance.validate("已完成并验证。"))
        self.execute("monitor", {"action": "input", "session_id": session_id, "input": "\n"})
        finished = self.execute("monitor", {"action": "wait", "session_id": session_id})
        self.assertEqual(finished["data"]["state"], "completed")
        self.assertIsNone(self.acceptance.validate("已完成并验证。"))

    def test_background_verification_cannot_cover_a_later_edit(self):
        self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
        arguments = {**self.command("input(); assert True"), "background": True}
        started = self.execute("exec", arguments)
        session_id = started["data"]["session_id"]
        self.execute("edit", {"path": "app.py", "edits": [{"oldText": "42", "newText": "43"}]})
        self.execute("monitor", {"action": "input", "session_id": session_id, "input": "\n"})
        self.execute("monitor", {"action": "wait", "session_id": session_id})
        self.assertIn("no successful verification", self.acceptance.validate("已完成并验证。"))

    def test_stopping_background_verification_is_not_a_successful_check(self):
        self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
        started = self.execute("exec", {**self.command("input(); assert True"), "background": True})
        self.execute("monitor", {"action": "stop", "session_id": started["data"]["session_id"]})
        self.assertIn("unresolved", self.acceptance.validate("已完成并验证。"))

    def test_shell_write_requires_a_later_check(self):
        self.execute(
            "exec",
            self.command("from pathlib import Path; Path('app.py').write_text('answer = 42\\n')"),
        )
        self.assertGreater(self.acceptance.code_revision, 0)
        self.assertIn("no successful verification", self.acceptance.validate("已修改。"))
        self.execute("exec", self.command("from app import answer; assert answer == 42"))
        self.assertIsNone(self.acceptance.validate("已修改并验证。"))

    def test_shell_delete_and_failed_shell_write_are_tracked(self):
        self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
        revision = self.acceptance.code_revision
        result = self.execute(
            "exec",
            self.command("from pathlib import Path; Path('app.py').unlink(); raise SystemExit(1)"),
        )
        self.assertEqual(result["status"], "error")
        self.assertGreater(self.acceptance.code_revision, revision)
        self.assertIn("no successful verification", self.acceptance.validate("已修改。"))

    def test_a_check_that_changes_source_cannot_verify_its_final_revision(self):
        self.execute("write", {"path": "app.py", "content": "answer = 41\n"})
        self.execute(
            "exec",
            self.command(
                "from pathlib import Path; assert True; Path('app.py').write_text('answer = 42\\n')"
            ),
        )
        self.assertIn("no successful verification", self.acceptance.validate("已完成。"))
        self.execute("exec", self.command("assert True"))
        self.assertIsNone(self.acceptance.validate("已完成并验证。"))

    def test_shell_composition_cannot_hide_a_failed_check(self):
        self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
        for suffix in (" | cat", " || true", "; true", "\ntrue", " # ignored\ntrue"):
            with self.subTest(suffix=suffix):
                arguments = self.command("assert False")
                arguments["command"] += suffix
                self.assertEqual(self.execute("exec", arguments)["status"], "ok")
                self.assertIn("no successful verification", self.acceptance.validate("已完成。"))
        self.execute("exec", self.command("assert '|' == '|'"))
        self.assertIsNone(self.acceptance.validate("已完成并验证。"))

    def test_newlines_inside_quoted_python_are_not_shell_composition(self):
        self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
        result = self.execute("exec", self.command("from app import answer\nassert answer == 42"))
        self.assertEqual(result["status"], "ok", result)
        self.assertIsNone(self.acceptance.validate("已完成并验证。"))

    def test_printed_assertions_and_help_output_are_not_verification(self):
        self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
        self.execute("exec", self.command("print('assert True')"))
        self.assertIn("no successful verification", self.acceptance.validate("已完成。"))
        self.assertFalse(self.acceptance.verification_command("pytest --collect-only"))
        self.assertFalse(self.acceptance.verification_command("ruff --version"))

    def test_user_edit_after_verification_requires_reverification(self):
        self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
        self.execute("exec", self.command("assert True"))
        (self.workspace / "app.py").write_text("answer = 43\n")
        self.assertIn("no successful verification", self.acceptance.validate("已完成。"))

    def test_verification_of_a_user_edit_before_the_check_is_accepted(self):
        self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
        (self.workspace / "app.py").write_text("answer = 43\n")
        self.execute("exec", self.command("from app import answer; assert answer == 43"))
        self.assertIsNone(self.acceptance.validate("已完成并验证。"))

    def test_same_size_edit_with_preserved_mtime_invalidates_verification(self):
        self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
        self.execute("exec", self.command("assert True"))
        target = self.workspace / "app.py"
        before = target.stat()
        target.write_text("answer = 43\n")
        os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
        self.assertIn("no successful verification", self.acceptance.validate("已完成。"))

    def test_ignored_artifacts_and_session_files_do_not_invalidate_a_check(self):
        session = self.workspace / "session.json"
        self.acceptance = TaskAcceptance(
            "修复代码", TaskContract("change"), workspace=self.workspace, ignored_paths=(session,)
        )
        self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
        self.execute("exec", self.command("assert True"))
        session.write_text("{}")
        artifact = self.workspace / ".agent-artifacts" / "generated.py"
        artifact.write_text("unverified = True")
        self.assertIsNone(self.acceptance.validate("已完成并验证。"))

    def test_external_command_working_directory_is_tracked_without_restricting_access(self):
        with tempfile.TemporaryDirectory() as external:
            arguments = self.command(
                "from pathlib import Path; Path('app.py').write_text('answer = 42\\n')"
            )
            arguments["cwd"] = external
            self.execute("exec", arguments)
            self.assertIn("no successful verification", self.acceptance.validate("已修改。"))
            arguments = self.command("from app import answer; assert answer == 42")
            arguments["cwd"] = external
            self.execute("exec", arguments)
            self.assertIsNone(self.acceptance.validate("已完成并验证。"))
