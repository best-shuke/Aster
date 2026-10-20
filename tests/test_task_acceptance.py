import json
import os
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import file_state
from task_acceptance import TaskAcceptance, TaskContract, parse_shell, standalone_checks
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

    def test_read_only_tools_skip_the_walk_but_changes_made_meanwhile_still_count(self):
        (self.workspace / "app.py").write_text("answer = 41\n")
        self.acceptance = TaskAcceptance(
            "修复代码",
            TaskContract("change"),
            workspace=self.workspace,
            passive_tools=frozenset({"read", "grep"}),
        )
        walks = []
        refresh = self.acceptance._refresh_files
        self.acceptance._refresh_files = lambda: walks.append(1) or refresh()
        self.execute("read", {"path": "app.py"})
        self.execute("grep", {"pattern": "answer"})
        self.assertEqual(walks, [])
        # Something else changes the code while only read-only tools run.
        (self.workspace / "app.py").write_text("answer = 42\n")
        self.execute("read", {"path": "app.py"})
        self.assertIn("no successful verification", self.acceptance.validate("已完成。"))
        # The walk before a check finds the change, so the check covers it.
        self.execute("exec", self.command("from app import answer; assert answer == 42"))
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

    def test_node_test_runner_and_syntax_check_are_verification(self):
        self.assertTrue(self.acceptance.verification_command("python manage.py test shop"))
        for command in ("node --test tests/", "node --check game.js", "node -c a.js"):
            with self.subTest(command=command):
                self.assertTrue(self.acceptance.verification_command(command))
        # Bare forms pass with nothing to check: no test files, or empty stdin.
        for command in (
            "node --test",
            "node --check",
            "node game.js",
            "node -e \"console.log('ok')\"",
            "node --version",
        ):
            with self.subTest(command=command):
                self.assertFalse(self.acceptance.verification_command(command))

    def test_a_shared_digest_cache_spares_later_tasks_hashing_unchanged_files(self):
        for index in range(5):
            (self.workspace / f"module_{index}.py").write_text(f"value = {index}\n")
        cache = {}
        # Digests are cached once a file's timestamps are older than the skew.
        skew = patch.object(file_state, "_DIGEST_CACHE_SKEW_NS", -(10**12))
        skew.start()
        self.addCleanup(skew.stop)
        TaskAcceptance(
            "修复代码", TaskContract("change"), workspace=self.workspace, digest_cache=cache
        )
        with patch("file_state.file_digest", wraps=file_state.file_digest) as digest:
            TaskAcceptance(
                "修复代码", TaskContract("change"), workspace=self.workspace, digest_cache=cache
            )
        self.assertEqual(digest.call_count, 0)

    def test_only_unquoted_operators_split_a_command_line(self):
        command = "python -c \"print('a | b; c && d')\" && python -c 'assert 1 << 2 == 4'"
        self.assertEqual(
            [(item.operator, item.words[-1]) for item in parse_shell(command)],
            [(None, "print('a | b; c && d')"), ("&&", "assert 1 << 2 == 4")],
        )
        self.assertEqual(
            [item.words for item in parse_shell("pytest -n 4 >log 2>&1 && echo '&&'")],
            [("pytest", "-n", "4"), ("echo", "&&")],
        )
        heredoc = "python3 - <<'PY'\nassert 1\nPY\necho a\necho b"
        self.assertEqual(
            [(item.operator, item.words, item.stdin) for item in parse_shell(heredoc)],
            [
                (None, ("python3", "-"), "assert 1\n"),
                (";", ("echo", "a"), None),
                (";", ("echo", "b"), None),
            ],
        )
        # A command substitution runs a command the line does not show: no check.
        self.assertFalse(TaskAcceptance.shell_checks("x=$(diff a b) && true", self.workspace)[0])

    def test_a_rerun_keeps_the_check_as_written_with_its_setup(self):
        self.assertEqual(
            standalone_checks(
                'cd ~ && FOO="a b" pytest tests/test_*.py 2>&1 | tail -3; echo "$?"',
                self.workspace,
            ),
            ['cd ~ && FOO="a b" pytest tests/test_*.py 2>&1'],
        )
        heredoc = "python3 - <<'PY' | tail -2\nassert 1\nPY"
        self.assertEqual(
            standalone_checks(heredoc, self.workspace), ["python3 - <<'PY'\nassert 1\nPY"]
        )
        # Writing a file through a here-document is not setup to rerun.
        self.assertEqual(
            standalone_checks(
                "cat > t.py <<EOF && python -m pytest t.py | tail\ndef test(): pass\nEOF",
                self.workspace,
            ),
            ["python -m pytest t.py"],
        )

    def test_node_programs_using_node_assert_are_verification(self):
        program = "import assert from 'node:assert/strict'; assert.equal(1, 1)"
        for command in (
            f"node --input-type=module -e {shlex.quote(program)}",
            f"node --input-type=module - <<'EOF'\n{program}\nEOF",
            "node -e \"const { strictEqual } = require('assert'); strictEqual(2, 2)\"",
        ):
            with self.subTest(command=command):
                self.assertEqual(len(TaskAcceptance.shell_checks(command, self.workspace)[0]), 1)
        # console.assert only logs, and a program without node:assert checks nothing.
        for command in ('node -e "console.assert(false)"', 'node -e "console.log(1)"'):
            with self.subTest(command=command):
                self.assertFalse(TaskAcceptance.shell_checks(command, self.workspace)[0])

    def test_deleting_an_unused_scratch_check_keeps_the_verification(self):
        self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
        self.execute("write", {"path": "verify_tmp.py", "content": "assert True\n"})
        quoted = shlex.quote(sys.executable)
        self.execute("exec", {"command": f"{quoted} verify_tmp.py"})
        self.assertIsNone(self.acceptance.validate("已完成并验证。"))
        self.execute("exec", {"command": "rm verify_tmp.py"})
        self.assertIsNone(self.acceptance.validate("已完成并验证。"))
        # A file the remaining code uses is no scratch file.
        self.execute("write", {"path": "helper.py", "content": "value = 42\n"})
        self.execute("write", {"path": "app.py", "content": "from helper import value\n"})
        self.execute("exec", self.command("from app import value; assert value == 42"))
        self.assertIsNone(self.acceptance.validate("已完成并验证。"))
        self.execute("exec", {"command": "rm helper.py"})
        self.assertIn("helper.py", self.acceptance.validate("已完成。"))

    def test_deleting_a_file_that_existed_before_the_task_needs_a_new_check(self):
        (self.workspace / "legacy.py").write_text("old = True\n")
        self.acceptance = TaskAcceptance(
            "修复代码", TaskContract("change"), workspace=self.workspace
        )
        self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
        self.execute("exec", self.command("from app import answer; assert answer == 42"))
        self.execute("exec", {"command": "rm legacy.py"})
        self.assertIn("legacy.py", self.acceptance.validate("已完成。"))

    def test_an_unverified_change_names_the_files_changed_since_the_last_check(self):
        self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
        self.execute("exec", self.command("from app import answer; assert answer == 42"))
        self.execute("write", {"path": "scratch_check.py", "content": "assert True\n"})
        self.execute("exec", {"command": "rm scratch_check.py"})
        problem = self.acceptance.validate("已完成。")
        self.assertIn("Changed since the last counted check", problem)
        self.assertIn("scratch_check.py", problem)
        self.assertNotIn("app.py", problem)

    def test_a_malformed_command_is_no_check_and_does_not_fail(self):
        command = [sys.executable, "-m", "unittest"]
        self.acceptance.observe(call("exec", {"command": command}), {"status": "error"})
        self.acceptance.observe(call("write", {"path": ["app.py"]}), {"status": "ok"})
        self.assertFalse(self.acceptance.verification_command(command))
        self.assertIsNone(TaskAcceptance.shell_segments(command))
        self.assertEqual(self.acceptance.last_own_edit, -1)

    def test_a_background_check_without_monitor_is_no_check(self):
        self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
        self.execute("exec", {**self.command("assert True"), "background": True, "timeout": 30})
        self.assertIn("monitor", self.acceptance.verification_problem())
        self.assertIn(
            "no successful verification",
            self.acceptance.verification_problem(monitored=False),
        )

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

    def test_descriptor_redirection_is_not_shell_composition(self):
        self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
        arguments = self.command("from app import answer; assert answer == 42")
        arguments["command"] += " 2>&1"
        self.execute("exec", arguments)
        self.assertIsNone(self.acceptance.validate("已完成并验证。"))

    def test_an_uncounted_composed_check_is_explained(self):
        self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
        arguments = self.command("from app import answer; assert answer == 42")
        arguments["command"] += " | cat"
        self.execute("exec", arguments)
        problem = self.acceptance.validate("已完成并验证。")
        self.assertIn("was not counted", problem)
        self.assertIn("cwd argument", problem)

    def test_a_check_in_an_and_list_is_proven_by_its_status(self):
        # "&&" runs each command only after the previous one succeeded: status 0 proves all.
        self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
        check = self.command("from app import answer; assert answer == 42")["command"]
        for command in (f"cd . && {check}", f"{check} && git --version | cat"):
            with self.subTest(command=command):
                self.acceptance.verified_revision = -1
                self.execute("exec", {"command": command})
                self.assertIsNone(self.acceptance.validate("已完成并验证。"))
        failing = self.command("assert False")["command"]
        self.execute("exec", {"command": f"true && {failing}"})
        self.assertIn("no successful verification", self.acceptance.validate("已完成。"))

    def test_a_failed_composed_check_leaves_no_failure_that_cannot_be_cleared(self):
        # "pytest || unittest" can never pass as a trusted check, so its failure must not
        # be what a later standalone check has to clear.
        self.execute("write", {"path": "app.py", "content": "answer = 41\n"})
        quoted = shlex.quote(sys.executable)
        failing = self.command("from app import answer; assert answer == 42")["command"]
        self.execute("exec", {"command": f"{quoted} -m no_such_runner || {failing}"})
        self.assertFalse(self.acceptance.failed_verifications)
        # Another size than before, so Python cannot reuse the stale bytecode.
        self.execute("write", {"path": "app.py", "content": "answer = 42  # fixed\n"})
        self.execute("exec", self.command("from app import answer; assert answer == 42"))
        self.assertIsNone(self.acceptance.validate("已完成并验证。"))

    def test_a_missing_test_runner_is_no_failed_check(self):
        self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
        missing = (
            ("python -m pytest -q", {"exit_code": 1, "output": "python: No module named pytest\n"}),
            ("pytest -q", {"exit_code": 127, "output": "sh: pytest: command not found\n"}),
            ("npm test", {"exit_code": 1, "output": 'npm error Missing script: "test"\n'}),
        )
        for command, data in missing:
            with self.subTest(command=command):
                invocation = call("exec", {"command": command})
                self.acceptance.before_tool(invocation)
                self.acceptance.observe(invocation, {"status": "error", "data": data})
                self.assertFalse(self.acceptance.failed_verifications)
        self.execute("exec", self.command("from app import answer; assert answer == 42"))
        self.assertIsNone(self.acceptance.validate("已完成并验证。"))
        # A runner that is there and fails is a failed check.
        invocation = call("exec", {"command": "pytest -q"})
        self.acceptance.observe(
            invocation, {"status": "error", "data": {"exit_code": 1, "output": "1 failed"}}
        )
        self.assertIn("unresolved", self.acceptance.validate("已完成。"))

    def test_a_failed_check_is_named_and_cleared_by_rerunning_it_however_verbose(self):
        self.execute("write", {"path": "app.py", "content": "answer = 41\n"})
        (self.workspace / "tests").mkdir()
        (self.workspace / "tests" / "test_app.py").write_text(
            "import unittest\nfrom app import answer\n\n\nclass T(unittest.TestCase):\n"
            "    def test(self):\n        self.assertEqual(answer, 42)\n"
        )
        quoted = shlex.quote(sys.executable)
        self.execute("exec", {"command": f"{quoted} -m unittest discover -s tests"})
        problem = self.acceptance.validate("已完成。")
        self.assertIn("unittest discover -s tests`", problem)
        self.execute("write", {"path": "app.py", "content": "answer = 42  # fixed\n"})
        self.execute("exec", {"command": f"{quoted} -m unittest discover -s tests -v 2>&1"})
        self.assertIsNone(self.acceptance.validate("已完成并验证。"))

    def test_a_corrected_inline_script_settles_the_failure_of_its_first_version(self):
        self.execute("write", {"path": "app.py", "content": "def double(x):\n    return 2 * x\n"})
        quoted = shlex.quote(sys.executable)
        script = (
            "from app import double\n"
            "assert double(0) == 0\n"
            "assert double(2) == 4\n"
            "assert double(-3) == {expected}\n"
            "print('all checks passed')\n"
        )
        # The script's own expectation was wrong, not the code.
        self.execute("exec", {"command": f"{quoted} - <<'PY'\n{script.format(expected=6)}PY"})
        problem = self.acceptance.validate("已完成。")
        self.assertIn("unresolved", problem)
        self.assertIn("corrected version", problem)
        # An unrelated passing script does not settle it.
        self.execute("exec", self.command("import json; assert json.loads('1') == 1"))
        self.assertIn("unresolved", self.acceptance.validate("已完成。"))
        self.execute("exec", {"command": f"{quoted} - <<'PY'\n{script.format(expected=-6)}PY"})
        self.assertIsNone(self.acceptance.validate("已完成并验证。"))

    def test_after_a_code_change_only_the_failed_script_itself_settles_its_failure(self):
        # A weaker script after a fix is no rerun of the check that failed before it.
        self.execute("write", {"path": "app.py", "content": "answer = 41\n"})
        failing = self.command("from app import answer; assert answer == 42")
        self.execute("exec", failing)
        self.execute("write", {"path": "app.py", "content": "answer = 42  # fixed\n"})
        self.execute("exec", self.command("from app import answer; assert answer"))
        self.assertIn("unresolved", self.acceptance.validate("已完成。"))
        self.execute("exec", failing)
        self.assertIsNone(self.acceptance.validate("已完成并验证。"))

    def test_assertions_from_a_heredoc_or_a_test_script_are_verification(self):
        self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
        quoted = shlex.quote(sys.executable)
        program = "from app import answer\nassert answer == 42\n"
        self.execute("exec", {"command": f"FLAG=1 {quoted} - <<'PY'\n{program}PY"})
        self.assertIsNone(self.acceptance.validate("已完成并验证。"))
        self.execute("write", {"path": "app.py", "content": "answer = 43\n"})
        self.execute("exec", {"command": f"{quoted} - <<'PY'\nprint('answer')\nPY"})
        self.assertIn("no successful verification", self.acceptance.validate("已完成。"))
        (self.workspace / "check_app.py").write_text(
            "import unittest\n\n\nclass T(unittest.TestCase):\n    def test(self):\n"
            "        pass\n\n\nif __name__ == '__main__':\n    unittest.main()\n"
        )
        self.execute("exec", {"command": f"{quoted} check_app.py"})
        self.assertIsNone(self.acceptance.validate("已完成并验证。"))
        self.execute("write", {"path": "app.py", "content": "answer = 44\n"})
        self.execute("exec", {"command": f"{quoted} app.py"})
        self.assertIn("no successful verification", self.acceptance.validate("已完成。"))

    def test_work_continuing_in_the_background_is_not_verified_yet(self):
        # A teammate's unfinished edit (not this agent's tool call) changed the code.
        self.execute("ls", {})
        (self.workspace / "app.py").write_text("answer = 41\n")
        self.assertIn("no successful verification", self.acceptance.validate("已安排。"))
        self.assertIsNone(self.acceptance.validate("已安排，成员在后台继续。", background=True))
        self.assertEqual(self.acceptance.status, "background")

    def test_own_edits_need_verification_even_when_work_continues(self):
        self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
        self.assertIn(
            "verify your own change", self.acceptance.validate("已安排。", background=True)
        )
        self.execute("exec", self.command("from app import answer; assert answer == 42"))
        (self.workspace / "other.py").write_text("unfinished = True\n")
        self.assertIsNone(self.acceptance.validate("已安排。", background=True))
        self.assertEqual(self.acceptance.status, "background")

    def test_a_failed_monitored_check_does_not_verify_own_edits_for_the_background(self):
        self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
        started = self.execute("exec", {**self.command("input(); assert True"), "background": True})
        self.execute("monitor", {"action": "stop", "session_id": started["data"]["session_id"]})
        self.assertIn(
            "verify your own change", self.acceptance.validate("已安排。", background=True)
        )

    def test_a_check_started_before_an_own_edit_does_not_verify_it_for_the_background(self):
        self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
        arguments = {**self.command("input(); assert True"), "background": True}
        session_id = self.execute("exec", arguments)["data"]["session_id"]
        self.execute("edit", {"path": "app.py", "edits": [{"oldText": "42", "newText": "43"}]})
        self.execute("monitor", {"action": "input", "session_id": session_id, "input": "\n"})
        self.execute("monitor", {"action": "wait", "session_id": session_id})
        self.assertIn(
            "verify your own change", self.acceptance.validate("已安排。", background=True)
        )
        self.execute("exec", self.command("assert True"))
        self.assertIsNone(self.acceptance.validate("已安排。", background=True))

    def test_editing_a_document_needs_no_check_when_work_continues(self):
        self.execute("write", {"path": "notes.md", "content": "plan\n"})
        self.assertIsNone(self.acceptance.validate("已安排。", background=True))
        self.assertEqual(self.acceptance.status, "background")

    def test_a_command_that_cannot_be_lexed_is_no_trusted_check(self):
        # The shell ignores the comment, so "; true" would hide a failing pytest.
        self.assertIsNone(TaskAcceptance.shell_segments("pytest tests; true #it's"))
        self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
        quoted = shlex.quote(sys.executable)
        self.execute("exec", {"command": f"{quoted} -m pytest --co -q; true #it's"})
        self.assertNotEqual(self.acceptance.verified_revision, self.acceptance.code_revision)

    def test_quoted_redirect_characters_do_not_hide_backgrounding(self):
        self.assertEqual(
            TaskAcceptance.shell_segments("pytest -k 'a>' & true"),
            [["pytest", "-k", "a>"], ["true"]],
        )
        for command in ("pytest 2>&1", "pytest >&2", "pytest &>log", "pytest > out 2>&1"):
            with self.subTest(command=command):
                self.assertEqual(len(TaskAcceptance.shell_segments(command)), 1)


class TaskKindTests(unittest.TestCase):
    def test_research_and_writing_that_mention_implementations_are_no_code_change(self):
        for prompt in (
            "调研 RAG 的主流实现方案并写一份报告",
            "比较三篇论文提出的方法在实现上的差异，整理成文献综述",
            "帮我调研一下新增的欧盟 AI 法案条款对初创公司的影响",
            "Research how vector databases implement HNSW and write a report",
        ):
            with self.subTest(prompt=prompt):
                self.assertEqual(TaskAcceptance(prompt).kind, "answer")

    def test_code_requests_still_need_verification(self):
        for prompt in (
            "修复 stats.py 的测试失败",
            "实现一个 LRU 缓存并加测试",
            "fix the bug in report.py",
            "implement the researcher_tool module",
            "fix the survey form validation bug",
            "修复论文管理系统的登录 bug",
            "实现一个文献检索的 API 并加测试",
            "撰写单元测试并修复 utils.py 的 bug",
            "implement the research dashboard page",
        ):
            with self.subTest(prompt=prompt):
                self.assertEqual(TaskAcceptance(prompt).kind, "change")


class HandoffTests(unittest.TestCase):
    def test_only_an_explicit_request_lets_delegated_work_run_on(self):
        for prompt in (
            "修复 app.py：交给 engineer，不要等待它完成",
            "让团队去做，不用等结果",
            "Build it with the team, don't wait for them",
            "Let the members continue in the background",
            "不需要等待他们完成，先回复我",
            "先回复我，不用等",
            "Don’t wait for the team",
            "dont wait for them",
        ):
            with self.subTest(prompt=prompt):
                self.assertTrue(TaskAcceptance(prompt).handoff_allowed)
        for prompt in (
            "做一个五子棋网页",
            "生成一个后台管理系统",
            "Build a gomoku web page",
            "Start the dev server in the background and test the page",
            "做一个支持人脸识别等功能的网页",
            "解释 TCP 和 UDP 的区别等。",
            "代码块不用等宽字体",
            "实现一个游戏，玩家不要等太久才能出牌",
        ):
            with self.subTest(prompt=prompt):
                self.assertFalse(TaskAcceptance(prompt).handoff_allowed)


class BlockerAttributionTests(unittest.TestCase):
    """Only the agent's own inability marks a task blocked, not a teammate's."""

    def acceptance(self, others=()):
        acceptance = TaskAcceptance("检查团队结果", TaskContract("answer"), others=others)
        acceptance.evidence.append({"tool": "exec", "status": "ok"})
        return acceptance

    def status(self, text, others=("critic", "engineer")):
        acceptance = self.acceptance(others)
        self.assertIsNone(acceptance.validate(text))
        return acceptance.status

    def test_own_inability_is_a_blocker(self):
        for text in (
            "无法完成，测试失败。",
            "由于缺少依赖，无法执行测试。",
            "我已经修复了一半，但我无法继续：缺少权限。",
            "I fixed the parser. I cannot proceed without the API key.",
            "Unable to complete the migration: the database is read-only.",
            "需要你确认 n=1 时的行为。",
            # 之前 here means "until"; 其他 is not the pronoun 他; critic is an object.
            "在你确认之前，我无法继续。",
            "其他测试无法执行，所以没有验证。",
            "我让 critic 运行测试，但无法完成验证。",
            "I asked the critic to run them, but cannot proceed.",
            # The user and critic only say where the file and tests came from.
            "用户提供的文件无法执行。",
            "critic 写的测试无法执行，缺少依赖。",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.status(text), "blocked")

    def test_a_teammates_limitation_is_not_the_agents_blocker(self):
        for text in (
            # The live run's false positive: the answer reports the critic's gap.
            "已修复 stats.py，4/4 测试通过。critic 无法执行测试，这一验证缺口由我重新运行测试补上。",
            "critic只有只读工具，无法执行测试；我已重新运行并通过。",
            "评审者无法执行测试，我补跑了全部测试。",
            "The critic is read-only and cannot proceed with running tests; I ran them: 4 passed.",
            "engineer 报告“无法完成迁移”，随后改用备用方案并已完成。",
            "Earlier I was blocked by a missing fixture; I added it and all tests pass.",
            "起初无法执行测试，安装 pytest 后全部通过。",
            "critic 的环境无法执行测试，我补跑了全部测试。",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.status(text), "completed")

    def test_unknown_names_still_count_as_the_agents_own_blocker(self):
        self.assertEqual(self.status("pytest 无法执行，缺少依赖。", others=()), "blocked")
        self.assertEqual(self.status("critic 无法执行测试。", others=()), "blocked")
        self.assertEqual(self.status("critic 无法执行测试。"), "completed")
