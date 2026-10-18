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

    def test_node_test_runner_and_syntax_check_are_verification(self):
        for command in (
            "node --test",
            "node --test tests/",
            "node --check game.js",
            "node -c a.js",
        ):
            with self.subTest(command=command):
                self.assertTrue(self.acceptance.verification_command(command))
        for command in ("node game.js", "node -e \"console.log('ok')\"", "node --version"):
            with self.subTest(command=command):
                self.assertFalse(self.acceptance.verification_command(command))

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
        arguments["command"] = "cd . && " + arguments["command"]
        self.execute("exec", arguments)
        problem = self.acceptance.validate("已完成并验证。")
        self.assertIn("was not counted", problem)
        self.assertIn("cwd argument", problem)

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


class HandoffTests(unittest.TestCase):
    def test_only_an_explicit_request_lets_delegated_work_run_on(self):
        for prompt in (
            "修复 app.py：交给 engineer，不要等待它完成",
            "让团队去做，不用等结果",
            "Build it with the team, don't wait for them",
            "Let the members continue in the background",
        ):
            with self.subTest(prompt=prompt):
                self.assertTrue(TaskAcceptance(prompt).handoff_allowed)
        for prompt in (
            "做一个五子棋网页",
            "生成一个后台管理系统",
            "Build a gomoku web page",
            "Start the dev server in the background and test the page",
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
