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
        if name == "exec":
            self.tools.artifacts.prepare()
        self.acceptance.before_tool(invocation)
        result = json.loads(self.tools.execute(invocation))
        self.acceptance.observe(invocation, result)
        return result

    def command(self, program):
        return {"command": f"{shlex.quote(sys.executable)} -c {shlex.quote(program)}"}

    def test_source_edits_require_verification_regardless_of_initial_intent(self):
        cases = (
            ("请先分析问题，然后修复 app.py 中的 bug", TaskContract()),
            ("Explain the bug and fix app.py", TaskContract()),
            ("帮我处理一下这个问题", TaskContract()),
            ("检查实现", TaskContract("inspect")),
            ("说明结果", TaskContract("answer")),
        )
        for index, (prompt, contract) in enumerate(cases):
            with self.subTest(prompt=prompt, contract=contract.kind):
                self.acceptance = TaskAcceptance(prompt, contract, workspace=self.workspace)
                name = f"module_{index}"
                self.assertEqual(
                    self.execute("write", {"path": f"{name}.py", "content": "answer = 42\n"})[
                        "status"
                    ],
                    "ok",
                )
                self.assertIn(
                    "no successful verification", self.acceptance.validate("已完成修改。")
                )
                self.assertNotEqual(self.acceptance.status, "completed")
                self.execute("exec", self.command(f"from {name} import answer; assert answer == 42"))
                self.assertIsNone(self.acceptance.validate("已完成修改并验证。"))
                self.assertEqual(self.acceptance.status, "completed")

    def test_initial_answer_tracks_shell_changes_even_when_the_command_fails(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                (self.workspace / "app.py").write_text("answer = 41\n")
                self.acceptance = TaskAcceptance("先分析问题再修复", workspace=self.workspace)
                self.execute("read", {"path": "app.py"})
                result = self.execute(
                    "exec",
                    self.command(
                        "from pathlib import Path; Path('app.py').write_text('answer = 42\\n')"
                        + ("; raise SystemExit(1)" if fail else "")
                    ),
                )
                self.assertEqual(result["status"], "error" if fail else "ok")
                problem = self.acceptance.validate("已完成修改。")
                self.assertIn("no successful verification", problem)
                self.assertIn("app.py", problem)
                self.execute("exec", self.command("from app import answer; assert answer == 42"))
                self.assertIsNone(self.acceptance.validate("已完成修改并验证。"))

    def test_initial_answer_tracks_source_deletions_by_external_tools(self):
        (self.workspace / "legacy.py").write_text("old = True\n")
        self.acceptance = TaskAcceptance("分析并处理旧模块", workspace=self.workspace)
        invocation = call("mcp", {"action": "call"})
        self.acceptance.before_tool(invocation)
        (self.workspace / "legacy.py").unlink()
        self.acceptance.observe(invocation, {"status": "ok", "data": {}})
        problem = self.acceptance.validate("已删除旧模块。")
        self.assertIn("no successful verification", problem)
        self.assertIn("legacy.py", problem)
        self.execute("exec", self.command("from pathlib import Path; assert not Path('legacy.py').exists()"))
        self.assertIsNone(self.acceptance.validate("已删除并验证。"))

    def test_initial_answer_cannot_verify_source_changed_by_the_check_itself(self):
        self.acceptance = TaskAcceptance("分析并处理问题", workspace=self.workspace)
        self.execute(
            "exec",
            self.command("from pathlib import Path; assert True; Path('app.py').write_text('answer = 42\\n')"),
        )
        self.assertIn("no successful verification", self.acceptance.validate("已完成。"))
        self.execute("exec", self.command("from app import answer; assert answer == 42"))
        self.assertIsNone(self.acceptance.validate("已完成并验证。"))

    def test_a_check_that_replaces_and_restores_source_does_not_pass(self):
        self.execute("write", {"path": "app.py", "content": "answer = 41\n"})
        program = (
            "from pathlib import Path; source = Path('app.py'); saved = source.read_bytes(); "
            "source.write_text('answer = 42\\n'); assert '42' in source.read_text(); "
            "source.write_bytes(saved)"
        )
        self.assertEqual(self.execute("exec", self.command(program))["status"], "ok")
        self.assertIn("source files were touched", self.acceptance.validate("Verified."))
        self.assertEqual((self.workspace / "app.py").read_text(), "answer = 41\n")
        self.execute("exec", self.command("from app import answer; assert answer == 41"))
        self.assertIsNone(self.acceptance.validate("Verified without editing."))

    def test_a_check_that_imports_then_removes_a_new_module_does_not_pass(self):
        self.execute("write", {"path": "app.py", "content": "from missing_dependency import answer\n"})
        program = (
            "from pathlib import Path; dependency = Path('missing_dependency.py'); "
            "dependency.write_text('answer = 42\\n'); from app import answer; "
            "assert answer == 42; dependency.unlink()"
        )
        self.assertEqual(self.execute("exec", self.command(program))["status"], "ok")
        self.assertFalse((self.workspace / "missing_dependency.py").exists())
        self.assertIn("source files were touched", self.acceptance.validate("Verified."))

    @unittest.skipUnless(sys.platform == "darwin" or sys.platform.startswith("linux"), "native file events required")
    def test_checks_may_rebuild_ignored_output_on_consecutive_runs(self):
        self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
        (self.workspace / "dist").mkdir()
        (self.workspace / "coverage.xml").write_text("<old/>\n")
        program = (
            "from pathlib import Path; from shutil import rmtree; from app import answer; "
            "assert answer == 42; dist = Path('dist'); rmtree(dist); dist.mkdir(); "
            "(dist / 'generated.py').write_text('answer = 42\\n'); "
            "report = Path('coverage.xml'); report.unlink(); report.write_text('<report/>\\n')"
        )
        for _ in range(2):
            self.assertEqual(self.execute("exec", self.command(program))["status"], "ok")
            self.assertIsNone(self.acceptance.validate("Verified."))

    @unittest.skipUnless(sys.platform == "darwin" or sys.platform.startswith("linux"), "native file events required")
    def test_check_may_create_an_irrelevant_report_without_rebuilding_directories(self):
        self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
        check = "from pathlib import Path; from app import answer; assert answer == 42; Path('coverage.xml').touch()"
        self.assertEqual(self.execute("exec", self.command(check))["status"], "ok")
        self.assertIsNone(self.acceptance.validate("Verified."))

    def test_ignored_output_does_not_mask_other_transient_source_changes(self):
        self.execute("write", {"path": "app.py", "content": "from dependency import answer\n"})
        program = (
            "from pathlib import Path; from shutil import rmtree; "
            "dist = Path('dist'); dist.mkdir(); rmtree(dist); "
            "dependency = Path('dependency.py'); dependency.write_text('answer = 42\\n'); "
            "from app import answer; assert answer == 42; dependency.unlink()"
        )
        self.assertEqual(self.execute("exec", self.command(program))["status"], "ok")
        self.assertIn("source files were touched", self.acceptance.validate("Verified."))

    def test_incomplete_file_events_cannot_explain_a_directory_change(self):
        self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
        program = "from pathlib import Path; assert True; Path('dist').mkdir()"
        with patch("task_acceptance.SourceEvents") as tracker:
            tracker.return_value.finish.return_value = ({(self.workspace / "dist", True, True)}, False)
            self.assertEqual(self.execute("exec", self.command(program))["status"], "ok")
        self.assertIn("source files were touched", self.acceptance.validate("Verified."))

    def test_shell_cd_failure_does_not_hide_a_later_relative_cd(self):
        with tempfile.TemporaryDirectory() as external:
            parent = Path(external).resolve()
            workspace = parent / "project"
            workspace.mkdir()
            missing = shlex.quote(str(parent / "missing" / "directory"))
            self.acceptance = TaskAcceptance("Analyze and fix", workspace=workspace)
            write_source = self.command("from pathlib import Path; Path('app.py').write_text('answer = 42\\n')")
            self.assertEqual(self.execute("exec", {
                "cwd": str(workspace), "command": f"cd {missing}; cd .. && {write_source['command']}",
            })["status"], "ok")
            self.assertTrue((parent / "app.py").is_file())
            self.assertIn("working directory could not be resolved", self.acceptance.validate("Fixed."))

    def test_shell_cd_distinguishes_logical_and_physical_symlink_parents(self):
        with tempfile.TemporaryDirectory() as external:
            root = Path(external).resolve()
            logical, physical = root / "logical", root / "physical"
            logical.mkdir()
            (physical / "child").mkdir(parents=True)
            alias = logical / "alias"
            alias.symlink_to(physical / "child", target_is_directory=True)
            write_source = self.command("from pathlib import Path; Path('app.py').write_text('answer = 42\\n')")
            for mode, parent in (("", logical), ("-L ", logical), ("-P ", physical)):
                with self.subTest(mode=mode):
                    source = parent / "app.py"
                    source.unlink(missing_ok=True)
                    self.acceptance = TaskAcceptance("Analyze and fix", workspace=self.workspace)
                    result = self.execute("exec", {"command": (
                        f"cd {mode}{shlex.quote(str(alias / '..'))} && {write_source['command']}"
                    )})
                    self.assertEqual(result["status"], "ok")
                    self.assertTrue(source.is_file())
                    problem = self.acceptance.validate("Fixed.")
                    self.assertIn(str(source) if mode else "working directory could not be resolved", problem)

    def test_shell_cd_with_inherited_symlink_pwd_cannot_skip_external_scope(self):
        with tempfile.TemporaryDirectory() as external:
            parent = Path(external).resolve()
            alias = parent / "alias"
            alias.symlink_to(self.workspace, target_is_directory=True)
            create_source = self.command("from pathlib import Path; Path('app.py').write_text('answer = 42\\n')")
            with patch.dict(os.environ, {"PWD": str(alias)}):
                result = self.execute("exec", {"command": f"cd .. && {create_source['command']}"})
            self.assertEqual(result["status"], "ok")
            self.assertTrue((parent / "app.py").is_file())
            self.assertIn("working directory could not be resolved", self.acceptance.validate("Fixed."))

    def test_runtime_created_external_directory_is_tracked(self):
        with tempfile.TemporaryDirectory() as external:
            root = Path(external).resolve() / "created"
            self.acceptance = TaskAcceptance("Analyze and fix", workspace=self.workspace)
            create_source = self.command("from pathlib import Path; Path('app.py').write_text('answer = 42\\n')")["command"]
            directory = shlex.quote(str(root))
            command = f"mkdir {directory} && cd {directory} && {create_source}"
            self.assertEqual(self.execute("exec", {"command": command})["status"], "ok")
            self.assertIn(str(root / "app.py"), self.acceptance.validate("Fixed."))
            self.execute("exec", {**self.command("from app import answer; assert answer == 42"), "cwd": str(root)})
            self.assertIsNone(self.acceptance.validate("Verified."))

    def test_runtime_computed_cd_does_not_claim_success(self):
        with tempfile.TemporaryDirectory() as external:
            self.acceptance = TaskAcceptance("Analyze and fix", workspace=self.workspace)
            create_source = self.command("from pathlib import Path; Path('app.py').write_text('answer = 42\\n')")["command"]
            command = f'ROOT=$(printf %s {shlex.quote(external)}) && cd "$ROOT" && {create_source}'
            self.assertEqual(self.execute("exec", {"command": command})["status"], "ok")
            self.assertTrue((Path(external) / "app.py").is_file())
            self.assertIn("working directory could not be resolved", self.acceptance.validate("Fixed."))
            self.assertEqual(self.acceptance.status, "incomplete")

    def test_standalone_and_composed_checks_keep_shell_local_directory(self):
        with tempfile.TemporaryDirectory(prefix="check path ") as external:
            source = Path(external) / "app.py"
            source.write_text("answer = 42\n")
            self.execute("read", {"path": str(source)})
            self.assertEqual(self.execute("write", {"path": str(source), "content": "answer = 41\n"})["status"], "ok")
            check = self.command("from app import answer; assert answer == 41")["command"]
            command = f'ROOT={shlex.quote(external)} && cd "$ROOT" && {check}'
            self.assertEqual(self.execute("exec", {"command": command + " | cat"})["status"], "ok")
            self.assertEqual(self.acceptance.rerun_candidates(), [(command, ".")])
            self.assertEqual(self.execute("exec", {"command": command})["status"], "ok")
            self.assertIsNone(self.acceptance.validate("Verified."))

    def test_retargeting_a_configured_manifest_needs_a_check(self):
        first = self.workspace / "first.json"
        second = self.workspace / "second.json"
        for source in (first, second):
            source.write_text("{}\n")
        link = self.workspace / "servers.json"
        link.symlink_to(first.name)
        self.acceptance = TaskAcceptance(
            "Change MCP config", workspace=self.workspace, configuration_paths=(link,),
        )
        self.execute("exec", self.command(
            "from pathlib import Path; link = Path('servers.json'); "
            "link.unlink(); link.symlink_to('second.json')"
        ))
        self.assertIn("servers.json", self.acceptance.validate("Config changed."))
        self.execute("exec", self.command(
            "from pathlib import Path; assert Path('servers.json').readlink().name == 'second.json'"
        ))
        self.assertIsNone(self.acceptance.validate("Checked config."))

    def test_check_cannot_change_source_in_a_shell_cd_target(self):
        with tempfile.TemporaryDirectory() as external:
            root = Path(external).resolve()
            source = root / "app.py"
            source.write_text("answer = 41\n")
            check = self.command(
                "from pathlib import Path; Path('app.py').write_text('answer = 42\\n'); assert True"
            )["command"]
            for option in ("", "-- ", "-P "):
                with self.subTest(option=option):
                    source.write_text("answer = 41\n")
                    self.acceptance = TaskAcceptance("Analyze and fix", workspace=self.workspace)
                    result = self.execute("exec", {"command": f"cd {option}{shlex.quote(str(root))} && {check}"})
                    self.assertEqual(result["status"], "ok")
                    self.assertIn(str(source), self.acceptance.validate("Checked."))

    def test_shell_cd_expands_environment_directory_for_tracking(self):
        with tempfile.TemporaryDirectory() as external, patch.dict(os.environ, {"ASTER_TEST_SOURCE": external}):
            self.acceptance = TaskAcceptance("Analyze and fix", workspace=self.workspace)
            command = self.command("from pathlib import Path; Path('app.py').write_text('answer = 2\\n')")["command"]
            self.execute("exec", {"command": f'cd "$ASTER_TEST_SOURCE" && {command}'})
            self.assertIn("app.py", self.acceptance.validate("Fixed."))

    def test_shell_cd_tracks_a_directory_assigned_inside_the_command(self):
        for prefix in ("export ", ""):
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory(prefix="external source ") as external:
                self.acceptance = TaskAcceptance("Analyze and fix", workspace=self.workspace)
                command = self.command("from pathlib import Path; Path('app.py').write_text('answer = 2\\n')")["command"]
                result = self.execute("exec", {"command": (
                    f"{prefix}ASTER_LOCAL_ROOT={shlex.quote(external)}; "
                    f'cd "${{ASTER_LOCAL_ROOT}}" && {command}'
                )})
                self.assertEqual(result["status"], "ok")
                self.assertIn("app.py", self.acceptance.validate("Fixed."))

    def test_unreplayable_directory_change_does_not_verify_the_original_cwd(self):
        with tempfile.TemporaryDirectory() as external:
            root = Path(external).resolve()
            (root / "app.py").write_text("answer = 0\n")
            self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
            check = self.command("from app import answer; assert answer == 42")["command"]
            self.execute("exec", {"command": f"cd {shlex.quote(str(root))}; {check} | cat"})
            self.assertIsNotNone(self.acceptance.validate("Checked."))
            self.assertEqual(self.acceptance.rerun_candidates(), [])

    def test_tracked_external_directory_can_be_deleted_and_verified(self):
        with tempfile.TemporaryDirectory() as external:
            root = Path(external).resolve() / "legacy"
            root.mkdir()
            (root / "app.py").write_text("answer = 0\n")
            self.execute("exec", {"cwd": str(root), "command": "pwd"})
            self.execute("exec", self.command(
                f"import shutil; from pathlib import Path; shutil.rmtree({str(root)!r}); "
                f"Path({str(root)!r}).write_text('replaced')"
            ))
            self.assertIn("app.py", self.acceptance.validate("Removed."))
            self.execute("exec", self.command(f"from pathlib import Path; assert Path({str(root)!r}).is_file()"))
            self.assertIsNone(self.acceptance.validate("Removed and verified."))
            revision = self.acceptance.verified_source()
            self.assertTrue(TaskAcceptance("", workspace=self.workspace).source_matches(revision))

    def test_contract_callbacks_must_all_pass_on_one_revision(self):
        source = self.workspace / "app.py"
        source.write_text("first = True\nsecond = False\n")
        acceptance = TaskAcceptance(
            "Inspect", TaskContract(kind="answer", checks=(lambda: None, lambda: None)),
            workspace=self.workspace,
        )
        validate = acceptance.validate_callback

        def switch_after_first(check):
            result = validate(check)
            source.write_text("first = False\nsecond = True\n")
            return result

        with patch.object(acceptance, "validate_callback", side_effect=switch_after_first):
            self.assertIn("source revision changed", acceptance.validate("All checks passed."))
        self.assertNotEqual(acceptance.status, "completed")
        self.assertFalse(acceptance.check_results)

    def test_process_write_after_resume_snapshot_is_not_rebaselined(self):
        source = self.workspace / "app.py"
        source.write_text("answer = 1\n")
        initial = TaskAcceptance("Fix app.py", TaskContract(kind="change"), workspace=self.workspace)
        resumed = TaskAcceptance("Summarize the command result", workspace=self.workspace)
        snapshot = resumed._snapshot

        def change_after_first_snapshot(*args):
            files = snapshot(*args)
            source.write_text("answer = 2\n")
            return files

        with patch.object(resumed, "_snapshot", side_effect=change_after_first_snapshot):
            resumed.resume_process("owned", initial.source_checkpoint())
        invocation = call("monitor", {"action": "wait", "session_id": "owned"})
        resumed.before_tool(invocation)
        resumed.observe(invocation, {"status": "ok", "data": {"state": "completed", "session_id": "owned"}})
        self.assertIsNotNone(resumed.validate("Done."))
        self.assertGreater(resumed.code_revision, 0)

    def test_configuration_scope_preserves_parent_segments_after_symlinks(self):
        target = self.workspace / "real" / "nested"
        target.mkdir(parents=True)
        alias = self.workspace / "alias"
        alias.symlink_to(target)
        config = alias / ".." / "servers.json"
        config.write_text("{}\n")
        self.acceptance = TaskAcceptance(
            "Change config", workspace=self.workspace, configuration_paths=(config,),
        )
        result = self.execute("exec", self.command(
            "from pathlib import Path; Path('real/servers.json').write_text('{\"change\": true}')"
        ))
        self.assertEqual(result["status"], "ok")
        self.assertIn("servers.json", self.acceptance.validate("Changed config."))

    def test_initial_answer_rejects_unverified_own_edits_during_background_handoff(self):
        self.acceptance = TaskAcceptance("分析并修改，团队后台继续", workspace=self.workspace)
        self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
        self.assertIn("verify your own change", self.acceptance.validate("已安排。", background=True))

    def test_read_only_answers_do_not_snapshot_the_workspace(self):
        (self.workspace / "app.py").write_text("answer = 42\n")
        with patch("task_acceptance.source_snapshot") as snapshot:
            self.acceptance = TaskAcceptance(
                "解释 app.py", workspace=self.workspace, passive_tools=frozenset({"read"})
            )
            self.assertEqual(self.execute("read", {"path": "app.py"})["status"], "ok")
            self.assertIsNone(self.acceptance.validate("answer 的值为 42。"))
        snapshot.assert_not_called()

    def test_writing_a_report_and_failed_edits_do_not_require_code_verification(self):
        (self.workspace / "app.py").write_text("answer = 42\n")
        self.acceptance = TaskAcceptance("调研实现并写一份报告", workspace=self.workspace)
        self.execute("read", {"path": "app.py"})
        result = self.execute(
            "edit", {"path": "app.py", "edits": [{"oldText": "absent", "newText": "43"}]}
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(
            self.execute("write", {"path": "report.md", "content": "# 研究报告\n"})["status"],
            "ok",
        )
        self.assertIsNone(self.acceptance.validate("已保存研究报告。"))
        self.assertEqual(self.acceptance.kind, "answer")

    def test_research_data_is_not_code_but_build_configuration_is(self):
        self.acceptance = TaskAcceptance("保存研究结果", workspace=self.workspace)
        self.execute("write", {"path": "results.json", "content": "{}\n"})
        self.assertIsNone(self.acceptance.validate("已保存。"))
        for path in (
            "package.json", "pyproject.toml", ".github/workflows/test.yml",
            "babel.config.json", ".eslintrc.json",
            "builtin/environment-templates/coding-team.json", ".agent/environments/test.json", "mcp.json",
        ):
            with self.subTest(path=path):
                self.acceptance = TaskAcceptance("说明结果", workspace=self.workspace)
                self.execute("write", {"path": path, "content": "{}\n"})
                self.assertIn(path, self.acceptance.validate("已完成。"))

    def test_no_op_write_does_not_invalidate_a_check_or_handoff(self):
        self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
        self.execute("exec", self.command("from app import answer; assert answer == 42"))
        self.execute("edit", {"path": "app.py", "edits": [{"oldText": "42", "newText": "42"}]})
        self.assertIsNone(self.acceptance.validate("已验证。"))
        self.assertIsNone(self.acceptance.validate("团队还在后台工作。", background=True))

    def test_plan_user_question_and_mcp_status_do_not_scan_an_answer_workspace(self):
        self.acceptance = TaskAcceptance("解释方案", workspace=self.workspace)
        with patch("task_acceptance.source_snapshot") as snapshot:
            for name, arguments in (
                ("update_plan", {}), ("ask_user", {}), ("mcp", {"action": "status"}),
            ):
                invocation = call(name, arguments)
                self.acceptance.before_tool(invocation)
                self.acceptance.observe(invocation, {"status": "ok"})
            self.assertIsNone(self.acceptance.validate("方案可行。"))
        snapshot.assert_not_called()

    def test_team_wait_changes_belong_to_an_integrating_task_not_a_report(self):
        for kind in ("answer", "change", "auto"):
            with self.subTest(kind=kind):
                self.acceptance = TaskAcceptance(
                    "先分析问题再修复 app.py", TaskContract(kind), workspace=self.workspace
                )
                invocation = call("team", {"action": "wait"})
                self.acceptance.before_tool(invocation)
                (self.workspace / f"member_{kind}.py").write_text("answer = 42\n")
                self.acceptance.observe(invocation, {"status": "ok"})
                problem = self.acceptance.validate("已经收到成员回复。")
                if kind == "answer":
                    self.assertIsNone(problem)
                else:
                    self.assertIn(f"member_{kind}.py", problem)

    def test_extending_scope_does_not_rebaseline_existing_workspace_changes(self):
        root = self.workspace / "nested"
        root.mkdir()
        path = root / "app.py"
        path.write_text("answer = 42\n")
        self.acceptance = TaskAcceptance("修复 app.py", workspace=root)
        self.execute("exec", {**self.command("from app import answer; assert answer == 42"), "cwd": str(root)})
        invocation = call("exec", {"command": "true", "cwd": str(self.workspace)})
        snapshot = self.acceptance._snapshot

        def snapshot_before_mutation(*args, **kwargs):
            files = snapshot(*args, **kwargs)
            path.write_text("answer = 0\n")
            return files

        with patch.object(self.acceptance, "_snapshot", side_effect=snapshot_before_mutation):
            self.acceptance.before_tool(invocation)
        self.acceptance.observe(invocation, {"status": "ok"})
        self.assertIn("app.py", self.acceptance.validate("已完成。"))

    def test_tracking_error_is_not_cleared_by_a_later_passing_check(self):
        self.acceptance = TaskAcceptance("说明结果", workspace=self.workspace)
        invocation = call("delegate", {"tasks": []})
        self.acceptance.before_tool(invocation)
        (self.workspace / "app.py").write_text("answer = 42\n")
        with patch("task_acceptance.source_snapshot", side_effect=PermissionError("unreadable")):
            self.acceptance.observe(invocation, {"status": "ok"})
        self.execute("exec", self.command("from app import answer; assert answer == 42"))
        self.assertIn("unreadable", self.acceptance.validate("已验证。"))
        self.assertEqual(self.acceptance.status, "incomplete")
        self.assertEqual(self.acceptance.outcome()["verified_revision"], -1)

    def test_snapshot_timeout_propagates(self):
        self.acceptance = TaskAcceptance("说明结果", workspace=self.workspace)
        with patch("task_acceptance.source_snapshot", side_effect=TimeoutError("deadline")):
            with self.assertRaisesRegex(TimeoutError, "deadline"):
                self.acceptance.before_tool(call("exec", {"command": "true"}))

    def test_new_dependency_invalidates_previous_check_after_owned_change(self):
        self.acceptance = TaskAcceptance("先分析再修复 app.py", workspace=self.workspace)
        self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
        self.execute("exec", self.command("from app import answer; assert answer == 42"))
        (self.workspace / "plugin.py").write_text("enabled = True\n")
        self.assertIn("plugin.py", self.acceptance.validate("已修改并验证。"))

    def test_manifest_deletion_invalidates_verification_even_inside_a_check(self):
        self.acceptance = TaskAcceptance("保存结果", workspace=self.workspace)
        self.execute("write", {"path": "package.json", "content": "{}"})
        self.execute("exec", self.command("assert __import__('json').load(open('package.json')) == {}"))
        self.assertIsNone(self.acceptance.validate("已修改并验证。"))
        self.execute("exec", self.command("from pathlib import Path; assert Path('package.json').is_file(); Path('package.json').unlink()"))
        self.assertIn("package.json", self.acceptance.validate("已修改并验证。"))

    def test_background_command_that_has_not_written_yet_prevents_completion(self):
        self.acceptance = TaskAcceptance("先分析再修复 app.py", workspace=self.workspace)
        command = self.command("from pathlib import Path; input(); Path('app.py').write_text('answer = 42')")
        started = self.execute("exec", {**command, "background": True})
        self.assertIn("still running", self.acceptance.validate("已完成。"))
        session_id = started["data"]["session_id"]
        self.execute("monitor", {"action": "input", "session_id": session_id, "input": "\n"})
        self.execute("monitor", {"action": "wait", "session_id": session_id})
        self.assertIn("app.py", self.acceptance.validate("已完成。"))

    def test_monitor_file_watch_does_not_own_a_teammate_change(self):
        self.acceptance = TaskAcceptance("汇报文件出现", TaskContract("answer"), workspace=self.workspace)
        invocation = call("monitor", {"action": "watch", "path": "member.py"})
        with patch("task_acceptance.source_snapshot") as snapshot:
            self.acceptance.before_tool(invocation)
            (self.workspace / "member.py").write_text("answer = 42\n")
            self.acceptance.observe(invocation, {"status": "ok"})
            self.assertIsNone(self.acceptance.validate("成员已经写入文件。"))
        snapshot.assert_not_called()

    def test_malformed_mcp_action_does_not_raise_in_tracking(self):
        self.acceptance = TaskAcceptance("说明结果", workspace=self.workspace)
        invocation = call("mcp", {"action": []})
        self.acceptance.before_tool(invocation)
        self.acceptance.observe(invocation, {"status": "error"})
        self.assertEqual(self.acceptance.code_revision, 0)

    def test_first_contract_callback_cannot_verify_its_own_changes(self):
        def mutating_check():
            (self.workspace / "app.py").write_text("answer = 42")

        self.acceptance = TaskAcceptance(
            "说明结果", TaskContract("answer", (mutating_check,)), workspace=self.workspace
        )
        self.assertIn("checks changed source", self.acceptance.validate("已验证。"))
        self.assertNotEqual(self.acceptance.status, "completed")

    def test_monitor_queries_for_unowned_processes_do_not_attribute_concurrent_writes(self):
        for action in ("status", "wait"):
            self.acceptance = TaskAcceptance("汇报状态", TaskContract("answer"), workspace=self.workspace)
            invocation = call("monitor", {"action": action, "session_id": "previous-turn"})
            self.acceptance.before_tool(invocation)
            (self.workspace / f"plugin_{action}.py").write_text("present = True\n")
            self.acceptance.observe(invocation, {"status": "ok", "data": {"state": "running"}})
            self.assertIsNone(self.acceptance.validate("进程仍在运行。"))

    def test_sending_input_to_a_prior_process_requires_observing_its_result(self):
        command = self.command("from pathlib import Path; input(); input(); Path('app.py').write_text('answer = 42')")
        started = json.loads(self.tools.execute(call("exec", {**command, "background": True})))
        session_id = started["data"]["session_id"]
        self.acceptance = TaskAcceptance("处理已有进程", workspace=self.workspace)
        self.execute("monitor", {"action": "input", "session_id": session_id, "input": "\n"})
        self.assertIn("still running", self.acceptance.validate("已完成。"))

    def test_mcp_discovery_code_side_effects_require_verification(self):
        for action in ("search", "describe"):
            self.acceptance = TaskAcceptance("检查工具目录", workspace=self.workspace)
            invocation = call("mcp", {"action": action})
            self.acceptance.before_tool(invocation)
            (self.workspace / f"start_{action}.py").write_text("present = True\n")
            self.acceptance.observe(invocation, {"status": "ok"})
            self.assertIn("no successful verification", self.acceptance.validate("目录可用。"))

    def test_deleting_a_scratch_check_used_by_unchanged_source_invalidates_verification(self):
        (self.workspace / "app.py").write_text("from pathlib import Path\nassert list(Path('.').glob('verify_*.py'))\n")
        self.acceptance = TaskAcceptance("分析 app.py 并修复", workspace=self.workspace)
        self.execute("write", {"path": "verify_tmp.py", "content": "assert True\n"})
        self.execute("exec", {"command": f"{shlex.quote(sys.executable)} verify_tmp.py"})
        self.assertIsNone(self.acceptance.validate("已检查。"))
        self.execute("exec", {"command": "rm verify_tmp.py"})
        self.assertIn("verify_tmp.py", self.acceptance.validate("已检查。"))

    def test_executable_bit_change_invalidates_previous_check(self):
        self.execute("write", {"path": "app.sh", "content": "#!/bin/sh\nexit 0\n"})
        self.execute("exec", {"command": "chmod +x app.sh"})
        self.execute("exec", self.command("from pathlib import Path; assert Path('app.sh').stat().st_mode & 0o111"))
        self.assertIsNone(self.acceptance.validate("已检查。"))
        self.execute("exec", {"command": "chmod -x app.sh"})
        self.assertIn("app.sh", self.acceptance.validate("已检查。"))

    def test_extensionless_launcher_changes_require_verification(self):
        (self.workspace / "bin").mkdir()
        path = self.workspace / "bin" / "agent"
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)
        self.acceptance = TaskAcceptance("检查上线步骤", workspace=self.workspace)
        for mode in ("-x", "+x"):
            self.execute("exec", self.command("from pathlib import Path; assert Path('bin/agent').exists()"))
            self.assertIsNone(self.acceptance.validate("已检查。"))
            self.execute("exec", {"command": f"chmod {mode} bin/agent"})
            self.assertIn("bin/agent", self.acceptance.validate("已检查。"))

    def test_acceptance_mode_tracking_preserves_legacy_resource_digests(self):
        path = self.workspace / "launch.py"
        path.write_text("print(42)\n")
        path.chmod(0o755)
        cache = {}
        expected = file_state.file_digest(path)
        with patch("file_state._digest_identity_is_settled", return_value=True):
            tracked = file_state.source_snapshot((self.workspace,), cache=cache)
            self.assertNotEqual(tracked[str(path)], expected)
            self.assertEqual(file_state.cached_file_digest(path, cache), expected)

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
        self.acceptance._refresh_files = lambda **options: walks.append(1) or refresh(**options)
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
        self.assertIn("result is unknown", self.acceptance.validate("已完成并验证。"))
        self.assertEqual(self.acceptance.outcome()["verification_status"], "unknown")

    def test_stopped_check_without_edits_carries_verification_into_the_next_round(self):
        self.execute("exec", self.command("assert True"))
        started = self.execute("exec", {**self.command("input(); assert True"), "background": True})
        self.execute("monitor", {"action": "stop", "session_id": started["data"]["session_id"]})

        self.assertIn("result is unknown", self.acceptance.validate("已完成。"))
        outcome = json.loads(json.dumps(self.acceptance.outcome()))
        self.assertEqual(outcome["code_revision"], 0)
        self.assertEqual(outcome["verification_status"], "unknown")
        self.assertTrue(outcome["verification_required"])
        self.assertEqual(self.acceptance.last_passed_check, -1)

        self.acceptance = TaskAcceptance(
            "继续任务", TaskContract("answer", verification_required=outcome["verification_required"]),
            workspace=self.workspace,
        )
        self.assertIn("no successful verification", self.acceptance.validate("已完成。"))
        self.execute("exec", self.command("assert True"))
        self.assertIsNone(self.acceptance.validate("已验证。"))
        self.assertFalse(self.acceptance.outcome()["verification_required"])

    def test_failed_explicit_acceptance_check_controls_outcome_until_recovery(self):
        for shell_checked in (False, True):
            with self.subTest(shell_checked=shell_checked):
                failure = "required integration check failed"
                results = iter((failure, None))
                self.acceptance = TaskAcceptance(
                    "检查集成", TaskContract("answer", checks=(lambda: next(results),)),
                    workspace=self.workspace,
                )
                if shell_checked:
                    self.execute("exec", self.command("assert True"))
                self.assertEqual(self.acceptance.validate("已完成。"), f"Task acceptance failed: {failure}")
                self.assertEqual(self.acceptance.outcome()["verification_status"], "failed")
                self.assertTrue(self.acceptance.outcome()["verification_required"])
                self.assertIsNone(self.acceptance.validate("已验证。"))
                self.assertEqual(self.acceptance.outcome()["verification_status"], "passed")
                self.assertFalse(self.acceptance.outcome()["verification_required"])

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

    def test_deleting_an_apparent_scratch_check_requires_verifying_final_layout(self):
        self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
        self.execute("write", {"path": "verify_tmp.py", "content": "assert True\n"})
        quoted = shlex.quote(sys.executable)
        self.execute("exec", {"command": f"{quoted} verify_tmp.py"})
        self.assertIsNone(self.acceptance.validate("已完成并验证。"))
        self.execute("exec", {"command": "rm verify_tmp.py"})
        self.assertIn("verify_tmp.py", self.acceptance.validate("已完成并验证。"))
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

    def test_background_handoff_requires_reverification_after_transient_source_edits(self):
        for prior_edit in (False, True):
            with self.subTest(prior_edit=prior_edit):
                self.acceptance = TaskAcceptance("Analyze and fix", workspace=self.workspace)
                if prior_edit:
                    self.execute("write", {"path": "app.py", "content": "answer = 42\n"})
                self.execute("exec", self.command("assert True"))
                self.execute("exec", self.command(
                    "from pathlib import Path; source = Path('temporary.py'); "
                    "source.write_text('answer = 42\\n'); from temporary import answer; "
                    "assert answer == 42; source.unlink()"
                ))
                self.assertIn("verify your own change", self.acceptance.validate("Handed off.", background=True))
                self.execute("exec", self.command("assert True"))
                (self.workspace / "teammate.py").write_text("work_in_progress = True\n")
                self.assertIsNone(self.acceptance.validate("Handed off.", background=True))

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
            "我已经修复了一半，但我无法继续：缺少权限。",
            "I fixed the parser. I cannot proceed without the API key.",
            "Unable to complete the migration: the database is read-only.",
            "需要你确认 n=1 时的行为。",
            # 之前 here means "until"; 其他 is not the pronoun 他; critic is an object.
            "在你确认之前，我无法继续。",
            "我让 critic 运行测试，但因缺少授权无法完成验证。",
            "I asked the critic to run them, but cannot proceed without credentials.",
            # The user and critic only say where the file and tests came from.
            "用户提供的文件无法执行，缺少权限。",
            "critic 写的测试无法执行，缺少凭据。",
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
        self.assertEqual(self.status("pytest 无法执行，缺少权限。", others=()), "blocked")
        self.assertEqual(self.status("critic 无法执行测试，缺少权限。", others=()), "blocked")
        self.assertEqual(self.status("critic 无法执行测试，缺少权限。"), "completed")

    def test_self_reported_inability_without_external_blocker_does_not_complete(self):
        for wording in ("无法验证。", "无法执行测试。", "无法完成，测试失败。"):
            acceptance = self.acceptance()
            self.assertIn("not finished", acceptance.validate(wording))
            self.assertNotIn(acceptance.status, {"blocked", "completed"})
