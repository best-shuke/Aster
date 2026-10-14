import json
import os
import shlex
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tool_runtime import MAX_RESULT_CHARS, ExecutionContext, ToolResult
from tools import CodingTools, ProcessResult, ToolPolicy


def call(name, arguments):
    return {
        "id": f"call-{name}",
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


def python_command(program):
    return f"{shlex.quote(sys.executable)} -c {shlex.quote(program)}"


class CodingToolsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temporary.name).resolve()
        self.tools = CodingTools(self.workspace)

    def tearDown(self):
        self.temporary.cleanup()

    def execute(self, name, arguments, **kwargs):
        raw = self.tools.execute(call(name, arguments), **kwargs)
        self.assertLessEqual(len(raw), MAX_RESULT_CHARS)
        result = json.loads(raw)
        self.assertEqual(
            set(result), {"status", "data", "error", "truncated", "next_cursor", "artifact_path"}
        )
        return result

    def test_malformed_arguments_return_a_structured_error(self):
        result = json.loads(self.tools.execute({"function": {"name": "read", "arguments": "{bad"}}))
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["error"]["code"], "invalid_arguments")

    def test_update_plan_publishes_title_and_steps_without_worker_access(self):
        events = []
        self.tools.on_task_update = events.append
        plan = {
            "title": "修复登录流程",
            "steps": [
                {"step": "定位问题", "status": "completed"},
                {"step": "修复并验证", "status": "in_progress"},
            ],
        }
        result = self.execute("update_plan", plan)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(self.tools.task_title, plan["title"])
        self.assertEqual(self.tools.task_plan, plan)
        self.assertEqual(events, [{"phase": "plan", **plan}])
        self.assertNotIn(
            "update_plan", {item["function"]["name"] for item in self.tools.worker_definitions}
        )
        denied = json.loads(
            self.tools.execute_workspace(
                call("update_plan", plan), ExecutionContext(time.monotonic() + 10)
            )
        )
        self.assertEqual(denied["error"]["code"], "unavailable_tool")

    def test_invalid_plan_does_not_replace_the_previous_plan(self):
        initial = {"title": "任务", "steps": [{"step": "检查", "status": "pending"}]}
        self.execute("update_plan", initial)
        for invalid in (
            {"title": "空计划", "steps": []},
            {"title": " ", "steps": [{"step": "检查", "status": "pending"}]},
            {"title": "任务", "steps": [{"step": " ", "status": "pending"}]},
            {"title": "任务", "steps": [{"step": "检查", "status": "unknown"}]},
            {
                "title": "任务",
                "steps": [
                    {"step": "检查", "status": "in_progress"},
                    {"step": "修改", "status": "in_progress"},
                ],
            },
            {"title": "长" * 121, "steps": [{"step": "检查", "status": "pending"}]},
        ):
            with self.subTest(invalid=invalid):
                result = self.execute("update_plan", invalid)
                self.assertEqual(result["error"]["code"], "invalid_arguments")
                self.assertEqual(self.tools.task_plan, initial)

    def test_ask_user_returns_the_structured_selection(self):
        received = []

        def choose(question, options):
            received.append((question, options))
            return {
                "answer": options[0]["label"],
                "selected_index": 1,
                "selected_label": options[0]["label"],
                "custom": False,
            }

        self.tools.attach_user_input(choose)
        options = [
            {"label": "快", "description": "更快完成。"},
            {"label": "稳", "description": "检查更完整。"},
        ]
        result = self.execute("ask_user", {"question": "如何执行？", "options": options})

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["data"]["answer"], "快")
        self.assertEqual(received, [("如何执行？", options)])
        self.assertNotIn(
            "ask_user", {item["function"]["name"] for item in self.tools.worker_definitions}
        )

    def test_ask_user_requires_a_description_for_every_option(self):
        self.tools.attach_user_input(lambda question, options: {})
        result = self.execute(
            "ask_user",
            {
                "question": "如何执行？",
                "options": [{"label": "A"}, {"label": "B", "description": "说明"}],
            },
        )

        self.assertEqual(result["error"]["code"], "invalid_arguments")

    def test_ask_user_rejects_duplicate_and_custom_options(self):
        self.tools.attach_user_input(lambda question, options: {})
        cases = [
            [
                {"label": "A", "description": "说明 A"},
                {"label": "A", "description": "重复"},
            ],
            [
                {"label": "A", "description": "说明 A"},
                {"label": "其他", "description": "不应由模型添加"},
            ],
        ]
        for options in cases:
            with self.subTest(options=options):
                result = self.execute("ask_user", {"question": "如何执行？", "options": options})
                self.assertEqual(result["status"], "error")

    def test_validation_rejects_bad_inputs_before_side_effects(self):
        (self.workspace / "sample.txt").write_text("original")
        cases = [
            ("read", []),
            ("read", {}),
            ("read", {"path": "sample.txt", "offset": 0}),
            ("read", {"path": "sample.txt", "offset": True}),
            ("read", {"path": "sample.txt", "limit": -1}),
            ("exec", {"command": "touch unexpected", "timeout": "1"}),
            ("write", {"path": "unexpected", "content": "x", "_context": {}}),
            ("edit", {"path": "sample.txt", "edits": []}),
            ("edit", {"path": "sample.txt", "edits": [{"oldText": "", "newText": "x"}]}),
            ("edit", {"path": "sample.txt", "edits": [{"oldText": "original"}]}),
        ]
        for name, arguments in cases:
            with self.subTest(name=name, arguments=arguments):
                result = self.execute(name, arguments)
                self.assertEqual(result["error"]["code"], "invalid_arguments")
        self.assertEqual((self.workspace / "sample.txt").read_text(), "original")
        self.assertFalse((self.workspace / "unexpected").exists())

    def read_all(self, name):
        cursor = {}
        chunks = []
        for _ in range(100):
            result = self.execute("read", {"path": name, **cursor})
            self.assertEqual(result["status"], "ok", result)
            chunks.append(result["data"]["content"])
            if result["next_cursor"] is None:
                self.assertTrue(result["data"]["eof"])
                self.assertFalse(result["truncated"])
                return "".join(chunks)
            self.assertTrue(result["truncated"])
            self.assertNotEqual(cursor, result["next_cursor"])
            cursor = result["next_cursor"]
        self.fail("read pagination did not finish")

    def test_read_pagination_never_skips_lines_or_long_line_fragments(self):
        content = "".join(f"{i:04d} " + "x" * 90 + "\n" for i in range(2001))
        (self.workspace / "large.txt").write_text(content)
        self.assertEqual(self.read_all("large.txt"), content)
        content = "中文" * 9000 + "\r\nlast line"
        (self.workspace / "long.txt").write_text(content, newline="")
        self.assertEqual(self.read_all("long.txt"), content)

    def test_read_cursor_and_limit_identify_actual_range(self):
        (self.workspace / "file").write_text("one\ntwo\nthree\n")
        result = self.execute("read", {"path": "file", "offset": 2, "limit": 1})
        self.assertEqual(result["data"]["content"], "two\n")
        self.assertEqual(result["data"]["start_line"], 2)
        self.assertEqual(result["next_cursor"], {"offset": 3, "column": 0})
        (self.workspace / "empty").touch()
        self.assertEqual(self.execute("read", {"path": "empty"})["data"]["content"], "")

    def test_oversized_structured_result_is_preserved_in_an_artifact(self):
        original = ToolResult(
            status="partial",
            data={"reports": ["中" * 20000]},
            error={"code": "incomplete", "message": "missing checks"},
        )
        result = json.loads(self.tools.artifacts.render(original))
        self.assertEqual(result["status"], "partial")
        self.assertTrue(result["truncated"])
        restored = json.loads(Path(result["artifact_path"]).read_text())
        self.assertEqual(restored, original.as_dict())
        self.assertEqual(json.loads(self.read_all(result["artifact_path"])), original.as_dict())

    def test_exec_keeps_complete_output_and_exit_status(self):
        program = "print('HEAD'); print('x' * 60000); print('TAIL'); raise SystemExit(7)"
        result = self.execute("exec", {"command": python_command(program)})
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["data"]["exit_code"], 7)
        self.assertTrue(result["truncated"])
        self.assertIn("HEAD", result["data"]["output"])
        self.assertIn("TAIL", result["data"]["output"])
        self.assertEqual(
            Path(result["artifact_path"]).read_text(), "HEAD\n" + "x" * 60000 + "\nTAIL\n"
        )

    def test_exec_honors_cwd(self):
        (self.workspace / "subdir").mkdir()
        result = self.execute(
            "exec", {"command": python_command("import os; print(os.getcwd())"), "cwd": "subdir"}
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["data"]["output"].strip(), str(self.workspace / "subdir"))
        self.assertFalse(result["data"]["timed_out"])

    def test_exec_obeys_fractional_remaining_task_deadline(self):
        started = time.monotonic()
        result = self.execute(
            "exec",
            {"command": python_command("import time; time.sleep(5)"), "timeout": 120},
            deadline=started + 0.15,
        )
        self.assertEqual(result["status"], "timeout")
        self.assertTrue(result["data"]["timed_out"])
        self.assertLess(time.monotonic() - started, 2)

    def test_expired_deadline_prevents_writes(self):
        result = self.execute(
            "write", {"path": "unexpected", "content": "x"}, deadline=time.monotonic() - 1
        )
        self.assertEqual(result["status"], "timeout")
        self.assertFalse((self.workspace / "unexpected").exists())

    def test_existing_file_requires_read_before_full_overwrite(self):
        target = self.workspace / "existing.py"
        target.write_text("original\n")
        arguments = {"path": "existing.py", "content": "replacement\n"}
        blocked = self.execute("write", arguments)
        self.assertEqual(blocked["error"]["code"], "read_required")
        self.assertEqual(target.read_text(), "original\n")
        self.assertEqual(self.execute("read", {"path": "existing.py"})["status"], "ok")
        self.assertEqual(self.execute("write", arguments)["status"], "ok")
        self.assertEqual(target.read_text(), "replacement\n")

    def test_new_file_can_be_created_without_a_read(self):
        self.assertEqual(
            self.execute("write", {"path": "new.py", "content": "new\n"})["status"], "ok"
        )

    def test_exec_cleans_up_children_after_the_parent_exits(self):
        marker = self.workspace / "late-write.txt"
        child = (
            "import time; from pathlib import Path; time.sleep(0.3); "
            f"Path({str(marker)!r}).write_text('late')"
        )
        parent = f"import subprocess, sys; subprocess.Popen([sys.executable, '-c', {child!r}])"
        result = self.execute("exec", {"command": python_command(parent)})
        self.assertEqual(result["status"], "ok", result)
        time.sleep(0.5)
        self.assertFalse(marker.exists())

    def test_fork_isolates_configuration_and_rebinds_handlers(self):
        self.tools.task_plan = {"title": "original", "steps": [{"step": "one"}]}
        self.tools.worker_roles = {"review": "original"}
        snapshot = self.tools.fork()
        snapshot.task_plan["steps"][0]["step"] = "changed"
        snapshot.worker_roles["review"] = "changed"
        snapshot.policy = ToolPolicy(deny=frozenset({"read"}))
        self.assertEqual(self.tools.task_plan["steps"][0]["step"], "one")
        self.assertEqual(self.tools.worker_roles["review"], "original")
        self.assertTrue(self.tools.policy.permits("read"))
        self.assertIs(snapshot.registry["write"].handler.__self__, snapshot)
        self.assertIs(snapshot._file_versions, self.tools._file_versions)
        self.assertIs(snapshot.worker_status, self.tools.worker_status)
        self.tools.apply_configuration(snapshot)
        self.assertIs(self.tools.registry["write"].handler.__self__, self.tools)
        self.assertFalse(self.tools.policy.permits("read"))

    def test_search_processes_inherit_task_deadline(self):
        log = self.workspace / "empty.log"
        log.touch()
        deadline = time.monotonic() + 0.5
        for name, arguments in [("grep", {"pattern": "missing"}), ("find", {"pattern": "*.txt"})]:
            with (
                self.subTest(name=name),
                patch("tools.run_process", return_value=ProcessResult(log, 1, False, 0)) as process,
            ):
                self.execute(name, arguments, deadline=deadline)
                self.assertEqual(process.call_args.args[3].deadline, deadline)

    def test_search_timeouts_are_not_reported_as_empty_success(self):
        log = self.workspace / "empty.log"
        log.touch()
        with patch("tools.run_process", return_value=ProcessResult(log, -9, True, 0.2)):
            result = self.execute("grep", {"pattern": "missing"})
        self.assertEqual(result["status"], "timeout")
        self.assertEqual(result["artifact_path"], str(log))

    def test_grep_treats_leading_hyphen_as_pattern_and_ignores_external_config(self):
        (self.workspace / "sample.txt").write_text("Find the --version option here\n")
        config = self.workspace / "rg-config"
        config.write_text("--glob=!*.txt\n")
        with patch.dict(os.environ, {"RIPGREP_CONFIG_PATH": str(config)}):
            result = self.execute(
                "grep", {"pattern": "--version", "literal": True, "path": "sample.txt"}
            )
        self.assertEqual(result["status"], "ok", result)
        self.assertEqual(len(result["data"]["matches"]), 1)
        self.assertEqual(result["data"]["matches"][0]["line"], 1)
        self.assertIn("option here", result["data"]["matches"][0]["text"])

    def test_grep_paginates_complete_records_and_excludes_its_artifacts(self):
        content = "".join(f"needle {i} " + "x" * 100 + "\n" for i in range(100))
        (self.workspace / "sample.txt").write_text(content)
        matches = []
        cursor = {}
        while True:
            result = self.execute("grep", {"pattern": "needle", "limit": 17, **cursor})
            self.assertEqual(result["status"], "ok", result)
            matches.extend(result["data"]["matches"])
            cursor = result["next_cursor"]
            if cursor is None:
                break
        self.assertEqual([entry["line"] for entry in matches], list(range(1, 101)))
        self.assertEqual("".join(entry["text"] for entry in matches), content)
        self.assertEqual({Path(entry["path"]).name for entry in matches}, {"sample.txt"})
        self.assertEqual(self.execute("grep", {"pattern": "absent"})["data"]["matches"], [])

    def test_search_honors_gitignore_without_a_git_repository_and_explicit_glob_can_override(self):
        (self.workspace / ".gitignore").write_text("ignored.txt\n")
        (self.workspace / "ignored.txt").write_text("needle")
        (self.workspace / "visible.txt").write_text("needle")
        result = self.execute("grep", {"pattern": "needle"})
        self.assertEqual(
            [Path(item["path"]).name for item in result["data"]["matches"]], ["visible.txt"]
        )
        result = self.execute("find", {"pattern": "*.txt"})
        self.assertEqual(result["data"]["paths"], ["ignored.txt", "visible.txt"])

    def test_find_large_directory_returns_every_complete_filename(self):
        expected = [f"{i:04d}-" + "x" * 100 + ".txt" for i in range(600)] + ["line\nbreak.txt"]
        for name in expected:
            (self.workspace / name).touch()
        paths = []
        cursor = {}
        for _ in range(30):
            result = self.execute("find", {"pattern": "*.txt", **cursor})
            self.assertEqual(result["status"], "ok", result)
            paths.extend(result["data"]["paths"])
            cursor = result["next_cursor"]
            self.assertEqual(result["truncated"], cursor is not None)
            if cursor is None:
                break
        self.assertEqual(paths, sorted(expected))
        self.assertEqual(len(set(paths)), len(paths))

    def test_ls_pagination(self):
        for name in ("a", "b", ".hidden"):
            (self.workspace / name).touch()
        (self.workspace / "c").mkdir()
        first = self.execute("ls", {"limit": 2})
        second = self.execute("ls", {"limit": 2, **first["next_cursor"]})
        self.assertEqual(
            first["data"]["entries"] + second["data"]["entries"], [".hidden", "a", "b", "c/"]
        )
        self.assertIsNone(second["next_cursor"])

    def test_edit_failure_leaves_file_untouched_and_success_preserves_crlf(self):
        path = self.workspace / "sample.txt"
        original = b"alpha\r\nbeta\r\n"
        path.write_bytes(original)
        result = self.execute(
            "edit",
            {
                "path": "sample.txt",
                "edits": [
                    {"oldText": "alpha", "newText": "changed"},
                    {"oldText": "missing", "newText": "x"},
                ],
            },
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(path.read_bytes(), original)
        result = self.execute(
            "edit", {"path": "sample.txt", "edits": [{"oldText": "alpha", "newText": "new"}]}
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(path.read_bytes(), b"new\r\nbeta\r\n")

    def test_overlapping_edits_leave_the_original_untouched(self):
        path = self.workspace / "sample.txt"
        path.write_text("abcdef")
        result = self.execute(
            "edit",
            {
                "path": "sample.txt",
                "edits": [
                    {"oldText": "abc", "newText": "one"},
                    {"oldText": "bcd", "newText": "two"},
                ],
            },
        )
        self.assertEqual(result["status"], "error")
        self.assertEqual(path.read_text(), "abcdef")

    def test_worker_cannot_execute_commands_write_or_delegate(self):
        definitions = {item["function"]["name"] for item in self.tools.worker_definitions}
        self.assertEqual(definitions, {"read", "grep", "find", "ls"})
        for name in ("exec", "bash", "observe", "write", "edit", "delegate"):
            with self.subTest(name=name):
                result = json.loads(
                    self.tools.execute_workspace(
                        call(name, {"command": "touch unexpected"}),
                        ExecutionContext(time.monotonic() + 1),
                    )
                )
                self.assertEqual(result["error"]["code"], "unavailable_tool")
        self.assertFalse((self.workspace / "unexpected").exists())

    def test_worker_allow_grants_listed_tools_and_keeps_others_blocked(self):
        self.tools.worker_allow = frozenset({"exec"})
        names = {item["function"]["name"] for item in self.tools.worker_definitions}
        self.assertIn("exec", names)
        self.assertNotIn("write", names)
        result = json.loads(
            self.tools.execute_workspace(
                call("exec", {"command": python_command("print('worker-ok')")}),
                ExecutionContext(time.monotonic() + 30),
            )
        )
        self.assertEqual(result["status"], "ok", result)
        self.assertIn("worker-ok", json.dumps(result["data"], ensure_ascii=False))
        blocked = json.loads(
            self.tools.execute_workspace(
                call("write", {"path": "nope.txt", "content": "x"}),
                ExecutionContext(time.monotonic() + 5),
            )
        )
        self.assertEqual(blocked["error"]["code"], "unavailable_tool")
        self.assertFalse((self.workspace / "nope.txt").exists())

    def test_worker_allow_cannot_grant_delegate_ask_user_or_plan(self):
        self.tools.worker_allow = frozenset({"delegate", "ask_user", "update_plan"})
        names = {item["function"]["name"] for item in self.tools.worker_definitions}
        self.assertEqual(names & {"delegate", "ask_user", "update_plan"}, set())
        result = json.loads(
            self.tools.execute_workspace(
                call(
                    "update_plan",
                    {"title": "x", "steps": [{"step": "s", "status": "pending"}]},
                ),
                ExecutionContext(time.monotonic() + 5),
            )
        )
        self.assertEqual(result["error"]["code"], "unavailable_tool")

    def test_worker_write_grant_does_not_override_policy_deny(self):
        self.tools.worker_allow = frozenset({"write", "edit"})
        self.tools.policy = ToolPolicy(deny=frozenset({"write"}))
        names = {item["function"]["name"] for item in self.tools.worker_definitions}
        self.assertNotIn("write", names)
        self.assertIn("edit", names)
        result = json.loads(
            self.tools.execute_workspace(
                call("write", {"path": "blocked.txt", "content": "not allowed"}),
                ExecutionContext(time.monotonic() + 5),
            )
        )
        self.assertEqual(result["error"]["code"], "unavailable_tool")
        self.assertFalse((self.workspace / "blocked.txt").exists())


if __name__ == "__main__":
    unittest.main()
