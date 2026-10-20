import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from commands import CommandResult
from conversation import Conversation
from main import ask, handle_command, main
from mcp_client import McpManager
from skills import Skill, SkillCatalog, SkillDiagnostic
from tests.support import make_settings
from tests.test_conversation import FakeClient
from tool_runtime import ToolResult
from tools import CodingTools
from ui import Console
from usage import CacheStats


class CliBrandingTests(unittest.TestCase):
    def test_help_uses_aster_brand_and_agent_command_without_loading_configuration(self):
        output = io.StringIO()
        with (
            patch("sys.argv", ["main.py", "--help"]),
            patch("sys.stdout", output),
            patch("main.load_settings") as settings,
            self.assertRaises(SystemExit) as raised,
        ):
            main()

        self.assertEqual(raised.exception.code, 0)
        self.assertIn("usage: agent ", output.getvalue())
        self.assertIn("Aster", output.getvalue())
        self.assertNotIn("DeepSeek coding agent", output.getvalue())
        settings.assert_not_called()


class SessionCommandTests(unittest.TestCase):
    def test_new_and_clear_start_a_clean_conversation_and_ui(self):
        for command in ("/new", "/clear"):
            with self.subTest(command=command), tempfile.TemporaryDirectory() as temporary:
                workspace = Path(temporary)
                settings = make_settings(
                    api_key="test",
                    base_url="https://example.invalid",
                    model="test",
                    workspace=workspace,
                    session_file=workspace / "session.json",
                    max_context_chars=120000,
                    keep_recent_turns=6,
                    max_task_seconds=3600,
                    max_subagents=4,
                )
                client = FakeClient({"role": "assistant", "content": "新回复"})
                client.cache_stats = CacheStats()
                client.cache_stats.record({"prompt_tokens": 10}, "solve")
                output = io.StringIO()
                with patch("sys.stdout", output):
                    console = Console()
                    tools = CodingTools(workspace, on_task_update=console.task_update)
                    conversation = Conversation(settings, client, tools)
                    conversation.messages.append({"role": "user", "content": "OLD_SESSION_MARKER"})
                    tools.task_title = "旧任务"
                    tools.task_plan = {
                        "title": "旧任务",
                        "steps": [{"step": "旧步骤", "status": "pending"}],
                    }
                    conversation.save()
                    console._task_title = "旧任务"
                    console._agents = {"A1": {"title": "旧代理"}}
                    console._todos = tools.task_plan["steps"]
                    self.assertIs(handle_command(conversation, console, command), CommandResult.HANDLED)
                    self.assertEqual(console._task_title, "")
                    self.assertEqual(console._agents, {})
                    self.assertEqual(console._todos, [])
                    self.assertEqual(conversation.messages, [])
                    self.assertIsNone(tools.task_plan)
                    self.assertEqual(client.cache_stats.snapshot()["calls"], 0)
                    self.assertFalse(settings.session_file.exists())
                    self.assertEqual(ask(conversation, console, "新问题"), "新回复")
                self.assertNotIn("OLD_SESSION_MARKER", str(client.requests))
                self.assertIn("新会话", output.getvalue())
                self.assertEqual(console._task_title, "新问题")

    def test_tasks_and_todo_commands_show_the_dashboard(self):
        console = Console()
        with patch.object(console, "tasks") as tasks:
            for command in ("/tasks", "/todo"):
                self.assertIs(handle_command(None, console, command), CommandResult.HANDLED)
        self.assertEqual(tasks.call_count, 2)

    def test_main_auto_connects_before_showing_the_ui_and_closes_on_exit(self):
        import main

        settings = Mock(mcp_enabled=True, websearch_enabled=False)
        manager = Mock(spec=McpManager)
        manager.server_count = 2
        servers = [
            {"name": "playwright", "status": "connected", "tool_count": 25},
            {"name": "js-reverse", "status": "connected", "tool_count": 24},
        ]
        manager.connect_all.return_value = ToolResult(data={"servers": servers, "failures": []})
        manager.status.return_value = ToolResult(data={"servers": servers})

        def verify_ready(conversation, console):
            manager.connect_all.assert_called_once()
            console.update_mcp.assert_called_once_with(servers)

        with (
            patch("main.load_settings", return_value=settings),
            patch("main.McpManager", return_value=manager),
            patch("main.Console") as console,
            patch("main.Conversation") as conversations,
            patch("main.SkillCatalog"),
            patch("main.DeepSeekClient"),
            patch("main.CodingTools"),
            patch("main.run_interactive", side_effect=verify_ready),
            patch("sys.argv", ["main.py"]),
        ):
            console.return_value.interactive = False
            main.main()
        conversations.return_value.close.assert_called_once()
        self.assertIs(conversations.call_args.kwargs["mcp_manager"], manager)

    def test_startup_failure_warns_but_does_not_prevent_chat(self):
        import main

        settings = Mock(mcp_enabled=True, websearch_enabled=False)
        manager = Mock(spec=McpManager)
        manager.server_count = 1
        manager.connect_all.return_value = ToolResult(
            status="error",
            data={
                "servers": [{"name": "broken", "status": "failed"}],
                "failures": [{"server": "broken", "error": "unavailable"}],
            },
        )
        with (
            patch("main.load_settings", return_value=settings),
            patch("main.McpManager", return_value=manager),
            patch("main.Console") as console,
            patch("main.Conversation") as conversations,
            patch("main.SkillCatalog"),
            patch("main.DeepSeekClient"),
            patch("main.CodingTools"),
            patch("main.run_interactive") as run,
            patch("sys.argv", ["main.py"]),
        ):
            console.return_value.interactive = False
            main.main()
        run.assert_called_once()
        self.assertIn("broken", console.return_value.warning.call_args.args[0])
        conversations.return_value.close.assert_called_once()
        self.assertIs(conversations.call_args.kwargs["mcp_manager"], manager)


class ResumeCommandTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        workspace = Path(temporary.name)
        settings = make_settings(
            api_key="test",
            base_url="https://example.invalid",
            model="test",
            workspace=workspace,
            session_file=workspace / "session.json",
            max_context_chars=120000,
            keep_recent_turns=6,
            max_task_seconds=3600,
            max_subagents=4,
        )
        self.console = Mock(spec=Console)
        self.console.interactive = True
        self.conversation = Conversation(settings, FakeClient(), CodingTools(workspace))
        self.old_id = "a" * 32
        self.current_id = "b" * 32
        self.conversation.session_id = self.old_id
        self.old_messages = [
            {"role": "user", "content": "旧提问"},
            {"role": "assistant", "content": "旧回答"},
        ]
        self.conversation.messages.extend(self.old_messages)
        self.plan = {"title": "旧任务", "steps": [{"step": "检查", "status": "completed"}]}
        self.conversation.tools.task_title = "旧任务"
        self.conversation.tools.task_plan = self.plan
        self.conversation.last_outcome = {"status": "completed"}
        self.conversation.clear()
        self.conversation.session_id = self.current_id
        self.conversation.messages.extend(
            [
                {"role": "user", "content": "当前提问"},
                {"role": "assistant", "content": "当前回答"},
            ]
        )
        self.conversation.save()

    def test_resume_menu_restores_the_selected_session_and_dashboard(self):
        self.console.select_option.return_value = 1

        self.assertIs(handle_command(self.conversation, self.console, "/resume"), CommandResult.HANDLED)

        self.assertEqual(self.conversation.session_id, self.old_id)
        self.assertEqual(self.conversation.messages, self.old_messages)
        self.assertEqual(self.console.select_option.call_args.kwargs["title"], "恢复会话")
        options = self.console.select_option.call_args.args[1]
        self.assertIn("当前会话", options[0]["label"])
        self.assertIn("旧提问", options[1]["label"])
        self.console.history.assert_called_once_with(self.old_messages, 10)
        self.console.restore_task.assert_called_once_with(
            "旧任务", self.plan, completed=True, status="completed"
        )
        self.assertEqual(self.console.reset_session.call_args.kwargs["history_count"], 2)
        self.assertEqual(len(self.conversation.sessions.list()), 2)

    def test_cancelling_resume_leaves_the_current_session_unchanged(self):
        self.console.select_option.return_value = None
        before = self.conversation.settings.session_file.read_text()
        with patch.object(self.conversation, "resume") as resume:
            self.assertIs(handle_command(self.conversation, self.console, "/resume"), CommandResult.HANDLED)

        resume.assert_not_called()
        self.console.info.assert_called_once_with("已取消恢复会话。")
        self.console.reset_session.assert_not_called()
        self.console.history.assert_not_called()
        self.assertEqual(self.conversation.settings.session_file.read_text(), before)

    def test_explicit_number_id_prefix_and_latest_skip_the_menu(self):
        for selector, session_id in (
            ("2", self.old_id),
            (self.old_id, self.old_id),
            (self.old_id[:8].upper(), self.old_id),
            ("latest", self.current_id),
        ):
            with self.subTest(selector=selector):
                self.console.reset_mock()
                with patch.object(self.conversation, "resume") as resume:
                    self.assertIs(handle_command(self.conversation, self.console, f"/resume {selector}"), CommandResult.HANDLED)
                resume.assert_called_once_with(session_id)
                self.console.select_option.assert_not_called()

    def test_numeric_selector_is_a_list_number_not_an_id_prefix(self):
        sessions = self.conversation.sessions.list()
        older = type(sessions[1])("1" + "a" * 31, "旧会话", sessions[1].updated_at, 2)
        with (
            patch.object(self.conversation.sessions, "list", return_value=[sessions[0], older]),
            patch.object(self.conversation, "resume") as resume,
        ):
            self.assertIs(handle_command(self.conversation, self.console, "/resume 1"), CommandResult.HANDLED)
        resume.assert_called_once_with(self.current_id)

    def test_invalid_resume_arguments_are_commands_and_do_not_modify_the_session(self):
        for arguments in ("0", "3", "-1", "missing", "1 extra"):
            with self.subTest(arguments=arguments):
                self.console.reset_mock()
                with patch.object(self.conversation, "resume") as resume:
                    self.assertIs(handle_command(self.conversation, self.console, f"/resume {arguments}"), CommandResult.HANDLED)
                resume.assert_not_called()
                self.console.error.assert_called_once()
                self.console.select_option.assert_not_called()
                self.console.reset_session.assert_not_called()

    def test_ambiguous_id_prefix_is_rejected(self):
        self.conversation.session_id = "a" * 31 + "b"
        self.conversation.save()

        with patch.object(self.conversation, "resume") as resume:
            self.assertIs(handle_command(self.conversation, self.console, "/resume aaaa"), CommandResult.HANDLED)

        resume.assert_not_called()
        self.console.error.assert_called_once()
        self.console.reset_session.assert_not_called()

    def test_redirected_resume_lists_sessions_without_waiting_or_switching(self):
        self.console.interactive = False
        with patch.object(self.conversation, "resume") as resume:
            self.assertIs(handle_command(self.conversation, self.console, "/resume"), CommandResult.HANDLED)

        resume.assert_not_called()
        self.console.select_option.assert_not_called()
        self.console.reset_session.assert_not_called()
        listing = str(self.console.info.call_args_list)
        self.assertIn("旧提问", listing)
        self.assertIn("当前提问", listing)
        self.assertIn("/resume latest", listing)

    def test_empty_resume_does_not_open_a_menu(self):
        with patch.object(self.conversation.sessions, "list", return_value=[]):
            self.assertIs(handle_command(self.conversation, self.console, "/resume"), CommandResult.HANDLED)

        self.console.select_option.assert_not_called()
        self.console.reset_session.assert_not_called()
        self.assertIn("暂无可恢复", self.console.info.call_args.args[0])

    def test_missing_selected_snapshot_reports_error_and_keeps_the_current_session(self):
        before = self.conversation.settings.session_file.read_text()

        def select_session(*args, **kwargs):
            self.conversation.sessions.snapshot_path(self.old_id).unlink()
            return 1

        self.console.select_option.side_effect = select_session
        self.assertIs(handle_command(self.conversation, self.console, "/resume"), CommandResult.HANDLED)

        self.console.error.assert_called_once()
        self.console.reset_session.assert_not_called()
        self.assertEqual(self.conversation.session_id, self.current_id)
        self.assertEqual(self.conversation.settings.session_file.read_text(), before)

    def test_corrupt_snapshot_is_reported_but_other_sessions_remain_available(self):
        broken = self.conversation.sessions.directory / "broken.json"
        broken.write_text("{bad")
        self.console.select_option.return_value = 1

        self.assertIs(handle_command(self.conversation, self.console, "/resume"), CommandResult.HANDLED)

        self.assertEqual(self.conversation.session_id, self.old_id)
        self.console.warning.assert_called_once()
        self.assertIn("broken.json", self.console.warning.call_args.args[0])
        self.assertTrue(broken.exists())

    def test_resumed_screen_shows_restored_history_instead_of_a_new_session(self):
        output = io.StringIO()
        with patch("sys.stdout", output):
            console = Console()
            self.assertIs(handle_command(self.conversation, console, f"/resume {self.old_id}"), CommandResult.HANDLED)

        self.assertIn("已恢复", output.getvalue())
        self.assertIn("旧提问", output.getvalue())
        self.assertIn("旧回答", output.getvalue())
        self.assertNotIn("新会话", output.getvalue())
        self.assertEqual(console._task_state, "completed")

    def test_new_session_save_failure_does_not_clear_the_conversation_or_ui(self):
        before = self.conversation.settings.session_file.read_text()
        with patch.object(self.conversation, "save", side_effect=OSError("disk full")):
            self.assertIs(handle_command(self.conversation, self.console, "/new"), CommandResult.HANDLED)

        self.console.error.assert_called_once()
        self.console.reset_session.assert_not_called()
        self.assertEqual(self.conversation.session_id, self.current_id)
        self.assertEqual(self.conversation.settings.session_file.read_text(), before)


class SkillStartupTests(unittest.TestCase):
    def test_cli_loads_explicit_paths_even_when_default_discovery_is_disabled(self):
        import main

        for enabled, flags in (
            (True, ["--no-skills", "--skill", "custom", "--skill", "other/SKILL.md"]),
            (False, ["--skill", "custom", "--skill", "other/SKILL.md"]),
        ):
            with self.subTest(enabled=enabled):
                settings = Mock(mcp_enabled=False, websearch_enabled=False, skills_enabled=enabled)
                with (
                    patch("main.load_settings", return_value=settings),
                    patch("main.Console") as console,
                    patch("main.Conversation") as conversations,
                    patch("main.SkillCatalog") as catalog,
                    patch("main.DeepSeekClient"),
                    patch("main.CodingTools"),
                    patch("main.run_interactive"),
                    patch("sys.argv", ["main.py", *flags]),
                ):
                    console.return_value.interactive = False
                    self.assertEqual(main.main(), 0)

                self.assertEqual(
                    catalog.call_args.kwargs,
                    {
                        "enabled": True,
                        "extra_dirs": (Path("custom"), Path("other/SKILL.md")),
                        "include_defaults": False,
                    },
                )
                conversations.return_value.close.assert_called_once()

    def test_noninteractive_skill_errors_are_reported_without_a_traceback(self):
        import main

        settings = Mock(mcp_enabled=False, websearch_enabled=False, skills_enabled=True)
        error = ValueError("没有找到 skill：missing；用 /skills 查看可用技能。")
        with (
            patch("main.load_settings", return_value=settings),
            patch("main.Console") as console,
            patch("main.Conversation") as conversations,
            patch("main.SkillCatalog"),
            patch("main.DeepSeekClient"),
            patch("main.CodingTools"),
            patch("main.ask", side_effect=error),
            patch("sys.argv", ["main.py", "/skill:missing", "task"]),
        ):
            console.return_value.interactive = False
            self.assertEqual(main.main(), 2)

        console.return_value.error.assert_called_once_with(error)
        conversations.return_value.close.assert_called_once()


class ChoiceCommandTests(unittest.TestCase):
    def setUp(self):
        self.console = Mock(spec=Console)
        self.console.interactive = True
        self.conversation = SimpleNamespace(
            messages=[
                {"role": "user", "content": "question"},
                {"role": "assistant", "content": "answer"},
            ],
            skills=None,
        )

    def attach_skills(self):
        catalog = Mock(spec=SkillCatalog)
        catalog.enabled = True
        catalog.diagnostics = []
        catalog.skills = [
            Skill("alpha", "检查接口", Path("skills/alpha/SKILL.md"), Path("skills/alpha")),
            Skill("beta", "检查测试", Path("skills/beta/SKILL.md"), Path("skills/beta"), True),
        ]
        self.conversation.skills = catalog
        self.conversation.refresh_skills = catalog.reload
        return catalog

    def test_history_menu_supports_presets_and_all(self):
        original = list(self.conversation.messages)
        for selected, limit in enumerate([10, 20, 50, None]):
            with self.subTest(selected=selected):
                self.console.reset_mock()
                self.console.select_option.return_value = selected
                self.assertIs(handle_command(self.conversation, self.console, "/history"), CommandResult.HANDLED)
                self.console.history.assert_called_once_with(self.conversation.messages, limit)
                self.assertEqual(
                    self.console.select_option.call_args.kwargs["title"], "选择历史范围"
                )
        self.assertEqual(self.conversation.messages, original)

    def test_cancelling_the_history_menu_does_not_show_or_modify_history(self):
        self.console.select_option.return_value = None
        original = list(self.conversation.messages)
        self.assertIs(handle_command(self.conversation, self.console, "/history"), CommandResult.HANDLED)
        self.console.history.assert_not_called()
        self.console.info.assert_called_once_with("已取消查看历史。")
        self.assertEqual(self.conversation.messages, original)

    def test_custom_history_count_retries_invalid_values_then_uses_the_number(self):
        self.console.select_option.return_value = 4
        self.console.read_value.side_effect = ["0", "101", "text", " ", " 37 "]
        self.assertIs(handle_command(self.conversation, self.console, "/history"), CommandResult.HANDLED)
        self.assertEqual(self.console.warning.call_count, 4)
        self.console.history.assert_called_once_with(self.conversation.messages, 37)

    def test_cancelling_a_custom_history_count_does_not_show_history(self):
        self.console.select_option.return_value = 4
        self.console.read_value.side_effect = ["invalid", None]
        self.assertIs(handle_command(self.conversation, self.console, "/history"), CommandResult.HANDLED)
        self.console.history.assert_not_called()
        self.console.info.assert_called_once_with("已取消查看历史。")

    def test_explicit_history_arguments_skip_the_menu(self):
        for argument, limit in [("1", 1), ("20", 20), ("100", 100), ("all", None)]:
            with self.subTest(argument=argument):
                self.console.reset_mock()
                self.assertIs(handle_command(self.conversation, self.console, f"/history {argument}"), CommandResult.HANDLED)
                self.console.select_option.assert_not_called()
                self.console.history.assert_called_once_with(self.conversation.messages, limit)

    def test_invalid_history_arguments_stay_commands_and_do_not_show_a_menu(self):
        for arguments in ("0", "101", "text", "20 extra"):
            with self.subTest(arguments=arguments):
                self.console.reset_mock()
                self.assertIs(handle_command(self.conversation, self.console, f"/history {arguments}"), CommandResult.HANDLED)
                self.console.error.assert_called_once()
                self.console.history.assert_not_called()
                self.console.select_option.assert_not_called()

    def test_empty_history_and_redirected_commands_do_not_wait_for_a_selection(self):
        for messages, interactive in [
            ([], True),
            ([{"role": "system", "content": "note"}], True),
            (self.conversation.messages, False),
        ]:
            with self.subTest(messages=messages, interactive=interactive):
                self.console.reset_mock()
                self.console.interactive = interactive
                self.conversation.messages = messages
                self.assertIs(handle_command(self.conversation, self.console, "/history"), CommandResult.HANDLED)
                self.console.select_option.assert_not_called()
                self.console.history.assert_called_once_with(messages, 10)

    def test_skills_menu_shows_the_selected_metadata_and_usage_without_loading_a_task(self):
        catalog = self.attach_skills()
        original = list(self.conversation.messages)
        self.console.select_option.return_value = 2
        self.assertIs(handle_command(self.conversation, self.console, "/skills"), CommandResult.HANDLED)
        catalog.reload.assert_called_once()
        options = self.console.select_option.call_args.args[1]
        self.assertEqual([option["label"] for option in options], ["查看全部技能", "alpha", "beta"])
        self.console.info.assert_any_call("beta: 检查测试")
        self.console.info.assert_any_call("文件：skills/beta/SKILL.md")
        self.console.info.assert_any_call("用法：/skill:beta <任务描述>")
        self.console.info.assert_any_call("此技能仅在你显式调用时加载，不会由模型自动选择。")
        self.assertEqual(self.conversation.messages, original)

    def test_skills_menu_can_list_every_skill(self):
        self.attach_skills()
        self.console.select_option.return_value = 0
        self.assertIs(handle_command(self.conversation, self.console, "/skills"), CommandResult.HANDLED)
        self.console.info.assert_any_call("alpha: 检查接口")
        self.console.info.assert_any_call("beta: 检查测试")
        self.assertEqual(self.console.info.call_count, 2)

    def test_cancelled_skills_menu_still_surfaces_catalog_diagnostics(self):
        catalog = self.attach_skills()
        catalog.diagnostics = [
            SkillDiagnostic(Path("skills/broken/SKILL.md"), "missing description")
        ]
        self.console.select_option.return_value = None
        self.assertIs(handle_command(self.conversation, self.console, "/skills"), CommandResult.HANDLED)
        self.console.warning.assert_called_once()
        self.console.info.assert_called_once_with("已取消查看技能。")

    def test_explicit_skill_name_and_all_skip_the_menu(self):
        self.attach_skills()
        for arguments in ("alpha", "all"):
            with self.subTest(arguments=arguments):
                self.console.reset_mock()
                self.assertIs(handle_command(self.conversation, self.console, f"/skills {arguments}"), CommandResult.HANDLED)
                self.console.select_option.assert_not_called()
                self.console.info.assert_any_call("alpha: 检查接口")
        self.console.info.assert_any_call("beta: 检查测试")

    def test_invalid_skill_arguments_stay_commands_instead_of_model_prompts(self):
        self.attach_skills()
        for arguments in ("missing", "alpha extra"):
            with self.subTest(arguments=arguments):
                self.console.reset_mock()
                self.assertIs(handle_command(self.conversation, self.console, f"/skills {arguments}"), CommandResult.HANDLED)
                self.console.error.assert_called_once()
                self.console.select_option.assert_not_called()

    def test_disabled_empty_and_redirected_skills_do_not_wait_for_a_selection(self):
        self.assertIs(handle_command(self.conversation, self.console, "/skills"), CommandResult.HANDLED)
        self.console.info.assert_called_once_with("Skills 已关闭")
        self.console.select_option.assert_not_called()

        catalog = self.attach_skills()
        skills = catalog.skills
        catalog.skills = []
        self.console.reset_mock()
        self.assertIs(handle_command(self.conversation, self.console, "/skills"), CommandResult.HANDLED)
        self.console.info.assert_called_once_with("没有发现可用的 skill")
        self.console.select_option.assert_not_called()

        catalog.skills = skills
        self.console.reset_mock()
        self.console.interactive = False
        self.assertIs(handle_command(self.conversation, self.console, "/skills"), CommandResult.HANDLED)
        self.console.select_option.assert_not_called()
        self.console.info.assert_any_call("alpha: 检查接口")
        self.console.info.assert_any_call("beta: 检查测试")

    def test_disabled_catalog_is_reported_as_disabled_instead_of_empty(self):
        catalog = self.attach_skills()
        catalog.enabled = False

        self.assertIs(handle_command(self.conversation, self.console, "/skills"), CommandResult.HANDLED)

        self.console.info.assert_called_once_with("Skills 已关闭")
        catalog.reload.assert_not_called()
        self.console.select_option.assert_not_called()


if __name__ == "__main__":
    unittest.main()
