import io
import json
import os
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

from commands import CommandResult
from config import DEFAULT_TEMPLATES_DIR, Settings
from conversation import Conversation
from main import handle_command
from sessions import SessionStore
from templates import SavedTemplate, TemplateStore
from tests.support import make_settings
from tests.test_conversation import FakeClient, tool_call
from tools import CodingTools
from ui import Console
from usage import CacheStats


class TemplateFixture:
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name).resolve()
        self.active_file = self.workspace / ".deepseek-session.json"
        self.source = self.workspace / "demo-session.ds-agent.json"
        self.messages = [
            {"role": "user", "content": "固定背景"},
            {"role": "assistant", "content": "已了解背景", "reasoning_content": "模板思考"},
        ]
        self.write_source()
        self.template_directory = self.workspace / "templates"
        self.store = TemplateStore(
            SessionStore(self.active_file), self.workspace, directory=self.template_directory
        )

    def write_source(self, **metadata):
        data = {"version": 1, "messages": self.messages, **metadata}
        self.source.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def conversation(self, *responses):
        settings = make_settings(
            api_key="test",
            base_url="https://example.invalid",
            model="test",
            session_file=self.active_file,
            max_context_chars=120000,
            keep_recent_turns=6,
            max_task_seconds=3600,
            max_subagents=4,
            workspace=self.workspace,
            templates_dir=self.template_directory,
        )
        client = FakeClient(*responses)
        client.cache_stats = CacheStats()
        return Conversation(settings, client, CodingTools(self.workspace))

    def existing_conversation(self):
        conversation = self.conversation()
        conversation.messages.extend(
            [{"role": "user", "content": "旧问题"}, {"role": "assistant", "content": "旧回答"}]
        )
        conversation.save()
        return conversation


class TemplateStoreTests(TemplateFixture, unittest.TestCase):
    def test_converted_files_are_discovered_without_being_modified_or_imported(self):
        original = self.source.read_bytes()

        template = self.store.list()[0]

        self.assertEqual(template.name, "demo-session")
        self.assertIsNone(template.template_id)
        self.assertEqual(template.message_count, 2)
        self.assertEqual(self.source.read_bytes(), original)
        self.assertFalse(self.store.directory.exists())

    def test_import_freezes_messages_and_deduplicates_the_source_file(self):
        template = self.store.add(self.source, "基础上下文")
        original = template.path.read_bytes()
        self.messages.append({"role": "user", "content": "源文件后来的改动"})
        self.write_source()

        repeated = self.store.add(self.source, "基础上下文")

        self.assertEqual(repeated.template_id, template.template_id)
        self.assertEqual(template.path.read_bytes(), original)
        self.assertEqual(len(self.store.read(template)["messages"]), 2)
        self.assertEqual(len(self.store.list()), 1)
        self.source.unlink()
        self.assertEqual(len(self.store.read(template)["messages"]), 2)

    def test_import_omits_old_runtime_state_and_usage(self):
        self.write_source(
            session_id="a" * 32,
            cache_stats={"calls": 99},
            tool_runs=[{"state": "pending"}],
            task_state={"title": "旧任务", "plan": None},
            last_outcome={"status": "completed"},
        )

        data = self.store.read(self.store.add(self.source))

        self.assertEqual(data["messages"], self.messages)
        for field in ("session_id", "cache_stats", "tool_runs", "task_state", "last_outcome"):
            self.assertNotIn(field, data)

    def test_template_choice_is_only_kept_in_the_running_instance(self):
        template = self.store.add(self.source)
        self.store.set_default(template.template_id)
        reopened = TemplateStore(
            SessionStore(self.active_file), self.workspace, directory=self.template_directory
        )

        self.assertIsNone(reopened.default())
        self.assertIsNone(reopened.selected_id)
        self.assertEqual(self.store.default().template_id, template.template_id)
        self.store.set_default(None)
        self.assertIsNone(self.store.default())
        self.assertFalse(self.store.settings_file.exists())
        self.assertTrue(template.path.exists())
        self.assertEqual(self.store.diagnostics, [])

    def test_empty_templates_and_duplicate_names_are_rejected(self):
        with self.assertRaises(ValueError):
            self.store.create("empty", [])
        with self.assertRaises(ValueError):
            self.store.create(" ", self.messages)
        template = self.store.create("固定背景", self.messages)
        original = template.path.read_bytes()
        with self.assertRaises(ValueError):
            self.store.create("固定背景", self.messages)
        self.assertEqual(template.path.read_bytes(), original)

    def test_corrupt_template_is_reported_without_hiding_valid_templates(self):
        template = self.store.add(self.source)
        broken = self.store.directory / "broken.json"
        broken.write_text("{bad")

        self.assertEqual(self.store.list(), [template])
        self.assertEqual(len(self.store.diagnostics), 1)
        self.assertIn("broken.json", self.store.diagnostics[0])
        self.assertTrue(broken.exists())

    def test_invalid_default_ids_cannot_escape_the_template_directory(self):
        with self.assertRaises(ValueError):
            self.store.set_default("../outside")
        self.assertIsNone(self.store.default())

    def test_templates_use_readable_filenames_in_the_shared_directory(self):
        template = self.store.add(self.source)

        self.assertEqual(template.path, self.template_directory / "demo-session.json")
        self.assertNotEqual(template.path.parent, self.store.settings_file.parent)

    def test_shared_catalog_has_independent_workspace_defaults(self):
        template = self.store.add(self.source)
        other_workspace = self.workspace / "other-project"
        other_workspace.mkdir()
        other = TemplateStore(
            SessionStore(other_workspace / ".deepseek-session.json"),
            other_workspace,
            directory=self.template_directory,
        )
        self.store.set_default(template.template_id)

        self.assertEqual(other.list(), [template])
        self.assertIsNone(other.default())
        other.set_default(template.template_id)
        self.store.set_default(None)
        self.assertEqual(other.default(), template)
        self.assertNotEqual(other.settings_file, self.store.settings_file)

    def test_legacy_template_and_default_migrate_without_changing_their_ids(self):
        self.store.local_directory.mkdir()
        template_id = "d" * 32
        legacy = self.store.local_directory / f"{template_id}.json"
        data = {
            "version": 1,
            "template_id": template_id,
            "name": "demo-session",
            "source": str(self.source),
            "messages": self.messages,
        }
        legacy.write_text(json.dumps(data, ensure_ascii=False))
        self.store.settings_file.write_text(
            json.dumps({"version": 1, "default_template_id": template_id})
        )
        original_source = self.source.read_bytes()
        original_legacy = legacy.read_bytes()

        template = self.store.list()[0]
        self.assertIsNone(self.store.default())
        self.store.set_default(template.template_id)

        self.assertEqual(template.template_id, template_id)
        self.assertEqual(template.path, self.template_directory / "demo-session.json")
        self.assertEqual(self.store.read(template)["messages"], self.messages)
        self.assertEqual(self.source.read_bytes(), original_source)
        self.assertEqual(legacy.read_bytes(), original_legacy)
        self.assertEqual(self.store.list(), [template])
        edited = self.store.read(template)
        edited["messages"] = [{"role": "user", "content": "正式模板的新版本"}]
        template.path.write_text(json.dumps(edited))
        self.store.default()
        self.assertEqual(self.store.read(template)["messages"], edited["messages"])

    def test_template_names_cannot_escape_the_directory_or_overwrite_colliding_filenames(self):
        escaped = self.store.create("../../outside", self.messages)
        first = self.store.create("基础 A", self.messages)
        second = self.store.create("基础/A", self.messages)

        self.assertEqual(escaped.path.parent, self.template_directory)
        self.assertEqual(first.path.parent, self.template_directory)
        self.assertNotEqual(first.path, second.path)
        self.assertEqual(self.store.read(first)["name"], "基础 A")
        self.assertEqual(self.store.read(second)["name"], "基础/A")

    def test_config_defaults_to_program_directory_and_supports_an_override(self):
        for configured in (None, str(self.template_directory)):
            with self.subTest(configured=configured):
                environment = {"DEEPSEEK_API_KEY": "test"}
                if configured is not None:
                    environment["DEEPSEEK_TEMPLATES_DIR"] = configured
                with (
                    patch("config.load_env"),
                    patch.dict(os.environ, environment, clear=True),
                    patch("config.Path.cwd", return_value=self.workspace),
                ):
                    settings = Settings.from_env()
                expected = DEFAULT_TEMPLATES_DIR if configured is None else self.template_directory
                self.assertEqual(settings.templates_dir, expected)


class ConversationTemplateTests(TemplateFixture, unittest.TestCase):
    def test_template_starts_a_new_session_and_resets_runtime_state(self):
        conversation = self.existing_conversation()
        old_id = conversation.session_id
        conversation.tools.task_title = "旧任务"
        conversation.tools.task_plan = {
            "title": "旧任务",
            "steps": [{"step": "检查", "status": "completed"}],
        }
        conversation.tool_runs = [{"state": "completed"}]
        conversation.last_outcome = {"status": "completed"}
        conversation.client.cache_stats.record(
            {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 50}}, "solve"
        )
        original = self.source.read_bytes()

        template = conversation.use_template(conversation.templates.list()[0])

        self.assertNotEqual(conversation.session_id, old_id)
        self.assertEqual(conversation.messages, self.messages)
        self.assertEqual(conversation.template_id, template.template_id)
        self.assertEqual(conversation.templates.default().template_id, template.template_id)
        self.assertEqual(conversation.tool_runs, [])
        self.assertEqual(conversation.tools.task_title, "")
        self.assertIsNone(conversation.tools.task_plan)
        self.assertIsNone(conversation.last_outcome)
        self.assertEqual(conversation.client.cache_stats.snapshot()["calls"], 0)
        self.assertEqual(self.source.read_bytes(), original)
        previous = conversation.sessions.read(conversation.sessions.snapshot_path(old_id))
        self.assertEqual(previous["messages"][0]["content"], "旧问题")
        self.assertEqual(previous["cache_stats"]["calls"], 1)

    def test_many_new_sessions_share_only_the_fixed_seed_not_each_others_changes(self):
        conversation = self.conversation()
        template = conversation.use_template(conversation.templates.list()[0])
        original = template.path.read_bytes()
        branch_ids = []
        for index in range(3):
            branch_ids.append(conversation.session_id)
            self.assertEqual(conversation.messages, self.messages)
            conversation.messages.append({"role": "user", "content": f"分支问题 {index}"})
            conversation.save()
            conversation.clear()

        self.assertEqual(len(set(branch_ids)), 3)
        self.assertNotIn(conversation.session_id, branch_ids)
        self.assertEqual(conversation.messages, self.messages)
        self.assertEqual(template.path.read_bytes(), original)
        for index, session_id in enumerate(branch_ids):
            saved = conversation.sessions.read(conversation.sessions.snapshot_path(session_id))
            self.assertEqual(saved["messages"][-1]["content"], f"分支问题 {index}")
        conversation.resume(branch_ids[0])
        self.assertEqual(conversation.messages[-1]["content"], "分支问题 0")
        self.assertEqual(conversation.template_id, template.template_id)

    def test_restart_starts_blank_and_preserves_the_template_branch_for_resume(self):
        template = self.store.add(self.source)
        conversation = self.conversation()
        self.assertEqual(conversation.messages, [])
        conversation.use_template(template)
        self.assertEqual(conversation.messages, self.messages)
        self.assertEqual(conversation.template_id, template.template_id)
        session_id = conversation.session_id
        conversation.messages.append({"role": "user", "content": "已经开始的新问题"})
        conversation.save()

        restarted = self.conversation()

        self.assertNotEqual(restarted.session_id, session_id)
        self.assertEqual(restarted.messages, [])
        self.assertIsNone(restarted.template_id)
        self.assertIsNone(restarted.templates.default())
        self.assertFalse(self.active_file.exists())
        restarted.resume(session_id)
        self.assertEqual(restarted.messages[-1]["content"], "已经开始的新问题")
        self.assertEqual(len(restarted.messages), 3)
        self.assertIsNone(restarted.templates.default())
        restarted.clear()
        self.assertEqual(restarted.messages, [])

    def test_old_persisted_template_selection_is_not_used_on_startup(self):
        template = self.store.add(self.source)
        self.store.settings_file.parent.mkdir()
        self.store.settings_file.write_text(
            json.dumps({"version": 1, "default_template_id": template.template_id})
        )

        conversation = self.conversation()

        self.assertEqual(conversation.messages, [])
        self.assertIsNone(conversation.templates.default())
        self.assertFalse(self.active_file.exists())

    def test_disabling_default_does_not_change_current_branch_until_new(self):
        conversation = self.conversation()
        template = conversation.use_template(conversation.templates.list()[0])
        current = deepcopy(conversation.messages)
        session_id = conversation.session_id
        conversation.templates.set_default(None)
        self.assertEqual(conversation.messages, current)
        self.assertEqual(conversation.session_id, session_id)

        conversation.clear()

        self.assertEqual(conversation.messages, [])
        self.assertIsNone(conversation.template_id)
        self.assertNotEqual(conversation.session_id, session_id)
        self.assertFalse(self.active_file.exists())
        self.assertTrue(template.path.exists())

    def test_missing_default_does_not_clear_the_existing_conversation(self):
        conversation = self.conversation()
        template = conversation.use_template(conversation.templates.list()[0])
        conversation.messages.append({"role": "user", "content": "当前分支的问题"})
        conversation.save()
        original = self.active_file.read_bytes()
        session_id = conversation.session_id
        template.path.unlink()

        with self.assertRaises(RuntimeError):
            conversation.clear()

        self.assertEqual(self.active_file.read_bytes(), original)
        self.assertEqual(conversation.session_id, session_id)
        self.assertEqual(conversation.messages[-1]["content"], "当前分支的问题")

    def test_incomplete_tool_calls_are_repaired_without_executing_or_changing_template(self):
        self.messages = [
            {"role": "user", "content": "背景任务"},
            {"role": "assistant", "tool_calls": [tool_call()]},
        ]
        self.write_source()
        original = self.source.read_bytes()
        conversation = self.conversation()
        with patch.object(conversation.tools, "execute") as execute:
            template = conversation.use_template(conversation.templates.list()[0])
            conversation.clear()

        execute.assert_not_called()
        self.assertIsNotNone(conversation.recovery_notice)
        self.assertEqual(conversation.messages[-1]["role"], "tool")
        self.assertEqual(self.store.read(template)["messages"], self.messages)
        self.assertEqual(self.source.read_bytes(), original)

    def test_followup_request_contains_template_context_but_not_the_previous_branch(self):
        conversation = self.conversation({"role": "assistant", "content": "新分支回答"})
        conversation.messages.append({"role": "user", "content": "不属于模板的旧问题"})
        conversation.use_template(conversation.templates.list()[0])

        self.assertEqual(conversation.ask("基于背景继续讨论"), "新分支回答")
        request = conversation.client.requests[-1]["messages"]
        for message in self.messages:
            self.assertIn(message, request)
        self.assertNotIn({"role": "user", "content": "不属于模板的旧问题"}, request)

    def test_current_session_file_cannot_also_be_used_as_the_template_source(self):
        conversation = self.existing_conversation()
        original = self.active_file.read_bytes()
        template = SavedTemplate(None, "active", self.active_file, 2)

        with self.assertRaises(ValueError):
            conversation.use_template(template)

        self.assertEqual(self.active_file.read_bytes(), original)
        self.assertIsNone(conversation.templates.default())

    def test_imported_template_source_cannot_be_overwritten_as_the_active_session(self):
        conversation = self.existing_conversation()
        template = conversation.templates.add(self.active_file, "active-copy")
        original = self.active_file.read_bytes()

        with self.assertRaises(ValueError):
            conversation.use_template(template)

        self.assertEqual(self.active_file.read_bytes(), original)
        self.assertIsNone(conversation.templates.default())

    def test_editing_the_formal_template_updates_new_sessions_without_changing_existing_ones(self):
        conversation = self.conversation()
        template = conversation.use_template(conversation.templates.list()[0])
        old_id = conversation.session_id
        original_messages = deepcopy(conversation.messages)
        original_source = self.source.read_bytes()
        data = self.store.read(template)
        updated_messages = [
            {"role": "user", "content": "修改后的固定背景"},
            {"role": "assistant", "content": "已了解新版背景"},
        ]
        data["messages"] = updated_messages
        template.path.write_text(json.dumps(data, ensure_ascii=False))

        self.assertEqual(conversation.messages, original_messages)
        conversation.clear()

        self.assertEqual(conversation.messages, updated_messages)
        self.assertNotEqual(conversation.session_id, old_id)
        self.assertEqual(conversation.template_id, template.template_id)
        previous = conversation.sessions.read(conversation.sessions.snapshot_path(old_id))
        self.assertEqual(previous["messages"], original_messages)
        self.assertEqual(self.source.read_bytes(), original_source)


class TemplateCommandTests(TemplateFixture, unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.console = Mock(spec=Console)
        self.console.interactive = True
        self.chat = self.existing_conversation()

    def test_menu_selects_converted_file_and_sets_default_for_future_new_commands(self):
        old_id = self.chat.session_id
        self.console.select_option.return_value = 1

        self.assertIs(handle_command(self.chat, self.console, "/template"), CommandResult.HANDLED)

        self.assertNotEqual(self.chat.session_id, old_id)
        self.assertEqual(self.chat.messages, self.messages)
        self.assertEqual(self.console.select_option.call_args.kwargs["title"], "选择对话模板")
        self.assertEqual(self.console.reset_session.call_args.kwargs["session_label"], "模板新会话")
        self.console.history.assert_called_once_with(self.messages, 10)
        template_id = self.chat.template_id
        for command in ("/new", "/clear"):
            previous_id = self.chat.session_id
            self.chat.messages.append({"role": "user", "content": "只属于上一分支"})
            self.assertIs(handle_command(self.chat, self.console, command), CommandResult.HANDLED)
            self.assertEqual(self.chat.messages, self.messages)
            self.assertNotEqual(self.chat.session_id, previous_id)
            self.assertEqual(self.chat.template_id, template_id)

    def test_cancelling_menu_does_not_import_or_modify_the_current_session(self):
        self.console.select_option.return_value = None
        original = self.active_file.read_bytes()

        self.assertIs(handle_command(self.chat, self.console, "/template"), CommandResult.HANDLED)

        self.assertEqual(self.active_file.read_bytes(), original)
        self.assertFalse(self.store.directory.exists())
        self.console.reset_session.assert_not_called()
        self.console.info.assert_called_once_with("已取消选择模板，当前会话不变。")

    def test_explicit_name_number_and_id_each_start_an_independent_session(self):
        template = self.store.add(self.source)
        for selector in ("demo-session", "1", template.template_id):
            with self.subTest(selector=selector):
                self.console.reset_mock()
                previous_id = self.chat.session_id
                self.assertIs(handle_command(self.chat, self.console, f"/template {selector}"), CommandResult.HANDLED)
                self.assertNotEqual(self.chat.session_id, previous_id)
                self.assertEqual(self.chat.messages, self.messages)
                self.console.select_option.assert_not_called()

    def test_add_quoted_path_and_name_registers_without_switching_or_setting_default(self):
        source = self.workspace / "转换记录" / "我的对话.json"
        source.parent.mkdir()
        source.write_bytes(self.source.read_bytes())
        current_id = self.chat.session_id
        command = '/template add "转换记录/我的对话.json" "基础 上下文"'

        self.assertIs(handle_command(self.chat, self.console, command), CommandResult.HANDLED)

        saved = next(template for template in self.store.list() if template.template_id)
        self.assertEqual(saved.name, "基础 上下文")
        self.assertEqual(self.store.read(saved)["messages"], self.messages)
        self.assertEqual(self.chat.session_id, current_id)
        self.assertIsNone(self.store.default())
        self.console.reset_session.assert_not_called()

    def test_save_current_dialogue_freezes_it_without_switching(self):
        current_id = self.chat.session_id
        original = deepcopy(self.chat.messages)

        self.assertIs(handle_command(self.chat, self.console, '/template save "我的基础对话"'), CommandResult.HANDLED)

        saved = next(template for template in self.store.list() if template.template_id)
        self.chat.messages.append({"role": "user", "content": "保存模板后的问题"})
        self.assertEqual(self.store.read(saved)["messages"], original)
        self.assertEqual(self.chat.session_id, current_id)
        self.assertIsNone(self.store.default())
        self.console.reset_session.assert_not_called()

    def test_off_immediately_starts_blank_and_preserves_previous_dialogue(self):
        self.assertIs(handle_command(self.chat, self.console, "/template 1"), CommandResult.HANDLED)
        current_id = self.chat.session_id
        self.console.reset_mock()

        self.assertIs(handle_command(self.chat, self.console, "/template off"), CommandResult.HANDLED)

        self.assertNotEqual(self.chat.session_id, current_id)
        self.assertEqual(self.chat.messages, [])
        self.assertIsNone(self.chat.template_id)
        self.assertIsNone(self.chat.templates.default())
        self.console.reset_session.assert_called_once()
        previous = self.chat.sessions.read(self.chat.sessions.snapshot_path(current_id))
        self.assertEqual(previous["messages"], self.messages)
        self.assertIs(handle_command(self.chat, self.console, "/new"), CommandResult.HANDLED)
        self.assertEqual(self.chat.messages, [])

    def test_redirected_menu_only_lists_templates_without_switching(self):
        self.console.interactive = False
        current_id = self.chat.session_id

        self.assertIs(handle_command(self.chat, self.console, "/template"), CommandResult.HANDLED)

        self.assertEqual(self.chat.session_id, current_id)
        self.console.select_option.assert_not_called()
        self.console.reset_session.assert_not_called()
        self.assertFalse(self.store.directory.exists())
        self.assertIn("demo-session", str(self.console.info.call_args_list))

    def test_invalid_template_commands_stay_commands_and_do_not_modify_the_session(self):
        for arguments in ("2", "missing", "add", "save", "off extra", '"unclosed'):
            with self.subTest(arguments=arguments):
                self.console.reset_mock()
                original = self.active_file.read_bytes()
                self.assertIs(handle_command(self.chat, self.console, f"/template {arguments}"), CommandResult.HANDLED)
                self.console.error.assert_called_once()
                self.console.reset_session.assert_not_called()
                self.assertEqual(self.active_file.read_bytes(), original)

    def test_missing_default_is_reported_without_resetting_the_ui_or_current_chat(self):
        self.assertIs(handle_command(self.chat, self.console, "/template 1"), CommandResult.HANDLED)
        self.chat.templates.default().path.unlink()
        original = self.active_file.read_bytes()
        self.console.reset_mock()

        self.assertIs(handle_command(self.chat, self.console, "/new"), CommandResult.HANDLED)

        self.console.error.assert_called_once()
        self.console.reset_session.assert_not_called()
        self.assertEqual(self.active_file.read_bytes(), original)

    def test_blank_menu_option_saves_current_dialogue_and_clears_template(self):
        self.assertIs(handle_command(self.chat, self.console, "/template 1"), CommandResult.HANDLED)
        current_id = self.chat.session_id
        template = self.chat.templates.default()
        original = template.path.read_bytes()
        self.console.reset_mock()
        self.console.select_option.return_value = 0

        self.assertIs(handle_command(self.chat, self.console, "/template"), CommandResult.HANDLED)

        options = self.console.select_option.call_args.args[1]
        self.assertIn("空白（无模板）", options[0]["label"])
        self.assertEqual(self.chat.messages, [])
        self.assertIsNone(self.chat.templates.selected_id)
        self.assertIsNone(self.chat.template_id)
        self.assertEqual(template.path.read_bytes(), original)
        previous = self.chat.sessions.read(self.chat.sessions.snapshot_path(current_id))
        self.assertEqual(previous["messages"], self.messages)

    def test_zero_starts_blank_without_deleting_templates(self):
        self.assertIs(handle_command(self.chat, self.console, "/template 1"), CommandResult.HANDLED)
        template = self.chat.templates.default()

        self.assertIs(handle_command(self.chat, self.console, "/template 0"), CommandResult.HANDLED)

        self.assertEqual(self.chat.messages, [])
        self.assertIsNone(self.chat.templates.default())
        self.assertTrue(template.path.exists())

    def test_blank_option_is_available_even_when_no_templates_exist(self):
        self.source.unlink()
        self.console.select_option.return_value = 0

        self.assertIs(handle_command(self.chat, self.console, "/template"), CommandResult.HANDLED)

        self.assertEqual(len(self.console.select_option.call_args.args[1]), 1)
        self.assertEqual(self.chat.messages, [])
        self.console.error.assert_not_called()

    def test_failed_save_cannot_clear_the_current_template_or_dialogue(self):
        self.assertIs(handle_command(self.chat, self.console, "/template 1"), CommandResult.HANDLED)
        current_id = self.chat.session_id
        template_id = self.chat.templates.selected_id
        original = self.active_file.read_bytes()
        self.console.reset_mock()
        with patch.object(self.chat, "save", side_effect=OSError("disk full")):
            self.assertIs(handle_command(self.chat, self.console, "/template off"), CommandResult.HANDLED)

        self.assertEqual(self.chat.session_id, current_id)
        self.assertEqual(self.chat.templates.selected_id, template_id)
        self.assertEqual(self.active_file.read_bytes(), original)
        self.console.reset_session.assert_not_called()
        self.console.error.assert_called_once()

    def test_new_template_screen_does_not_mislabel_the_branch_as_resumed(self):
        output = io.StringIO()
        with patch("sys.stdout", output):
            self.assertIs(handle_command(self.chat, Console(), "/template 1"), CommandResult.HANDLED)

        self.assertIn("模板新会话", output.getvalue())
        self.assertIn("固定背景", output.getvalue())
        self.assertNotIn("已恢复", output.getvalue())
        self.assertNotIn("旧问题", output.getvalue())
