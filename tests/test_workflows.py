import io
import os
import tempfile
import unittest
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

from commands import CommandResult
from config import DEFAULT_WORKFLOWS_DIR, Settings
from conversation import Conversation
from main import handle_command, run_interactive
from tests.support import make_settings
from tests.test_conversation import FakeClient
from ui import Console
from workflows import WorkflowStore


class WorkflowStoreTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.directory = self.root / "workflows"
        self.store = WorkflowStore(self.directory)

    def test_initialization_and_listing_do_not_create_files(self):
        self.assertEqual(self.store.list(), [])
        self.assertFalse(self.directory.exists())

    def test_new_workflow_preserves_multiline_steps_and_survives_restart(self):
        workflow = self.store.create("  代码审查  ", "  1. 阅读变更\n2. 检查风险\n3. 给出建议\n")

        self.assertEqual(workflow.name, "代码审查")
        self.assertEqual(workflow.path, self.directory / "代码审查.md")
        self.assertEqual(workflow.path.read_text(), workflow.instructions + "\n")
        self.assertEqual(WorkflowStore(self.directory).list(), [workflow])

    def test_plain_markdown_files_are_discovered_and_readme_is_not_a_workflow(self):
        self.directory.mkdir()
        (self.directory / "README.md").write_text("documentation")
        (self.directory / "Beta.md").write_text("# Second workflow\n\n1. Inspect tests")
        (self.directory / "alpha.md").write_text("1. Read files")
        (self.directory / "not-a-workflow.txt").write_text("ignore")

        workflows = self.store.list()

        self.assertEqual([workflow.name for workflow in workflows], ["alpha", "Beta"])
        self.assertEqual(workflows[1].description, "Second workflow")

    def test_duplicate_names_do_not_overwrite_saved_steps(self):
        workflow = self.store.create("Review", "Original steps")
        for name in ("Review", "review", "  Review  "):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.store.create(name, "Replacement steps")

        self.assertEqual(self.store.read(workflow.path).instructions, "Original steps")

    def test_invalid_names_cannot_escape_the_directory_or_replace_documentation(self):
        for name in (
            "",
            " ",
            ".",
            "..",
            "../escape",
            "/escape",
            "a/b",
            "a\\b",
            "a\0b",
            "README",
            "x" * 81,
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.store.create(name, "Steps")
        self.assertFalse(self.directory.exists())

    def test_empty_instructions_are_rejected_without_creating_a_file(self):
        for instructions in ("", " \n\t "):
            with self.subTest(instructions=instructions), self.assertRaises(ValueError):
                self.store.create("Review", instructions)
        self.assertFalse(self.directory.exists())

    def test_existing_invalid_file_is_not_overwritten(self):
        self.directory.mkdir()
        path = self.directory / "Review.md"
        path.write_text("")
        with self.assertRaises(ValueError):
            self.store.create("Review", "New steps")
        self.assertEqual(path.read_text(), "")

    def test_read_uses_the_latest_edited_steps(self):
        workflow = self.store.create("Review", "Original steps")
        workflow.path.write_text("Updated steps\n")

        self.assertEqual(self.store.read(workflow.path).instructions, "Updated steps")
        self.assertEqual(self.store.list()[0].instructions, "Updated steps")

    def test_invalid_files_are_reported_without_hiding_valid_workflows(self):
        valid = self.store.create("Valid", "Steps")
        (self.directory / "Empty.md").write_text(" ")
        (self.directory / "Invalid.md").write_bytes(b"\xff")

        self.assertEqual(self.store.list(), [valid])
        self.assertEqual(len(self.store.diagnostics), 2)
        (self.directory / "Empty.md").unlink()
        (self.directory / "Invalid.md").unlink()
        self.store.list()
        self.assertEqual(self.store.diagnostics, [])

    def test_read_and_discovery_reject_a_symlink_outside_the_store(self):
        outside = self.root / "outside.md"
        outside.write_text("Unselected instructions")
        self.directory.mkdir()
        link = self.directory / "Outside.md"
        link.symlink_to(outside)

        with self.assertRaises(ValueError):
            self.store.read(outside)
        self.assertEqual(self.store.list(), [])
        self.assertEqual(len(self.store.diagnostics), 1)

    def test_a_symlink_loop_does_not_prevent_selecting_valid_workflows(self):
        valid = self.store.create("Valid", "Steps")
        link = self.directory / "Loop.md"
        link.symlink_to(link)

        self.assertEqual(self.store.list(), [valid])
        self.assertEqual(len(self.store.diagnostics), 1)

    def test_import_makes_an_independent_copy_and_keeps_the_source_unchanged(self):
        source = self.root / "source.md"
        source.write_text("1. Inspect\n2. Test\n")

        workflow = self.store.add(source, "Imported")

        self.assertEqual(source.read_text(), "1. Inspect\n2. Test\n")
        source.write_text("Changed source")
        self.assertEqual(self.store.read(workflow.path).instructions, "1. Inspect\n2. Test")

    def test_import_defaults_to_the_source_file_name(self):
        source = self.root / "Review.md"
        source.write_text("Review steps")
        self.assertEqual(self.store.add(source).name, "Review")

    def test_prompt_contains_saved_steps_and_optional_task_context(self):
        workflow = self.store.create("Review", "1. Inspect\n2. Test")

        self.assertIn("workflow「Review」", workflow.prompt())
        self.assertIn(workflow.instructions, workflow.prompt())
        self.assertNotIn("本次任务补充", workflow.prompt(" "))
        self.assertTrue(workflow.prompt("  Check login  ").endswith("本次任务补充：\nCheck login"))


class WorkflowSettingsTests(unittest.TestCase):
    def test_default_and_custom_directory_are_loaded_from_environment(self):
        with (
            tempfile.TemporaryDirectory() as temporary,
            patch("config.load_env"),
            patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test"}, clear=True),
        ):
            self.assertEqual(Settings.from_env().workflows_dir, DEFAULT_WORKFLOWS_DIR)
            os.environ["DEEPSEEK_WORKFLOWS_DIR"] = temporary
            self.assertEqual(Settings.from_env().workflows_dir, Path(temporary).resolve())


class WorkflowCommandTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        settings = make_settings(
            api_key="test",
            base_url="https://example.invalid",
            model="test",
            workspace=self.root,
            session_file=self.root / "session.json",
            max_context_chars=120000,
            keep_recent_turns=6,
            max_task_seconds=3600,
            max_subagents=4,
            templates_dir=self.root / "templates",
            workflows_dir=self.root / "workflows",
        )
        self.chat = Conversation(settings, FakeClient(), None)
        self.chat.messages.extend(
            [
                {"role": "user", "content": "Earlier task"},
                {"role": "assistant", "content": "Earlier answer"},
            ]
        )
        self.previous_messages = deepcopy(self.chat.messages)
        self.console = Mock(spec=Console)
        self.console.interactive = True
        self.console.interrupt_on_escape.side_effect = nullcontext
        self.ask_patch = patch("main.ask")
        self.runner = self.ask_patch.start()
        self.addCleanup(self.ask_patch.stop)

    def test_empty_menu_can_create_a_workflow_without_running_or_clearing_chat(self):
        self.console.select_option.return_value = 0
        self.console.read_value.side_effect = ["代码审查", "1. 阅读变更\n2. 检查风险"]

        self.assertIs(handle_command(self.chat, self.console, "/workflow"), CommandResult.HANDLED)

        self.assertEqual(self.chat.workflows.list()[0].name, "代码审查")
        self.assertEqual(self.chat.messages, self.previous_messages)
        self.runner.assert_not_called()
        options = self.console.select_option.call_args.args[1]
        self.assertEqual(len(options), 1)
        self.assertIn("新建 workflow", options[0]["label"])

    def test_new_command_accepts_a_quoted_name_and_multiline_steps(self):
        self.console.read_value.return_value = "1. Inspect\n2. Test"

        handle_command(self.chat, self.console, '/workflow new "Code review"')

        self.assertEqual(self.chat.workflows.list()[0].name, "Code review")
        self.assertEqual(self.console.read_value.call_count, 1)
        self.runner.assert_not_called()

    def test_creation_can_be_cancelled_at_the_name_or_steps_prompt(self):
        for answers in ([None], ["Review", None]):
            with self.subTest(answers=answers):
                self.console.read_value.side_effect = answers
                handle_command(self.chat, self.console, "/workflow new")
                self.assertEqual(self.chat.workflows.list(), [])
                self.assertEqual(self.chat.messages, self.previous_messages)
        self.runner.assert_not_called()

    def test_invalid_creation_does_not_overwrite_existing_steps(self):
        workflow = self.chat.workflows.create("Review", "Original")
        self.console.read_value.return_value = "New steps"

        handle_command(self.chat, self.console, "/workflow new Review")

        self.console.error.assert_called_once()
        self.assertEqual(self.chat.workflows.read(workflow.path).instructions, "Original")
        self.runner.assert_not_called()

    def test_menu_selects_existing_workflow_and_collects_optional_context(self):
        workflow = self.chat.workflows.create("Review", "1. Inspect\n2. Test")
        self.console.select_option.return_value = 0
        self.console.read_value.return_value = "Check login"

        handle_command(self.chat, self.console, "/workflow")

        self.runner.assert_called_once_with(self.chat, self.console, workflow.prompt("Check login"))
        self.assertEqual(self.chat.messages, self.previous_messages)
        options = self.console.select_option.call_args.args[1]
        self.assertEqual(options[0]["label"], "1. Review")
        self.assertIn("新建 workflow", options[-1]["label"])

    def test_empty_task_context_confirms_running_the_saved_steps(self):
        workflow = self.chat.workflows.create("Review", "Steps")
        self.console.select_option.return_value = 0
        self.console.read_value.return_value = ""

        handle_command(self.chat, self.console, "/workflow")

        self.runner.assert_called_once_with(self.chat, self.console, workflow.prompt())

    def test_menu_and_run_confirmation_can_be_cancelled_without_sending_a_request(self):
        self.chat.workflows.create("Review", "Steps")
        for selected in (None, 0):
            with self.subTest(selected=selected):
                self.console.select_option.return_value = selected
                self.console.read_value.return_value = None
                handle_command(self.chat, self.console, "/workflow")
                self.assertEqual(self.chat.messages, self.previous_messages)
        self.runner.assert_not_called()

    def test_create_is_available_alongside_existing_workflows(self):
        self.chat.workflows.create("Existing", "Original steps")
        self.console.select_option.return_value = 1
        self.console.read_value.side_effect = ["New workflow", "New steps"]

        handle_command(self.chat, self.console, "/workflow")

        self.assertEqual(
            [workflow.name for workflow in self.chat.workflows.list()], ["Existing", "New workflow"]
        )
        self.runner.assert_not_called()

    def test_direct_commands_resolve_number_name_and_quoted_context(self):
        workflow = self.chat.workflows.create("Code review", "Steps")
        for command, context in (
            ("/workflow 1", ""),
            ('/workflow "code REVIEW"', ""),
            ('/workflow "Code review" "Check login" and logout', "Check login and logout"),
        ):
            with self.subTest(command=command):
                self.runner.reset_mock()
                handle_command(self.chat, self.console, command)
                self.runner.assert_called_once_with(
                    self.chat, self.console, workflow.prompt(context)
                )
        self.console.select_option.assert_not_called()
        self.console.read_value.assert_not_called()

    def test_show_displays_saved_steps_without_running_or_collecting_input(self):
        workflow = self.chat.workflows.create("Review", "1. Inspect\n2. Test")

        handle_command(self.chat, self.console, "/workflow show 1")

        self.console.info.assert_any_call(workflow.instructions)
        self.runner.assert_not_called()
        self.console.read_value.assert_not_called()

    def test_import_resolves_paths_relative_to_the_workspace_and_does_not_run(self):
        source = self.root / "source file.md"
        source.write_text("1. Inspect\n2. Test")

        handle_command(self.chat, self.console, '/workflow add "source file.md" "Code review"')

        self.assertEqual(self.chat.workflows.list()[0].name, "Code review")
        self.assertEqual(source.read_text(), "1. Inspect\n2. Test")
        self.runner.assert_not_called()

    def test_invalid_commands_and_missing_files_are_handled_without_model_requests(self):
        for command in (
            "/workflow missing",
            "/workflow 0",
            "/workflow 99999999999999999999999999999999999999",
            "/workflow show",
            "/workflow show missing extra",
            "/workflow new name extra",
            "/workflow new ../escape",
            "/workflow add",
            "/workflow add absent.md",
            '/workflow "unterminated',
        ):
            with self.subTest(command=command):
                self.console.error.reset_mock()
                self.assertIs(handle_command(self.chat, self.console, command), CommandResult.HANDLED)
                self.console.error.assert_called_once()
        self.runner.assert_not_called()
        self.console.read_value.assert_not_called()

    def test_ambiguous_names_are_not_run(self):
        self.chat.workflows.create("Review", "Original")
        (self.chat.workflows.directory / "Review .md").write_text("Other steps")

        handle_command(self.chat, self.console, "/workflow REVIEW")

        self.console.error.assert_called_once()
        self.runner.assert_not_called()

    def test_running_reads_edits_made_after_the_menu_was_opened(self):
        workflow = self.chat.workflows.create("Review", "Original")

        def select(*args, **kwargs):
            workflow.path.write_text("Edited steps")
            return 0

        self.console.select_option.side_effect = select
        self.console.read_value.return_value = ""

        handle_command(self.chat, self.console, "/workflow")

        self.assertIn("Edited steps", self.runner.call_args.args[2])
        self.assertNotIn("Original", self.runner.call_args.args[2])

    def test_invalid_saved_file_emits_a_diagnostic_and_keeps_creation_available(self):
        self.chat.workflows.directory.mkdir()
        (self.chat.workflows.directory / "Empty.md").write_text("")
        self.console.select_option.return_value = None

        handle_command(self.chat, self.console, "/workflow")

        self.console.warning.assert_called_once()
        self.assertEqual(len(self.console.select_option.call_args.args[1]), 1)
        self.runner.assert_not_called()

    def test_redirected_list_and_creation_never_open_interactive_prompts(self):
        self.chat.workflows.create("Review", "Steps")
        self.console.interactive = False

        handle_command(self.chat, self.console, "/workflow")
        handle_command(self.chat, self.console, "/workflow new Name")

        self.console.info.assert_any_call("1. Review · Steps")
        self.console.select_option.assert_not_called()
        self.console.read_value.assert_not_called()
        self.runner.assert_not_called()

    def test_workflow_errors_and_interruptions_return_to_the_chat_loop(self):
        self.chat.workflows.create("Review", "Steps")
        for error in (RuntimeError("failed"), KeyboardInterrupt()):
            with self.subTest(error=error), patch("sys.stdout", io.StringIO()):
                self.runner.side_effect = error
                self.console.prompt.side_effect = ["/workflow Review", "/status", "/exit"]
                self.console.status.reset_mock()
                run_interactive(self.chat, self.console)
                self.console.status.assert_called_once()

    def test_workflow_runs_through_the_normal_conversation_and_saves_its_history(self):
        self.ask_patch.stop()
        workflow = self.chat.workflows.create("Review", "1. Inspect\n2. Report")
        self.chat.client = FakeClient({"role": "assistant", "content": "Completed review"})
        session_id = self.chat.session_id

        handle_command(self.chat, self.console, "/workflow Review Check login")

        self.assertEqual(self.chat.session_id, session_id)
        self.assertEqual(self.chat.messages[: len(self.previous_messages)], self.previous_messages)
        self.assertTrue(
            any(
                message.get("content") == workflow.prompt("Check login")
                for message in self.chat.messages
            )
        )
        self.assertEqual(self.chat.messages[-1]["content"], "Completed review")
        self.assertTrue(self.chat.settings.session_file.exists())
        self.console.task_start.assert_called_once()
        self.console.task_end.assert_called_once_with(True, failed=False, status="completed")
