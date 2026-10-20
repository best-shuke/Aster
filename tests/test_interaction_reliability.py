import io
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import main
from commands import CommandResult, parse_command
from conversation import Conversation
from deepseek_client import CompletionError
from tests.support import make_settings
from tests.test_conversation import FakeClient
from tools import CodingTools
from ui import Console


class InteractionReliabilityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.settings = make_settings(
            api_key="test",
            base_url="https://example.invalid",
            model="test",
            workspace=self.workspace,
            session_file=self.workspace / "session.json",
            max_context_chars=120000,
            keep_recent_turns=6,
            max_task_seconds=60,
            max_subagents=1,
            mcp_enabled=False,
            websearch_enabled=False,
        )
        self.console = Mock(spec=Console)
        self.console.interactive = False

    def run_main(self, arguments, **patches):
        with (
            patch("sys.argv", ["main.py", *arguments]),
            patch("main.load_settings", return_value=self.settings),
            patch("main.DeepSeekClient", return_value=FakeClient()),
            patch("main.Console", return_value=self.console),
            ExitStack() as stack,
        ):
            for name, replacement in patches.items():
                stack.enter_context(patch(f"main.{name}", replacement))
            return main.main()

    def conversation_stub(self):
        return SimpleNamespace(
            settings=self.settings,
            messages=[],
            skills=None,
            tools=None,
            last_environment_id=None,
            environment_id=None,
            recovery_notice=None,
            last_outcome={"status": "completed"},
            context_status=Mock(return_value={}),
            client=SimpleNamespace(cache_stats=None),
            ask=Mock(),
        )

    def test_multiline_command_like_paste_remains_a_task(self):
        for prompt in (
            "/effort high\n解释这个命令",
            "/environment review\nkeep --yes",
            "/workflow show\nreview",
            "/history\nall",
            "/resume latest\r新的任务",
        ):
            with self.subTest(prompt=prompt):
                self.assertIsNone(parse_command(prompt))
        self.assertEqual(parse_command(" /history all\r\n").arguments, ("all",))

    def test_cli_preserves_shell_argument_boundaries_for_commands(self):
        for arguments in (
            ["/template", "add", "some file.json", 'name with "quotes"'],
            ["/workflow", "show", "review login"],
        ):
            with self.subTest(arguments=arguments):
                handler = Mock(return_value=CommandResult.HANDLED)
                self.assertEqual(self.run_main(arguments, handle_command=handler), 0)
                parsed = parse_command(handler.call_args.args[2])
                self.assertEqual(parsed.arguments, tuple(arguments[1:]))

    def test_cli_preserves_quoted_arguments_in_a_single_command_string(self):
        handler = Mock(return_value=CommandResult.HANDLED)
        self.run_main(['/template add "some file.json" "sample name"'], handle_command=handler)
        parsed = parse_command(handler.call_args.args[2])
        self.assertEqual(parsed.arguments, ("add", "some file.json", "sample name"))

    def test_cli_environment_uses_the_same_argument_parser(self):
        use_environment = Mock(return_value=True)
        self.assertEqual(
            self.run_main(
                ["/env", "show", "program:review login"], use_environment=use_environment
            ),
            0,
        )
        self.assertEqual(use_environment.call_args.args[2], ["show", "program:review login"])

    def test_cli_reports_malformed_environment_quotes_without_a_traceback(self):
        use_environment = Mock()
        self.assertEqual(self.run_main(['/env "unfinished'], use_environment=use_environment), 2)
        self.console.error.assert_called_once()
        use_environment.assert_not_called()

    def test_cli_multiline_environment_text_does_not_switch_environments(self):
        handler = Mock(return_value=CommandResult.HANDLED)
        use_environment = Mock()
        prompt = "/env show\nprogram:review"
        self.assertEqual(
            self.run_main([prompt], handle_command=handler, use_environment=use_environment), 0
        )
        handler.assert_called_once()
        self.assertEqual(handler.call_args.args[2], prompt)
        use_environment.assert_not_called()

    def test_cli_runtime_failure_is_reported_without_a_traceback(self):
        ask_task = Mock(side_effect=CompletionError("task_deadline", "task timed out"))
        self.assertEqual(self.run_main(["new task"], ask=ask_task), 1)
        self.console.error.assert_called_once_with(ask_task.side_effect)

    def test_cli_interruption_has_a_nonzero_exit_code(self):
        ask_task = Mock(side_effect=KeyboardInterrupt())
        self.assertEqual(self.run_main(["new task"], ask=ask_task), 130)
        self.console.info.assert_called_once()

    def test_ctrl_c_in_chat_cancels_input_instead_of_exiting(self):
        conversation = self.conversation_stub()
        self.console.prompt.side_effect = [KeyboardInterrupt(), "/exit"]
        with (
            patch("main.sync_environment_ui"),
            patch("main.show_skill_diagnostics"),
        ):
            main.run_interactive(conversation, self.console)
        self.assertEqual(self.console.prompt.call_count, 2)
        conversation.ask.assert_not_called()

    def test_refresh_error_keeps_the_command_prompt_available(self):
        conversation = self.conversation_stub()
        conversation.context_status.side_effect = [{}, ValueError("bad context")]
        self.console.prompt.return_value = "/exit"
        with (
            patch("main.sync_environment_ui"),
            patch("main.show_skill_diagnostics"),
        ):
            main.run_interactive(conversation, self.console)
        self.console.error.assert_called_once()
        self.console.prompt.assert_called_once()

    def test_failed_task_does_not_reuse_the_previous_completion_status(self):
        conversation = self.conversation_stub()
        conversation.ask.side_effect = ValueError("preflight rejected")
        with self.assertRaisesRegex(ValueError, "preflight rejected"):
            main.ask(conversation, self.console, "new task")
        self.console.task_end.assert_called_once_with(False, failed=True, status="error")

    def test_interrupted_task_is_displayed_as_paused(self):
        conversation = self.conversation_stub()
        conversation.ask.side_effect = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            main.ask(conversation, self.console, "new task")
        self.console.task_end.assert_called_once_with(False, failed=False, status="paused")

    def test_context_refresh_failure_does_not_mask_a_task_error_or_skip_cleanup(self):
        conversation = self.conversation_stub()
        conversation.context_status.side_effect = [{}, ValueError("context refresh failed")]
        conversation.ask.side_effect = RuntimeError("original task error")
        with self.assertRaisesRegex(RuntimeError, "original task error"):
            main.ask(conversation, self.console, "new task")
        self.console.task_end.assert_called_once_with(False, failed=True, status="error")
        self.console.error.assert_called_once()

    def test_failed_dashboard_cannot_be_overridden_by_a_completed_status(self):
        with patch("sys.stdout", io.StringIO()):
            console = Console()
            console.task_start("new task")
            console.task_end(False, failed=True, status="completed")
        self.assertEqual(console._task_state, "error")

    def test_escape_setup_interruption_restores_terminal_without_changing_signals(self):
        console = Console()
        console.interactive = True
        input_reader = Mock()
        raw_mode = MagicMock()
        raw_mode.__enter__.side_effect = KeyboardInterrupt()
        input_reader.raw_mode.return_value = raw_mode
        with (
            patch("ui.create_input", return_value=input_reader),
            patch("ui.signal.signal") as set_handler,
            self.assertRaises(KeyboardInterrupt),
            console.interrupt_on_escape(),
        ):
            self.fail("interrupted setup must not start the task")
        raw_mode.__exit__.assert_called_once_with(None, None, None)
        input_reader.close.assert_called_once()
        set_handler.assert_not_called()
        self.assertIsNone(console._task_input_monitor)

    def test_messages_only_copies_history_without_runtime_or_template_state(self):
        conversation = Conversation(self.settings, FakeClient(), CodingTools(self.workspace))
        self.addCleanup(conversation.close)
        messages = [
            {"role": "user", "content": "template question"},
            {"role": "assistant", "content": "template answer"},
        ]
        template = conversation.templates.create("seed", messages)
        conversation.use_template(template)
        conversation.tools.task_title = "old task"
        conversation.tools.task_plan = {
            "title": "old task",
            "steps": [{"step": "finished", "status": "completed"}],
        }
        conversation.last_outcome = {"status": "completed"}
        conversation.tool_runs = [{"state": "completed", "tool_call_id": "old-call"}]
        conversation.save()
        previous_id = conversation.session_id
        conversation.resume(previous_id, messages_only=True)
        self.assertNotEqual(conversation.session_id, previous_id)
        self.assertEqual(conversation.messages, messages)
        self.assertIsNone(conversation.template_id)
        self.assertIsNone(conversation.last_outcome)
        self.assertEqual(conversation.tool_runs, [])
        self.assertEqual(conversation.tools.task_title, "")
        self.assertIsNone(conversation.tools.task_plan)
        previous = conversation.sessions.read(conversation.sessions.snapshot_path(previous_id))
        self.assertEqual(previous["template_id"], template.template_id)
        self.assertEqual(previous["last_outcome"]["status"], "completed")
        restored = Conversation(self.settings, FakeClient(), None)
        self.addCleanup(restored.close)
        self.assertEqual(restored.messages, messages)


if __name__ == "__main__":
    unittest.main()
