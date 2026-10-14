import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock

from command_ui import present_command
from commands import (
    CommandActions,
    CommandName,
    CommandResult,
    EffectKind,
    execute_command,
    parse_command,
)


class CommandBoundaryTests(unittest.TestCase):
    def test_empty_text_unknown_commands_and_skill_commands_remain_prompts(self):
        for prompt in ("", "  ", "ordinary 'text", "/unknown", "/skill:review task"):
            with self.subTest(prompt=prompt):
                self.assertIsNone(parse_command(prompt))
                execution = execute_command(parse_command(prompt), None)
                self.assertIs(execution.result, CommandResult.NOT_COMMAND)
                self.assertEqual(execution.effects, ())

    def test_aliases_have_one_canonical_command(self):
        for alias, canonical in (
            ("/quit", "/exit"),
            ("/clear", "/new"),
            ("/todo", "/tasks"),
            ("/env", "/environment"),
        ):
            with self.subTest(alias=alias):
                self.assertEqual(parse_command(alias), parse_command(canonical))

    def test_quoted_arguments_preserve_spaces_and_escaped_quotes(self):
        command = parse_command('/template add "some file.json" "a \\"name\\""')
        self.assertIs(command.name, CommandName.TEMPLATE)
        self.assertEqual(command.arguments, ("add", "some file.json", 'a "name"'))
        self.assertIsNone(command.error)

    def test_malformed_quoting_is_a_handled_error_without_domain_execution(self):
        conversation = Mock()
        command = parse_command('/environment "unfinished')
        execution = execute_command(command, conversation)
        self.assertIs(execution.result, CommandResult.HANDLED)
        self.assertIs(execution.effects[0].kind, EffectKind.ERROR)
        self.assertIsInstance(execution.effects[0].payload, ValueError)
        self.assertEqual(conversation.mock_calls, [])

    def test_no_argument_commands_with_arguments_keep_the_existing_prompt_behavior(self):
        for command in ("/exit now", "/new extra", "/tasks extra", "/status extra"):
            with self.subTest(command=command):
                self.assertIsNone(parse_command(command))

    def test_exit_has_an_explicit_result_without_effects(self):
        execution = execute_command(parse_command("/exit"), None)
        self.assertIs(execution.result, CommandResult.EXIT)
        self.assertEqual(execution.effects, ())

    def test_new_executes_without_a_console_and_returns_a_view_effect(self):
        conversation = Mock()
        execution = execute_command(parse_command("/new"), conversation)
        conversation.clear.assert_called_once_with()
        self.assertIs(execution.result, CommandResult.HANDLED)
        self.assertIs(execution.effects[0].kind, EffectKind.NEW_SESSION)

    def test_failed_execution_returns_an_error_effect(self):
        conversation = Mock()
        conversation.clear.side_effect = RuntimeError("failed reset")
        execution = execute_command(parse_command("/new"), conversation)
        self.assertIs(execution.result, CommandResult.HANDLED)
        self.assertEqual(str(execution.effects[0].payload), "failed reset")
        self.assertIs(execution.effects[0].kind, EffectKind.ERROR)

    def test_interactive_commands_defer_selection_without_mutating_domain_state(self):
        conversation = Mock()
        for name in (
            "history",
            "resume",
            "template",
            "environment",
            "workflow",
            "skills",
            "effort",
        ):
            with self.subTest(name=name):
                execution = execute_command(parse_command(f"/{name}"), conversation)
                self.assertIs(execution.effects[0].kind, EffectKind.INTERACTION)
                self.assertEqual(conversation.mock_calls, [])

    def test_presenter_dispatches_the_parsed_selection_request(self):
        conversation, console, interaction = Mock(), Mock(), Mock()
        execution = execute_command(parse_command('/template "sample name"'), conversation)
        present_command(
            execution,
            conversation,
            console,
            help_text="help",
            interactions={CommandName.TEMPLATE: interaction},
        )
        interaction.assert_called_once_with(conversation, console, ["sample name"])
        self.assertEqual(conversation.mock_calls, [])

    def test_prompt_execution_collects_text_before_presentation(self):
        conversation = Mock()
        conversation.settings.prompt_file = Path("synthetic.md")
        conversation.system_message = {"role": "system", "content": "synthetic instructions"}
        execution = execute_command(parse_command("/prompt"), conversation)
        conversation.reload_prompt.assert_called_once_with()
        conversation.reset_mock()
        console = Mock()
        with redirect_stdout(io.StringIO()) as output:
            present_command(execution, conversation, console, help_text="help", interactions={})
        self.assertIn("synthetic instructions", output.getvalue())
        self.assertEqual(conversation.mock_calls, [])

    def test_domain_actions_validate_inputs_without_ui_dependencies(self):
        conversation = Mock()
        actions = CommandActions(conversation)
        for arguments in ([], ["save"], ["add", "file", "name", "extra"]):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                actions.save_template(arguments)
        with self.assertRaises(ValueError):
            actions.set_effort(["low", "high"])
        self.assertEqual(conversation.mock_calls, [])
        actions.reset(blank_template=True)
        conversation.clear_template.assert_called_once_with()
        actions.resume("session-id")
        conversation.resume.assert_called_once_with("session-id")


if __name__ == "__main__":
    unittest.main()
