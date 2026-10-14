import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

from config import Settings
from conversation import Conversation
from deepseek_client import DeepSeekClient
from main import run_interactive, update_effort
from tests.support import make_settings
from tools import CodingTools
from ui import Console


def response(request, **kwargs):
    body = json.loads(request.data)
    message = {"role": "assistant", "content": "done"}
    choice = {"finish_reason": "stop", "delta" if body["stream"] else "message": message}
    payload = json.dumps({"choices": [choice]})
    return io.BytesIO((f"data: {payload}\n" if body["stream"] else payload).encode())


class EffortTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.settings = make_settings(
            api_key="test",
            base_url="https://api.cline.bot/api/v1",
            model="cline-pass/deepseek-v4.1-flash",
            session_file=root / "session.json",
            max_context_chars=120000,
            keep_recent_turns=6,
            max_task_seconds=3600,
            max_subagents=1,
            workspace=root,
        )

    def conversation(self, tools=None):
        return Conversation(self.settings, DeepSeekClient(self.settings), tools)

    def test_environment_default_override_and_invalid_value(self):
        with (
            patch("config.load_env"),
            patch.dict("os.environ", {"DEEPSEEK_API_KEY": "test"}, clear=True),
        ):
            self.assertEqual(self.settings.reasoning_effort, "max")
            self.assertEqual(Settings.from_env().reasoning_effort, "max")
            for value, expected in [
                (" MAX ", "max"),
                ("off", "off"),
                ("default", "max"),
                ("", "max"),
                ("  ", "max"),
            ]:
                with (
                    self.subTest(value=value),
                    patch.dict("os.environ", {"DEEPSEEK_REASONING_EFFORT": value}),
                ):
                    self.assertEqual(Settings.from_env().reasoning_effort, expected)
            with (
                patch.dict("os.environ", {"DEEPSEEK_REASONING_EFFORT": "medium"}),
                self.assertRaisesRegex(ValueError, "无效的推理强度"),
            ):
                Settings.from_env()

    def test_explicit_env_file_keeps_current_directory_as_workspace(self):
        config_dir = Path(self.temporary.name) / "config"
        workspace = Path(self.temporary.name) / "workspace"
        config_dir.mkdir()
        workspace.mkdir()
        env_file = config_dir / ".env"
        env_file.write_text(
            "DEEPSEEK_API_KEY=from-fixed-config\nDEEPSEEK_MODEL=fixed-model\n",
            encoding="utf-8",
        )

        with (
            patch.dict(
                "os.environ",
                {"DEEPSEEK_ENV_FILE": str(env_file)},
                clear=True,
            ),
            patch("pathlib.Path.cwd", return_value=workspace),
        ):
            settings = Settings.from_env()

        self.assertEqual(settings.api_key, "from-fixed-config")
        self.assertEqual(settings.model, "fixed-model")
        self.assertEqual(settings.workspace, workspace)
        self.assertEqual(settings.session_file, (workspace / ".deepseek-session.json").resolve())

    def test_request_parameters_for_direct_and_gateway_routes(self):
        for url, gateway in [
            ("https://api.deepseek.com", False),
            ("https://api.cline.bot/api/v1", True),
            ("https://openrouter.ai/api/v1", True),
        ]:
            for effort in (None, "default", "off", "low", "high", "max"):
                with self.subTest(url=url, effort=effort):
                    settings = replace(self.settings, base_url=url, reasoning_effort=effort)
                    with patch("http_transport.open_url", side_effect=response) as opened:
                        DeepSeekClient(settings).complete([])
                    body = json.loads(opened.call_args.args[0].data)
                    if effort == "off":
                        self.assertEqual(body["thinking"], {"type": "disabled"})
                        self.assertNotIn("reasoning_effort", body)
                        if gateway:
                            self.assertEqual(body["reasoning"], {"enabled": False})
                        else:
                            self.assertNotIn("reasoning", body)
                    else:
                        expected = "max" if effort in (None, "default") else effort
                        self.assertEqual(body["thinking"], {"type": "enabled"})
                        if gateway:
                            self.assertEqual(
                                body["reasoning"], {"enabled": True, "effort": expected}
                            )
                            self.assertNotIn("reasoning_effort", body)
                        else:
                            self.assertEqual(body["reasoning_effort"], expected)
                            self.assertNotIn("reasoning", body)

    def test_every_phase_preserves_the_selected_effort(self):
        definitions = [{"type": "function", "function": {"name": "read"}}]
        for url in (
            "https://api.deepseek.com",
            "https://api.cline.bot/api/v1",
            "https://openrouter.ai/api/v1",
        ):
            for effort in ("default", "max", "high", "low", "off"):
                with self.subTest(url=url, effort=effort):
                    client = DeepSeekClient(
                        replace(self.settings, base_url=url, reasoning_effort=effort)
                    )
                    with patch("http_transport.open_url", side_effect=response) as opened:
                        client.complete([], purpose="summary")
                        client.complete([], tools=definitions, purpose="final")
                        client.complete([])
                    summary, final, solver = [
                        json.loads(call.args[0].data) for call in opened.call_args_list
                    ]
                    expected = "max" if effort == "default" else effort
                    for body in (summary, final, solver):
                        self.assertEqual(
                            body["thinking"], {"type": "disabled" if effort == "off" else "enabled"}
                        )
                        if url == "https://api.deepseek.com":
                            if effort == "off":
                                self.assertNotIn("reasoning_effort", body)
                            else:
                                self.assertEqual(body["reasoning_effort"], expected)
                        else:
                            self.assertEqual(
                                body["reasoning"],
                                {"enabled": False}
                                if effort == "off"
                                else {"enabled": True, "effort": expected},
                            )
                    self.assertEqual(final["tools"], definitions)
                    self.assertNotIn("tools", summary)

    def test_interactive_changes_reach_requests_without_entering_history(self):
        conversation = self.conversation()
        console = Mock(spec=Console)
        console.prompt.side_effect = [
            "/effort",
            "/effort\tHIGH",
            "one",
            "/effort max",
            "/effort",
            "two",
            "/effort off",
            "three",
            "/effort default",
            "four",
            "/exit",
        ]
        with patch("http_transport.open_url", side_effect=response) as opened:
            run_interactive(conversation, console)
        bodies = [json.loads(call.args[0].data) for call in opened.call_args_list]
        self.assertEqual(
            [body.get("reasoning") for body in bodies],
            [
                {"enabled": True, "effort": "high"},
                {"enabled": True, "effort": "max"},
                {"enabled": False},
                {"enabled": True, "effort": "max"},
            ],
        )
        console.info.assert_any_call("当前推理强度：max")
        console.error.assert_not_called()
        saved = json.loads(self.settings.session_file.read_text())["messages"]
        self.assertEqual(
            [m["content"] for m in saved if m["role"] == "user"], ["one", "two", "three", "four"]
        )

    def test_invalid_commands_preserve_effort_and_do_not_send_requests(self):
        conversation = self.conversation()
        conversation.set_reasoning_effort("low")
        console = Mock(spec=Console)
        console.prompt.side_effect = ["/effort medium", "/effort max extra", "/exit"]
        with patch("http_transport.open_url") as opened:
            run_interactive(conversation, console)
        opened.assert_not_called()
        self.assertEqual(console.error.call_count, 2)
        self.assertEqual(conversation.settings.reasoning_effort, "low")
        self.assertEqual(conversation.client.settings.reasoning_effort, "low")
        self.assertFalse(self.settings.session_file.exists())

    def test_effort_menu_highlights_the_current_value_and_applies_the_selection(self):
        conversation = self.conversation()
        conversation.set_reasoning_effort("low")
        console = Mock(spec=Console)
        console.interactive = True
        console.select_option.return_value = 1

        update_effort(conversation, console, [])

        menu = console.select_option.call_args
        self.assertEqual(menu.kwargs["default_index"], 2)
        self.assertEqual(menu.args[1][2]["label"], "low（当前）")
        self.assertEqual(conversation.settings.reasoning_effort, "high")
        self.assertEqual(conversation.client.settings.reasoning_effort, "high")
        self.assertEqual(console.reasoning_effort, "high")
        self.assertEqual(conversation.messages, [])
        console.error.assert_not_called()

    def test_cancelling_the_effort_menu_preserves_the_current_configuration(self):
        conversation = self.conversation()
        conversation.set_reasoning_effort("low")
        console = Mock(spec=Console)
        console.interactive = True
        console.select_option.return_value = None

        update_effort(conversation, console, [])

        self.assertEqual(conversation.settings.reasoning_effort, "low")
        self.assertEqual(conversation.client.settings.reasoning_effort, "low")
        console.success.assert_not_called()
        console.info.assert_called_once_with("已取消选择，推理强度保持 low。")

    def test_effort_menu_can_restore_the_default_and_direct_commands_still_work(self):
        conversation = self.conversation()
        conversation.set_reasoning_effort("off")
        console = Mock(spec=Console)
        console.interactive = True
        console.select_option.return_value = 4

        update_effort(conversation, console, [])

        self.assertEqual(conversation.settings.reasoning_effort, "max")
        self.assertEqual(console.reasoning_effort, "max")
        console.select_option.reset_mock()
        update_effort(conversation, console, ["off"])
        console.select_option.assert_not_called()
        self.assertEqual(conversation.settings.reasoning_effort, "off")

    def test_delegation_uses_effort_changed_after_tools_were_attached(self):
        tools = CodingTools(self.settings.workspace)
        conversation = self.conversation(tools)
        conversation.set_reasoning_effort("max")
        call = {
            "function": {
                "name": "delegate",
                "arguments": json.dumps(
                    {
                        "tasks": [
                            {
                                "task": "Compute 1 + 1",
                                "expected_output": "A number",
                                "mode": "reason",
                            }
                        ],
                    }
                ),
            }
        }
        with patch("http_transport.open_url", side_effect=response) as opened:
            result = json.loads(tools.execute(call))
        self.assertEqual(result["status"], "ok")
        self.assertEqual(opened.call_count, 1)
        body = json.loads(opened.call_args.args[0].data)
        self.assertEqual(body["reasoning"], {"enabled": True, "effort": "max"})

    def test_delegation_defaults_to_max_without_an_effort_command(self):
        tools = CodingTools(self.settings.workspace)
        self.conversation(tools)
        call = {
            "function": {
                "name": "delegate",
                "arguments": json.dumps(
                    {
                        "tasks": [
                            {
                                "task": "Compute 1 + 1",
                                "expected_output": "A number",
                                "mode": "reason",
                            }
                        ],
                    }
                ),
            }
        }
        with patch("http_transport.open_url", side_effect=response) as opened:
            result = json.loads(tools.execute(call))
        self.assertEqual(result["status"], "ok")
        body = json.loads(opened.call_args.args[0].data)
        self.assertEqual(body["thinking"], {"type": "enabled"})
        self.assertEqual(body["reasoning"], {"enabled": True, "effort": "max"})


if __name__ == "__main__":
    unittest.main()
