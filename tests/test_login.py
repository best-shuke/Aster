import base64
import io
import json
import os
import stat
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import auth
import login_ui
import main
from auth import AUTH_NAMESPACE, AuthStore, BrowserLogin, DeviceCode, LoginExpired
from chatgpt_client import ChatGPTClient, responses_input, responses_tool
from commands import CommandName, parse_command
from config import CHATGPT_BASE_URL, Login, LoginRequired, Settings
from context_budget import LOCAL_METADATA
from conversation import Conversation
from deepseek_client import CompletionError
from tests.support import make_settings
from tests.test_conversation import FakeClient
from ui import _quota_text


def jwt(claims):
    def encode(value):
        return base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()

    return f"{encode({'alg': 'none'})}.{encode(claims)}.signature"


def token_response(*, expires_in=3600, account="account-1", refresh="refresh-1", plan="plus"):
    namespace = {"chatgpt_account_id": account, "chatgpt_plan_type": plan}
    return {
        "access_token": jwt({"exp": time.time() + expires_in, AUTH_NAMESPACE: namespace}),
        "id_token": jwt({"email": "user@example.com", AUTH_NAMESPACE: namespace}),
        "refresh_token": refresh,
        "expires_in": expires_in,
    }


def sse(*events):
    return io.BytesIO(
        b"".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n".encode() for e in events)
    )


class TempDirTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = AuthStore(self.root / "aster" / "auth.json")
        self.addCleanup(auth.forget_account, self.store.path)
        # A missing env file keeps every load_env() caller away from the repo's real .env.
        isolated = {
            "DEEPSEEK_AUTH_FILE": str(self.store.path),
            "DEEPSEEK_ENV_FILE": str(self.root / "none.env"),
        }
        environment = patch.dict(os.environ, isolated, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    def save_chatgpt(self, **overrides):
        tokens = auth.account_tokens(token_response(**overrides))
        models = [
            {
                "slug": "gpt-test",
                "name": "GPT Test",
                "description": "Test model",
                "context_window": 258400,
                "reasoning_levels": ["low", "medium", "high", "xhigh"],
                "priority": 1,
            }
        ]
        self.store.update(
            provider="chatgpt", chatgpt={**tokens, "model": "gpt-test", "models": models}
        )
        return tokens


class AuthTests(TempDirTest):
    def test_token_response_yields_account_identity_plan_and_expiry(self):
        tokens = auth.account_tokens(token_response(plan="pro"))
        self.assertEqual(tokens["account_id"], "account-1")
        self.assertEqual(tokens["email"], "user@example.com")
        self.assertEqual(auth.account_label(tokens), "user@example.com · Pro")
        self.assertGreater(tokens["expires_at"], time.time() + 3000)
        with self.assertRaisesRegex(auth.AuthError, "账户信息"):
            auth.account_tokens({"access_token": jwt({}), "refresh_token": "r"})

    def test_pasted_callback_accepts_url_query_or_code_and_checks_state(self):
        url = "http://localhost:1455/auth/callback?code=abc&state=s1"
        self.assertEqual(auth.parse_callback(url, "s1"), "abc")
        self.assertEqual(auth.parse_callback("?code=abc&state=s1", "s1"), "abc")
        self.assertEqual(auth.parse_callback("  raw-code  ", "s1"), "raw-code")
        for text, message in [
            (url, "state"),
            ("http://localhost:1455/auth/callback?error=access_denied", "拒绝"),
            ("http://localhost:1455/auth/callback?state=s1", "授权码"),
            ("", "没有输入"),
        ]:
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, message):
                auth.parse_callback(text, "s2" if message == "state" else "s1")

    def test_browser_login_uses_pkce_and_the_local_callback(self):
        login = auth.browser_login()
        self.assertIn("code_challenge_method=S256", login.url)
        self.assertIn("redirect_uri=http%3A%2F%2Flocalhost%3A1455%2Fauth%2Fcallback", login.url)
        self.assertIn(f"state={login.state}", login.url)
        self.assertNotIn(login.verifier, login.url)

    def test_store_is_private_and_resolves_each_saved_choice(self):
        self.assertIsNone(self.store.login())
        self.store.update(provider="env")
        self.assertEqual(self.store.login(), Login("env"))
        mode = stat.S_IMODE(self.store.path.stat().st_mode)
        self.assertEqual(mode & 0o077, 0)
        self.store.update(
            provider="api", api={"base_url": "https://x.example/v1", "api_key": "k", "model": "m"}
        )
        self.assertEqual(self.store.login(), Login("api", "m", "https://x.example/v1", "k"))
        self.save_chatgpt()
        login = self.store.login()
        self.assertEqual(
            (login.provider, login.model, login.context_window), ("chatgpt", "gpt-test", 258400)
        )
        self.assertEqual(login.account, "user@example.com · Plus")
        self.store.update(provider=None, chatgpt=None, api=None)
        self.assertIsNone(self.store.login())

    def test_expiring_token_is_refreshed_once_and_saved(self):
        self.save_chatgpt(expires_in=60)
        fresh = auth.account_tokens(token_response(refresh="refresh-2"))
        account = auth.chatgpt_account(self.store.path)
        with patch("auth.refresh_tokens", return_value=fresh) as refresh:
            self.assertEqual(account.credentials(), (fresh["access_token"], "account-1"))
            self.assertEqual(account.credentials()[0], fresh["access_token"])
        refresh.assert_called_once()
        self.assertEqual(self.store.read()["chatgpt"]["refresh_token"], "refresh-2")
        self.assertEqual(self.store.read()["chatgpt"]["model"], "gpt-test")

    def test_rejected_token_reuses_a_refresh_another_process_already_saved(self):
        old = self.save_chatgpt()
        newer = auth.account_tokens(token_response(refresh="refresh-3"))
        self.store.update(chatgpt={**self.store.read()["chatgpt"], **newer})
        account = auth.chatgpt_account(self.store.path)
        with patch("auth.refresh_tokens") as refresh:
            account.refresh(old["access_token"])
            self.assertEqual(account.credentials()[0], newer["access_token"])
        refresh.assert_not_called()

    def test_refused_refresh_asks_the_user_to_log_in_again(self):
        self.save_chatgpt(expires_in=10)
        error = urllib.error.HTTPError(
            auth.TOKEN_URL, 400, "Bad", {}, io.BytesIO(b'{"error": "invalid_grant"}')
        )
        with patch("auth._post", side_effect=error), self.assertRaisesRegex(LoginExpired, "/login"):
            auth.chatgpt_account(self.store.path).credentials()

    def test_callback_server_receives_the_browser_redirect(self):
        server = auth.CallbackServer("s1")
        if not server.listening:
            self.skipTest("port 1455 is in use")
        try:
            url = "http://127.0.0.1:1455/auth/callback?code=c1&state=s1"
            with urllib.request.urlopen(url, timeout=5) as response:
                self.assertIn("登录成功", response.read().decode())
            self.assertTrue(server.received.wait(1))
            self.assertEqual(server.code, "c1")
        finally:
            server.close()


class ResponsesTests(TempDirTest):
    def client(self, **settings):
        values = {
            "model": "gpt-test",
            "base_url": CHATGPT_BASE_URL,
            "api_key": "",
            "auth_file": self.store.path,
            "max_output_tokens": 131072,
            "max_context_chars": 120000,
            "request_retries": 0,
            "reasoning_effort": "high",
        }
        values.update(settings)
        return ChatGPTClient(SimpleNamespace(**values))

    def test_history_converts_to_responses_items(self):
        reasoning = {"type": "reasoning", "summary": [], "encrypted_content": "opaque"}
        instructions, items = responses_input(
            [
                {"role": "system", "content": "base rules"},
                {"role": "user", "content": "task"},
                {
                    "role": "assistant",
                    "content": "checking",
                    "reasoning_content": "summary text",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "read", "arguments": '{"path":"a"}'},
                        }
                    ],
                    LOCAL_METADATA: {"responses_reasoning": [reasoning], "usage": {}},
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "file text"},
                {"role": "system", "content": "late note"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look"},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
                    ],
                },
            ]
        )
        self.assertEqual(instructions, "base rules")
        self.assertEqual(
            [item["type"] for item in items],
            [
                "message",
                "reasoning",
                "message",
                "function_call",
                "function_call_output",
                "message",
                "message",
            ],
        )
        self.assertEqual(items[1], reasoning)
        self.assertEqual(items[2]["content"], [{"type": "output_text", "text": "checking"}])
        self.assertEqual(items[3]["call_id"], "call_1")
        self.assertEqual(
            items[4], {"type": "function_call_output", "call_id": "call_1", "output": "file text"}
        )
        self.assertEqual(items[5]["role"], "developer")
        self.assertEqual(
            items[6]["content"][1], {"type": "input_image", "image_url": "data:image/png;base64,AA"}
        )
        tool = responses_tool(
            {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}
        )
        self.assertEqual(tool["name"], "read")
        self.assertFalse(tool["strict"])

    def test_stream_events_build_one_validated_assistant_message(self):
        response = sse(
            {"type": "response.created", "response": {}},
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "reasoning", "id": "rs_1"},
            },
            {
                "type": "response.reasoning_summary_text.delta",
                "item_id": "rs_1",
                "summary_index": 0,
                "delta": "Plan.",
            },
            {
                "type": "response.reasoning_summary_text.delta",
                "item_id": "rs_1",
                "summary_index": 1,
                "delta": "Check.",
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [{"type": "summary_text", "text": "Plan."}],
                    "encrypted_content": "enc",
                },
            },
            {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "Reading "},
            {"type": "response.output_text.delta", "item_id": "msg_1", "delta": "the file."},
            {
                "type": "response.output_item.added",
                "output_index": 2,
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_9",
                    "name": "read",
                    "arguments": "",
                },
            },
            {
                "type": "response.function_call_arguments.delta",
                "item_id": "fc_1",
                "delta": '{"path":',
            },
            {
                "type": "response.output_item.done",
                "output_index": 2,
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_9",
                    "name": "read",
                    "arguments": '{"path":"a"}',
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "usage": {
                        "input_tokens": 100,
                        "input_tokens_details": {"cached_tokens": 80},
                        "output_tokens": 20,
                        "output_tokens_details": {"reasoning_tokens": 5},
                        "total_tokens": 120,
                    }
                },
            },
        )
        streamed, usage = [], []
        message = ChatGPTClient._read_stream(response, streamed.append, on_usage=usage.append)
        self.assertEqual(message["content"], "Reading the file.")
        self.assertEqual("".join(streamed), "Reading the file.")
        self.assertEqual(message["reasoning_content"], "Plan.\n\nCheck.")
        self.assertEqual(
            message["tool_calls"],
            [
                {
                    "id": "call_9",
                    "type": "function",
                    "function": {"name": "read", "arguments": '{"path":"a"}'},
                }
            ],
        )
        self.assertEqual(
            message[LOCAL_METADATA]["responses_reasoning"],
            [
                {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "Plan."}],
                    "encrypted_content": "enc",
                }
            ],
        )
        self.assertEqual(
            usage[-1]["usage"],
            {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "prompt_tokens_details": {"cached_tokens": 80},
                "completion_tokens_details": {"reasoning_tokens": 5},
            },
        )

    def test_incomplete_and_failed_responses_are_not_answers(self):
        cases = [
            (
                {
                    "type": "response.incomplete",
                    "response": {"incomplete_details": {"reason": "max_output_tokens"}},
                },
                "output_limit",
            ),
            (
                {
                    "type": "response.failed",
                    "response": {
                        "error": {"code": "context_length_exceeded", "message": "too long"}
                    },
                },
                "context_overflow",
            ),
            ({"type": "error", "error": {"code": "rate_limit_exceeded"}}, "http_429"),
        ]
        for terminal, code in cases:
            response = sse(
                {"type": "response.output_text.delta", "item_id": "m", "delta": "partial"}, terminal
            )
            with self.subTest(code=code), self.assertRaises(CompletionError) as raised:
                ChatGPTClient._read_stream(response, lambda _: None)
            self.assertEqual(raised.exception.code, code)
        with self.assertRaises(CompletionError) as raised:
            ChatGPTClient._read_stream(
                sse({"type": "response.output_text.delta", "item_id": "m", "delta": "x"}),
                lambda _: None,
            )
        self.assertEqual(raised.exception.code, "incomplete_stream")

    def test_request_body_targets_the_codex_backend(self):
        self.save_chatgpt()
        client = self.client()
        tools = [
            {"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}
        ]
        messages = [{"role": "system", "content": "rules"}, {"role": "user", "content": "hi"}]
        body = client._request_body(messages, tools, True, None)
        self.assertEqual(body["instructions"], "rules")
        self.assertFalse(body["store"])
        self.assertTrue(body["stream"])
        self.assertEqual(body["include"], ["reasoning.encrypted_content"])
        self.assertEqual(body["reasoning"], {"effort": "high", "summary": "auto"})
        self.assertNotIn("max_output_tokens", body)
        self.assertEqual(body["tools"][0]["name"], "read")
        self.assertEqual(
            body["prompt_cache_key"],
            client._request_body(messages, None, False, 50)["prompt_cache_key"],
        )
        request = client._http_request(body)
        self.assertEqual(request.full_url, f"{CHATGPT_BASE_URL}/responses")
        self.assertEqual(request.get_header("Chatgpt-account-id"), "account-1")
        self.assertEqual(request.get_header("Session_id"), body["prompt_cache_key"])
        self.assertEqual(len(body["prompt_cache_key"]), 36)

    def test_effort_maps_to_levels_the_model_supports(self):
        self.save_chatgpt()
        for effort, expected in [
            ("off", "low"),
            ("low", "low"),
            ("high", "high"),
            ("max", "xhigh"),
        ]:
            with self.subTest(effort=effort):
                self.assertEqual(self.client(reasoning_effort=effort)._effort(), expected)
        self.assertEqual(self.client(model="unknown", reasoning_effort="max")._effort(), "high")

    def test_rejected_token_is_refreshed_and_the_request_retried_once(self):
        old = self.save_chatgpt()
        fresh = auth.account_tokens(token_response(refresh="refresh-2"))
        client = self.client()
        tokens = []

        def perform(request, *args):
            tokens.append(request.get_header("Authorization"))
            if len(tokens) == 1:
                raise urllib.error.HTTPError(
                    request.full_url, 401, "Unauthorized", {}, io.BytesIO(b"{}")
                )
            return {"role": "assistant", "content": "ok"}

        with (
            patch("auth.refresh_tokens", return_value=fresh),
            patch.object(client, "_perform_request", side_effect=perform),
        ):
            self.assertEqual(client.complete([{"role": "user", "content": "hi"}])["content"], "ok")
        self.assertEqual(
            tokens, [f"Bearer {old['access_token']}", f"Bearer {fresh['access_token']}"]
        )

    def test_usage_limit_and_request_errors_explain_themselves(self):
        self.save_chatgpt()
        client = self.client()
        limit = urllib.error.HTTPError(
            "u",
            429,
            "Too Many",
            {},
            io.BytesIO(
                json.dumps(
                    {
                        "error": {
                            "type": "usage_limit_reached",
                            "plan_type": "plus",
                            "resets_in_seconds": 7200,
                        }
                    }
                ).encode()
            ),
        )
        error = client._http_error(limit)
        self.assertEqual(error.code, "usage_limit")
        self.assertFalse(error.retryable)
        self.assertIn("2 小时", str(error))
        bad = urllib.error.HTTPError(
            "u", 400, "Bad", {}, io.BytesIO(b'{"detail": "Unsupported model"}')
        )
        self.assertIn("Unsupported model", str(client._http_error(bad)))
        client._observe_headers(
            {
                "X-Codex-Primary-Used-Percent": "9",
                "x-codex-primary-window-minutes": "300",
                "x-codex-secondary-used-percent": "1",
                "x-codex-secondary-window-minutes": "10080",
            }
        )
        self.assertEqual(_quota_text(client.account.quota), "5 小时 已用 9% · 7 天 已用 1%")


class SettingsTests(TempDirTest):
    def test_login_selects_endpoint_credentials_and_context_window(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "env-key", "DEEPSEEK_MODEL": "env-model"}):
            settings = Settings.from_env()
            self.assertEqual(
                (settings.provider, settings.api_key, settings.model),
                ("api", "env-key", "env-model"),
            )
            api = Settings.from_env(Login("api", "m", "https://x.example/v1/", "k"))
            self.assertEqual(
                (api.base_url, api.api_key, api.account), ("https://x.example/v1", "k", "x.example")
            )
            chatgpt = Settings.from_env(
                Login("chatgpt", "gpt-test", context_window=258400, account="a · Plus")
            )
        self.assertEqual(
            (chatgpt.provider, chatgpt.base_url, chatgpt.api_key), ("chatgpt", CHATGPT_BASE_URL, "")
        )
        self.assertEqual(chatgpt.context_window_tokens, 258400)
        self.assertEqual(chatgpt.auth_file, self.store.path)
        with self.assertRaises(LoginRequired):
            Settings.from_env()

    def test_switch_login_keeps_the_conversation_and_changes_the_client(self):
        settings = make_settings(
            api_key="old-key",
            base_url="https://old.example/v1",
            model="old",
            workspace=self.root,
            session_file=self.root / "session.json",
            max_context_chars=120000,
            keep_recent_turns=6,
            max_task_seconds=60,
            max_subagents=1,
        )
        conversation = Conversation(settings, FakeClient(), None)
        conversation.messages = [{"role": "user", "content": "keep me"}]
        client = FakeClient()
        conversation.switch_login(Login("api", "new", "https://new.example/v1", "new-key"), client)
        self.assertIs(conversation.client, client)
        self.assertIs(client.settings, conversation.settings)
        self.assertEqual(
            (conversation.settings.model, conversation.settings.api_key), ("new", "new-key")
        )
        self.assertEqual(conversation._baseline_settings.model, "new")
        self.assertEqual(conversation.environments.baseline.base_url, "https://new.example/v1")
        self.assertEqual(conversation.messages, [{"role": "user", "content": "keep me"}])
        conversation.team = object()
        with self.assertRaisesRegex(RuntimeError, "env exit"):
            conversation.switch_login(Login("env"), FakeClient())

    def test_login_commands_take_no_arguments(self):
        self.assertEqual(parse_command("/login").name, CommandName.LOGIN)
        self.assertEqual(parse_command("/logout").name, CommandName.LOGOUT)
        self.assertIsNone(parse_command("/login now"))


class ScriptedConsole:
    def __init__(self, selections=(), values=()):
        self.selections, self.values = list(selections), list(values)
        self.questions, self.prompts, self.messages = [], [], []

    def select_option(self, question, options, *, default_index=0, title=""):
        self.questions.append((title, [option["label"] for option in options], default_index))
        choice = self.selections.pop(0)
        return default_index if choice == "default" else choice

    def read_value(self, prompt, *, secret=False, until=None):
        self.prompts.append((prompt, secret))
        return self.values.pop(0)

    def info(self, message):
        self.messages.append(message)

    success = warning = info

    def error(self, error):
        self.messages.append(str(error))


class LoginFlowTests(TempDirTest):
    def test_first_run_offers_the_environment_configuration_by_default(self):
        console = ScriptedConsole(["default"])
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "k", "DEEPSEEK_MODEL": "env-model"}):
            login = login_ui.run_login(console, self.store, first_run=True)
        title, labels, default = console.questions[0]
        self.assertIn("欢迎", title)
        self.assertEqual(labels[default], "沿用环境变量配置")
        self.assertEqual(login, Login("env"))
        self.assertEqual(self.store.read()["provider"], "env")
        cancelled = ScriptedConsole([None])
        self.assertIsNone(login_ui.run_login(cancelled, self.store))
        self.assertEqual(len(cancelled.questions[0][1]), 2)  # No variables, no third option.
        self.assertEqual(self.store.read()["provider"], "env")

    def test_api_login_verifies_the_key_and_saves_the_chosen_model(self):
        console = ScriptedConsole([1, 1], ["https://api.example.com/v1/", "bad-key", "good-key"])
        statuses = [(401, []), (200, ["model-a", "model-b"])]
        with patch("login_ui.check_api", side_effect=statuses) as check:
            login = login_ui.run_login(console, self.store)
        self.assertEqual(login, Login("api", "model-b", "https://api.example.com/v1", "good-key"))
        self.assertEqual([prompt[1] for prompt in console.prompts], [False, True, True])
        self.assertEqual(check.call_args.args, ("https://api.example.com/v1", "good-key"))

    def test_device_login_waits_for_approval_then_saves_account_and_model(self):
        console = ScriptedConsole([0, 1, 0])
        device = DeviceCode("device-1", "ABCD-1234", 1, time.time() + 60)
        tokens = auth.account_tokens(token_response())
        models = [
            {
                "slug": "gpt-a",
                "name": "A",
                "description": "",
                "context_window": 1000,
                "reasoning_levels": [],
                "priority": 1,
            }
        ]
        with (
            patch("login_ui.request_device_code", return_value=device),
            patch("login_ui.poll_device_code", side_effect=[None, tokens]) as poll,
            patch("login_ui.list_models", return_value=models),
            patch("login_ui.time.sleep"),
        ):
            login = login_ui.run_login(console, self.store)
        self.assertEqual(poll.call_count, 2)
        self.assertTrue(any("ABCD-1234" in message for message in console.messages))
        self.assertEqual(
            (login.provider, login.model, login.context_window), ("chatgpt", "gpt-a", 1000)
        )
        self.assertEqual(self.store.read()["chatgpt"]["refresh_token"], "refresh-1")

    def test_browser_login_accepts_a_pasted_redirect(self):
        class NoServer:
            listening, code, error = False, None, None

            def __init__(self, state):
                self.received = threading.Event()

            def close(self):
                pass

        console = ScriptedConsole(
            [0, 0, 0], ["not a url", "http://localhost:1455/auth/callback?code=c&state=s"]
        )
        tokens = auth.account_tokens(token_response())
        with (
            patch(
                "login_ui.browser_login",
                return_value=BrowserLogin("https://auth.example/x", "s", "v"),
            ),
            patch("login_ui.CallbackServer", NoServer),
            patch("login_ui.exchange_code", return_value=tokens) as exchange,
            patch("login_ui.list_models", side_effect=auth.AuthError("offline")),
            patch("login_ui._remote_session", return_value=True),
            patch("builtins.print"),
        ):
            console.values.append("")  # Model name: accept the default.
            login = login_ui.run_login(console, self.store)
        exchange.assert_called_once_with("c", "v", auth.REDIRECT_URI)
        self.assertEqual(login.model, "gpt-5.5")

    def test_logout_removes_saved_credentials_from_running_clients_too(self):
        self.save_chatgpt()
        account = auth.chatgpt_account(self.store.path)
        account.credentials()
        console = ScriptedConsole()
        self.assertTrue(login_ui.logout(console, self.store))
        self.assertIsNone(self.store.login())
        self.assertNotIn("chatgpt", self.store.read())
        with self.assertRaisesRegex(LoginExpired, "/login"):
            account.credentials()

    def test_startup_asks_only_on_an_interactive_first_run(self):
        terminal = SimpleNamespace(isatty=lambda: True)
        piped = SimpleNamespace(isatty=lambda: False)
        api = Login("api", "m", "https://x.example/v1", "k")
        with patch("main.run_login", return_value=api) as ask, patch("main.Console"):
            with patch("sys.stdin", piped), patch("sys.stderr", io.StringIO()) as errors:
                self.assertIsNone(main.load_settings(choose=False, one_shot=False))
            self.assertIn("尚未登录", errors.getvalue())
            ask.assert_not_called()
            with patch("sys.stdin", terminal), patch("sys.stdout", terminal):
                self.assertEqual(main.load_settings(choose=False, one_shot=False).model, "m")
                self.assertTrue(ask.call_args.kwargs["first_run"])
                self.store.update(provider="env")
                with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "env-key"}):
                    ask.reset_mock()
                    self.assertEqual(
                        main.load_settings(choose=False, one_shot=False).api_key, "env-key"
                    )
                    ask.assert_not_called()
                    self.assertEqual(main.load_settings(choose=True, one_shot=True).model, "m")
                    self.assertFalse(ask.call_args.kwargs["first_run"])


if __name__ == "__main__":
    unittest.main()
