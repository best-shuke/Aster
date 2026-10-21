import io
import json
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import auth
import http_transport
from config import CHATGPT_BASE_URL, Login, Settings
from conversation import Conversation
from tests.support import make_settings
from tests.test_conversation import FakeClient
from tests.test_login import token_response
from tool_runtime import MAX_RESULT_CHARS, ExecutionContext
from tools import CodingTools
from websearch import ChatGPTWebSearchClient, WebSearchClient, websearch_client


def settings(workspace, **overrides):
    values = dict(
        api_key="cline-key",
        base_url="https://api.cline.bot/api/v1",
        model="main-model",
        workspace=workspace,
        session_file=workspace / "session.json",
        max_context_chars=120_000,
        keep_recent_turns=6,
        max_task_seconds=3600,
        max_subagents=4,
    )
    values.update(overrides)
    return make_settings(**values)


def response():
    return {
        "success": True,
        "data": {
            "query": "Python latest stable release 2026",
            "durationMs": 716,
            "results": [
                {"title": "Download Python", "url": "https://www.python.org/downloads/"},
                {"title": "Python versions", "url": "https://devguide.python.org/versions/"},
            ],
        },
    }


def call(**arguments):
    return {
        "id": "call-search",
        "type": "function",
        "function": {
            "name": "websearch",
            "arguments": json.dumps({"query": "Python latest stable release 2026", **arguments}),
        },
    }


class WebSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        self.settings = settings(self.workspace)
        self.client = WebSearchClient(self.settings)
        self.tools = CodingTools(self.workspace)
        self.tools.attach_websearch(self.client)

    def execute(self, payload=None, **arguments):
        with patch.object(
            self.client.opener,
            "open",
            return_value=io.BytesIO(
                json.dumps(payload if payload is not None else response()).encode()
            ),
        ):
            return json.loads(self.tools.execute(call(**arguments)))

    def test_request_matches_cline_native_endpoint_and_reuses_key(self):
        with patch.object(
            self.client.opener, "open", return_value=io.BytesIO(json.dumps(response()).encode())
        ) as opened:
            result = self.client.search(
                " Python latest stable release 2026 ",
                allowed_domains=[" python.org ", "python.org", " "],
                _context=ExecutionContext(time.monotonic() + 10),
            )
        request = opened.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.cline.bot/api/v1/search/websearch")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer cline-key")
        self.assertEqual(
            json.loads(request.data),
            {"query": "Python latest stable release 2026", "allowed_domains": ["python.org"]},
        )
        self.assertGreater(opened.call_args.kwargs["timeout"], 0)
        self.assertLessEqual(opened.call_args.kwargs["timeout"], 10)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["allowed_domains"], ["python.org"])

    def test_custom_base_and_key_and_blocked_domains(self):
        self.client.settings = replace(
            self.settings,
            websearch_base_url="https://search.invalid/api/v1/",
            websearch_api_key="separate",
        )
        with patch.object(
            self.client.opener, "open", return_value=io.BytesIO(json.dumps(response()).encode())
        ) as opened:
            self.client.search(
                "test",
                blocked_domains=["example.com"],
                _context=ExecutionContext(time.monotonic() + 10),
            )
        request = opened.call_args.args[0]
        self.assertEqual(request.full_url, "https://search.invalid/api/v1/search/websearch")
        self.assertEqual(request.get_header("Authorization"), "Bearer separate")
        self.assertEqual(
            json.loads(request.data), {"query": "test", "blocked_domains": ["example.com"]}
        )

    def test_preserves_sources_and_search_timing_without_inventing_answer(self):
        result = self.execute()
        self.assertEqual(result["status"], "ok")
        data = result["data"]
        self.assertEqual(data["provider"], "cline")
        self.assertEqual(data["query"], "Python latest stable release 2026")
        self.assertEqual(data["sources"], response()["data"]["results"])
        self.assertEqual(data["duration_ms"], 716)
        self.assertIn("searched_at", data)
        self.assertNotIn("answer", data)

    def test_empty_results_are_success_but_missing_results_are_error(self):
        self.assertEqual(
            self.execute({"success": True, "data": {"results": []}})["data"]["sources"], []
        )
        for payload in [[], {}, {"success": True}, {"data": None}, {"data": {"results": "wrong"}}]:
            with self.subTest(payload=payload):
                result = self.execute(payload)
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["error"]["code"], "invalid_response")

    def test_provider_errors_are_not_hidden_by_http_success(self):
        for payload in [
            {"success": False, "data": {"results": []}},
            {"error": "secret"},
            {"data": {"results": [], "error": "secret"}},
        ]:
            with self.subTest(payload=payload):
                result = self.execute(payload)
                self.assertEqual(result["status"], "error")
                self.assertEqual(result["error"]["code"], "provider_error")
                self.assertNotIn("secret", json.dumps(result))

    def test_bad_result_entries_keep_usable_sources_as_partial(self):
        invalid = [
            None,
            {"title": "missing URL"},
            {"url": "javascript:alert(1)"},
            {"url": "https://"},
            {"url": "https://[broken"},
        ]
        for item in invalid:
            with self.subTest(item=item):
                payload = response()
                payload["data"]["results"].append(item)
                result = self.execute(payload)
                self.assertEqual(result["status"], "partial")
                self.assertEqual(len(result["data"]["sources"]), 2)
        self.assertEqual(self.execute({"data": {"results": invalid}})["status"], "error")

    def test_network_and_decode_failures_are_structured(self):
        cases = [
            (
                urllib.error.HTTPError(
                    "https://test.invalid", 401, "bad auth", {}, io.BytesIO(b"secret")
                ),
                "http_401",
                "error",
            ),
            (
                urllib.error.HTTPError(
                    "https://test.invalid", 429, "rate limit", {}, io.BytesIO(b"secret")
                ),
                "http_429",
                "error",
            ),
            (urllib.error.URLError("offline"), "connection_error", "error"),
            (urllib.error.URLError(TimeoutError()), "deadline_exceeded", "timeout"),
            (TimeoutError(), "deadline_exceeded", "timeout"),
        ]
        for error, code, status in cases:
            with (
                self.subTest(code=code),
                patch.object(self.client.opener, "open", side_effect=error),
            ):
                result = json.loads(self.tools.execute(call()))
                self.assertEqual(result["status"], status)
                self.assertEqual(result["error"]["code"], code)
                self.assertNotIn("secret", json.dumps(result))
        with patch.object(self.client.opener, "open", return_value=io.BytesIO(b"not json")):
            self.assertEqual(
                json.loads(self.tools.execute(call()))["error"]["code"], "invalid_response"
            )

    def test_redirect_does_not_forward_api_key(self):
        handler = next(
            item
            for item in self.client.opener.handlers
            if isinstance(item, urllib.request.HTTPRedirectHandler)
        )
        request = urllib.request.Request(
            "https://api.cline.bot/api/v1/search/websearch",
            headers={"Authorization": "Bearer cline-key"},
        )
        self.assertIsNone(
            handler.redirect_request(request, None, 302, "Found", {}, "https://other.invalid")
        )

    def test_validation_and_expired_deadline_prevent_requests(self):
        with patch.object(self.client.opener, "open") as opened:
            for arguments in [
                {"query": " "},
                {"timeout": 0},
                {"max_uses": 3},
                {"allowed_domains": "python.org"},
                {"blocked_domains": [1]},
                {"allowed_domains": ["python.org"], "blocked_domains": ["example.com"]},
            ]:
                with self.subTest(arguments=arguments):
                    self.assertEqual(
                        json.loads(self.tools.execute(call(**arguments)))["error"]["code"],
                        "invalid_arguments",
                    )
            result = json.loads(self.tools.execute(call(), deadline=time.monotonic() - 1))
            self.assertEqual(result["status"], "timeout")
            opened.assert_not_called()

    def test_missing_cline_credentials_prevent_requests(self):
        self.client.settings = replace(self.settings, base_url="https://api.deepseek.com")
        with patch.object(self.client.opener, "open") as opened:
            result = json.loads(self.tools.execute(call()))
        self.assertEqual(result["error"]["code"], "websearch_not_configured")
        opened.assert_not_called()

    def test_request_timeout_and_late_response(self):
        self.client.settings = replace(self.settings, request_timeout_seconds=2)
        with patch.object(
            self.client.opener, "open", return_value=io.BytesIO(json.dumps(response()).encode())
        ) as opened:
            self.client.search("test", timeout=30, _context=ExecutionContext(time.monotonic() + 60))
        self.assertLessEqual(opened.call_args.kwargs["timeout"], 2)
        with (
            patch.object(
                self.client.opener, "open", return_value=io.BytesIO(json.dumps(response()).encode())
            ),
            patch.object(
                ExecutionContext, "remaining", side_effect=[1, 1, 1, TimeoutError("deadline")]
            ),
        ):
            result = json.loads(self.tools.execute(call()))
        self.assertEqual(result["status"], "timeout")

    def test_large_results_remain_readable_from_artifact(self):
        payload = response()
        payload["data"]["results"][0]["title"] = "large title " * 2000
        result = self.execute(payload)
        self.assertTrue(result["truncated"])
        self.assertLessEqual(len(json.dumps(result, ensure_ascii=False)), MAX_RESULT_CHARS)
        saved = json.loads(Path(result["artifact_path"]).read_text())
        self.assertEqual(saved["data"]["sources"], payload["data"]["results"])

    def test_result_reaches_next_model_turn_and_workers_can_search(self):
        client = Mock(spec=["complete"])
        client.complete.side_effect = [
            {"role": "assistant", "content": None, "tool_calls": [call()]},
            {"role": "assistant", "content": "完成。"},
        ]
        conversation = Conversation(self.settings, client, self.tools)
        with patch.object(
            self.client.opener, "open", return_value=io.BytesIO(json.dumps(response()).encode())
        ):
            self.assertEqual(conversation.ask("搜索最新 Python 版本"), "完成。")
        messages = client.complete.call_args.args[0]
        tool_message = next(item for item in messages if item["role"] == "tool")
        self.assertEqual(tool_message["tool_call_id"], "call-search")
        self.assertEqual(
            json.loads(tool_message["content"])["data"]["sources"], response()["data"]["results"]
        )
        worker_tools = self.tools.registry["delegate"].handler.__self__.runner.worker_tools
        self.assertIn("websearch", {item["function"]["name"] for item in worker_tools})
        with patch.object(
            self.client.opener, "open", return_value=io.BytesIO(json.dumps(response()).encode())
        ):
            result = json.loads(
                self.tools.execute_workspace(call(), ExecutionContext(time.monotonic() + 10))
            )
        self.assertEqual(result["status"], "ok")

    def test_main_only_registers_search_when_enabled_and_configured(self):
        import main

        for configured, enabled in [(True, True), (False, True), (True, False)]:
            current = replace(
                self.settings,
                base_url="https://api.cline.bot/api/v1"
                if configured
                else "https://api.deepseek.com",
                websearch_enabled=enabled,
                mcp_enabled=False,
            )
            with (
                self.subTest(configured=configured, enabled=enabled),
                patch("main.load_settings", return_value=current),
                patch("main.Console"),
                patch("main.Conversation") as conversation,
                patch("main.run_interactive"),
                patch("sys.argv", ["main.py"]),
            ):
                main.main()
            self.assertEqual(
                "websearch" in conversation.call_args.args[2].registry, configured and enabled
            )


class WebSearchSettingsTests(unittest.TestCase):
    def test_credentials_are_only_reused_for_same_origin(self):
        base = settings(Path("/tmp"))
        self.assertEqual(base.resolved_websearch_api_key, "cline-key")
        self.assertIsNone(
            replace(base, base_url="https://api.deepseek.com/v1").resolved_websearch_api_key
        )
        self.assertIsNone(
            replace(
                base, websearch_base_url="http://api.cline.bot/api/v1"
            ).resolved_websearch_api_key
        )
        self.assertIsNone(
            replace(
                base, websearch_base_url="https://api.cline.bot:444/api/v1"
            ).resolved_websearch_api_key
        )
        self.assertEqual(
            replace(base, websearch_api_key="separate").resolved_websearch_api_key, "separate"
        )

    def test_env_uses_native_cline_defaults(self):
        with (
            patch("config.load_env"),
            patch.dict(
                "os.environ",
                {
                    "DEEPSEEK_API_KEY": "cline-key",
                    "DEEPSEEK_BASE_URL": "https://api.cline.bot/api/v1",
                },
                clear=True,
            ),
        ):
            configured = Settings.from_env()
        self.assertEqual(configured.websearch_base_url, "https://api.cline.bot/api/v1")
        self.assertEqual(configured.resolved_websearch_api_key, "cline-key")
        self.assertTrue(configured.websearch_enabled)

    def test_env_supports_separate_search_credentials_and_disable(self):
        with (
            patch("config.load_env"),
            patch.dict(
                "os.environ",
                {
                    "DEEPSEEK_API_KEY": "main",
                    "DEEPSEEK_BASE_URL": "https://main.invalid/v1",
                    "DEEPSEEK_WEBSEARCH_API_KEY": "search",
                    "DEEPSEEK_WEBSEARCH_BASE_URL": "https://search.invalid/api/v1/",
                    "DEEPSEEK_WEBSEARCH_ENABLED": "false",
                },
                clear=True,
            ),
        ):
            configured = Settings.from_env()
        self.assertEqual(configured.resolved_websearch_api_key, "search")
        self.assertEqual(configured.websearch_base_url, "https://search.invalid/api/v1")
        self.assertFalse(configured.websearch_enabled)


class FakeStream:
    """An accepted Responses event stream, as ``http_transport.open_url`` yields it."""

    def __init__(self, *events, headers=None):
        self.headers = headers or {}
        self.lines = [
            line
            for event in events
            for line in (
                f"event: {event.get('type', 'raw') if isinstance(event, dict) else 'raw'}\n",
                f"data: {event if isinstance(event, str) else json.dumps(event)}\n",
                "\n",
            )
        ]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __iter__(self):
        return iter(line.encode() for line in self.lines)


CITED = "https://blog.rust-lang.org/2025/09/18/Rust-1.90.0/"


def hosted_search(*, completed=True):
    events = [
        {"type": "response.created", "response": {}},
        {
            "type": "response.output_item.done",
            "item": {"type": "reasoning", "summary": [], "encrypted_content": "opaque"},
        },
        {
            "type": "response.output_item.done",
            "item": {
                "id": "ws_1",
                "type": "web_search_call",
                "status": "completed",
                "action": {
                    "type": "search",
                    "query": "Rust 1.90 release",
                    "queries": ["Rust 1.90 release", "Rust 1.90 release notes"],
                    "sources": [
                        {"type": "url", "url": "https://doc.rust-lang.org/releases.html"},
                        {"type": "url", "url": CITED},
                        {"type": "url", "url": "javascript:alert(1)"},
                    ],
                },
            },
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "web_search_call",
                "action": {"type": "open_page", "url": "https://blog.rust-lang.org/releases/"},
            },
        },
        {
            "type": "response.output_item.done",
            "item": {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "Rust 1.90 was released on September 18, 2025. "
                        f"([blog.rust-lang.org]({CITED}?utm_source=openai))",
                        "annotations": [
                            {
                                "type": "url_citation",
                                "title": "Announcing Rust 1.90.0 | Rust Blog",
                                "url": f"{CITED}?utm_source=openai",
                            }
                        ],
                    }
                ],
            },
        },
    ]
    if completed:
        events.append({"type": "response.completed", "response": {"usage": {}}})
    return events


class ChatGPTWebSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        self.store = auth.AuthStore(self.workspace / "auth.json")
        self.addCleanup(auth.forget_account, self.store.path)
        self.tokens = auth.account_tokens(token_response())
        self.store.update(
            provider="chatgpt",
            chatgpt={
                **self.tokens,
                "model": "gpt-test",
                "models": [{"slug": "gpt-test", "reasoning_levels": ["low", "medium", "high"]}],
            },
        )
        self.settings = settings(
            self.workspace,
            api_key="",
            base_url=CHATGPT_BASE_URL,
            model="gpt-test",
            provider="chatgpt",
            auth_file=self.store.path,
        )
        self.client = ChatGPTWebSearchClient(self.settings)
        self.tools = CodingTools(self.workspace)
        self.tools.attach_websearch(self.client)

    def search(self, *streams, **arguments):
        with patch("websearch.http_transport.open_url", side_effect=list(streams)) as opened:
            result = json.loads(self.tools.execute(call(**arguments)))
        return result, opened

    def test_request_uses_the_account_and_hosted_search_tool(self):
        context = ExecutionContext(time.monotonic() + 120)
        headers = {"x-codex-primary-used-percent": "12", "x-codex-primary-window-minutes": "300"}
        with patch(
            "websearch.http_transport.open_url",
            return_value=FakeStream(*hosted_search(), headers=headers),
        ) as opened:
            result = self.client.search(
                " Rust 1.90 release ", allowed_domains=[" rust-lang.org "], _context=context
            )
        request = opened.call_args.args[0]
        self.assertEqual(request.full_url, f"{CHATGPT_BASE_URL}/responses")
        self.assertEqual(
            request.get_header("Authorization"), f"Bearer {self.tokens['access_token']}"
        )
        self.assertEqual(request.get_header("Chatgpt-account-id"), "account-1")
        body = json.loads(request.data)
        self.assertEqual(body["model"], "gpt-test")
        self.assertEqual(
            body["tools"],
            [{"type": "web_search", "filters": {"allowed_domains": ["rust-lang.org"]}}],
        )
        self.assertEqual(body["tool_choice"], "required")
        self.assertEqual(body["include"], ["web_search_call.action.sources"])
        self.assertEqual(body["reasoning"], {"effort": "low"})
        self.assertFalse(body["store"])
        self.assertEqual(body["input"][0]["content"][0]["text"], "Rust 1.90 release")
        self.assertIn("Current date:", body["instructions"])
        # Bounded by the tool call and cancellable with it.
        self.assertIs(opened.call_args.kwargs["cancelled"], context.cancelled)
        self.assertLessEqual(opened.call_args.kwargs["deadline"], context.deadline)
        self.assertLessEqual(opened.call_args.kwargs["timeout"], 90)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.data["allowed_domains"], ["rust-lang.org"])
        self.assertEqual(self.client.account.quota["primary"]["used-percent"], 12)

    def test_result_has_digest_searches_and_cited_sources_first(self):
        result, _ = self.search(FakeStream(*hosted_search()))
        self.assertEqual(result["status"], "ok")
        data = result["data"]
        self.assertEqual(data["provider"], "chatgpt")
        self.assertEqual(data["query"], "Python latest stable release 2026")
        self.assertEqual(data["searches"], ["Rust 1.90 release", "Rust 1.90 release notes"])
        self.assertEqual(
            data["summary"],
            f"Rust 1.90 was released on September 18, 2025. ([blog.rust-lang.org]({CITED}))",
        )
        self.assertEqual(
            data["sources"],
            [
                {"url": CITED, "title": "Announcing Rust 1.90.0 | Rust Blog", "cited": True},
                {"url": "https://doc.rust-lang.org/releases.html"},
                {"url": "https://blog.rust-lang.org/releases/"},
            ],
        )
        self.assertIn("searched_at", data)

    def test_blocked_domains_and_effort_follow_the_model(self):
        with patch.object(
            self.client.account, "reasoning_levels", return_value=["minimal", "medium", "high"]
        ):
            _, opened = self.search(FakeStream(*hosted_search()), blocked_domains=["example.com"])
        body = json.loads(opened.call_args.args[0].data)
        self.assertEqual(
            body["tools"], [{"type": "web_search", "filters": {"blocked_domains": ["example.com"]}}]
        )
        self.assertEqual(body["reasoning"], {"effort": "medium"})

    def test_rejected_token_is_refreshed_once(self):
        rejected = urllib.error.HTTPError(
            CHATGPT_BASE_URL, 401, "Unauthorized", {}, io.BytesIO(b"{}")
        )
        fresh = auth.account_tokens(token_response(refresh="refresh-2"))
        with patch("auth.refresh_tokens", return_value=fresh) as refresh:
            result, opened = self.search(rejected, FakeStream(*hosted_search()))
        refresh.assert_called_once()
        self.assertEqual(result["status"], "ok")
        self.assertEqual(
            opened.call_args.args[0].get_header("Authorization"), f"Bearer {fresh['access_token']}"
        )

    def test_backend_failures_are_structured(self):
        limit = json.dumps(
            {
                "error": {
                    "type": "usage_limit_reached",
                    "plan_type": "plus",
                    "resets_in_seconds": 600,
                }
            }
        ).encode()
        cases = [
            (
                urllib.error.HTTPError(CHATGPT_BASE_URL, 429, "Limit", {}, io.BytesIO(limit)),
                "usage_limit",
                "error",
            ),
            (
                urllib.error.HTTPError(
                    CHATGPT_BASE_URL, 400, "Bad", {}, io.BytesIO(b'{"detail": "bad filter"}')
                ),
                "http_400",
                "error",
            ),
            (
                FakeStream(
                    {
                        "type": "response.failed",
                        "response": {"error": {"code": "server_error", "message": "boom"}},
                    }
                ),
                "provider_error",
                "error",
            ),
            (FakeStream("not json"), "invalid_response", "error"),
            (urllib.error.URLError("offline"), "connection_error", "error"),
            (http_transport.TransportDeadline("late"), "deadline_exceeded", "timeout"),
            (http_transport.TransportCancelled("stop"), "deadline_exceeded", "timeout"),
        ]
        for error, code, status in cases:
            with self.subTest(code=code):
                result, _ = self.search(error)
                self.assertEqual(result["status"], status)
                self.assertEqual(result["error"]["code"], code)
                if code == "usage_limit":
                    self.assertIn("用量已达上限", result["error"]["message"])

    def test_unfinished_or_unsearched_answers_are_not_presented_as_results(self):
        result, _ = self.search(FakeStream(*hosted_search(completed=False)))
        self.assertEqual(
            (result["status"], result["error"]["code"]), ("partial", "invalid_response")
        )
        self.assertTrue(result["data"]["summary"])
        incomplete = [
            *hosted_search(completed=False),
            {
                "type": "response.incomplete",
                "response": {"incomplete_details": {"reason": "max_output_tokens"}},
            },
        ]
        result, _ = self.search(FakeStream(*incomplete))
        self.assertEqual(
            (result["status"], result["error"]["code"]), ("partial", "incomplete_response")
        )
        from_memory = [hosted_search()[4], {"type": "response.completed", "response": {}}]
        result, _ = self.search(FakeStream(*from_memory))
        self.assertEqual(
            (result["status"], result["error"]["code"]), ("error", "no_search_performed")
        )

    def test_missing_login_and_invalid_arguments_prevent_requests(self):
        self.store.update(chatgpt=None)
        auth.forget_account(self.store.path)
        with patch("websearch.http_transport.open_url") as opened:
            result = json.loads(self.tools.execute(call()))
            self.assertEqual(result["error"]["code"], "auth_error")
            self.assertIn("/login", result["error"]["message"])
            for arguments in [
                {"query": " "},
                {"allowed_domains": ["python.org"], "blocked_domains": ["example.com"]},
            ]:
                self.assertEqual(
                    json.loads(self.tools.execute(call(**arguments)))["error"]["code"],
                    "invalid_arguments",
                )
        opened.assert_not_called()

    def test_tool_describes_the_account_backend(self):
        definition = next(
            item["function"]
            for item in self.tools.definitions
            if item["function"]["name"] == "websearch"
        )
        self.assertIn("ChatGPT account", definition["description"])
        self.assertNotIn("Cline", definition["description"])
        self.assertEqual(definition["parameters"]["properties"]["timeout"]["default"], 90)

    def test_switching_login_moves_search_to_the_new_backend(self):
        api = replace(
            self.settings, provider="api", api_key="k", base_url="https://api.deepseek.com"
        )
        tools = CodingTools(self.workspace)
        client = FakeClient()
        client.settings = api
        conversation = Conversation(api, client, tools)
        self.assertNotIn("websearch", conversation.tools.registry)
        with patch.dict("os.environ", {"DEEPSEEK_AUTH_FILE": str(self.store.path)}):
            conversation.switch_login(Login("chatgpt", "gpt-test", account="a"), FakeClient())
        search = conversation.tools.registry["websearch"].handler.__self__
        self.assertIsInstance(search, ChatGPTWebSearchClient)
        self.assertIn("websearch", conversation._baseline_registry)
        worker_tools = conversation.tools.registry["delegate"].handler.__self__.runner.worker_tools
        self.assertIn("websearch", {item["function"]["name"] for item in worker_tools})
        # An environment that turned search off keeps it off; the baseline still follows.
        conversation.settings = replace(conversation.settings, websearch_enabled=False)
        with patch.dict("os.environ", {"DEEPSEEK_AUTH_FILE": str(self.store.path)}):
            conversation.switch_login(Login("chatgpt", "gpt-test", account="a"), FakeClient())
        self.assertNotIn("websearch", conversation.tools.registry)
        self.assertIn("websearch", conversation._baseline_registry)
        conversation.switch_login(Login("api", "m", "https://api.deepseek.com", "k"), FakeClient())
        self.assertNotIn("websearch", conversation._baseline_registry)

    def test_main_registers_account_search_without_a_cline_key(self):
        import main

        with (
            patch("main.load_settings", return_value=replace(self.settings, mcp_enabled=False)),
            patch("main.Console") as console,
            patch("main.Conversation") as conversation,
            patch("main.run_interactive"),
            patch("sys.argv", ["main.py"]),
        ):
            main.main()
        search = conversation.call_args.args[2].registry["websearch"].handler.__self__
        self.assertIsInstance(search, ChatGPTWebSearchClient)
        self.assertEqual(console.call_args.kwargs["websearch_status"], "可用")


class WebSearchBackendTests(unittest.TestCase):
    def test_backend_follows_credentials_and_login(self):
        base = settings(Path("/tmp"), base_url="https://api.deepseek.com", api_key="k")
        account = replace(base, provider="chatgpt", base_url=CHATGPT_BASE_URL, api_key="")
        cases = [
            (base, None, None),
            (replace(base, websearch_api_key="cline"), "cline", WebSearchClient),
            (account, "chatgpt", ChatGPTWebSearchClient),
            # A Cline key configured on purpose keeps Cline, even on a ChatGPT account.
            (replace(account, websearch_api_key="cline"), "cline", WebSearchClient),
        ]
        for current, provider, kind in cases:
            with self.subTest(provider=provider, account=current.provider):
                self.assertEqual(current.websearch_provider, provider)
                search = websearch_client(current)
                self.assertIs(type(search) if search is not None else None, kind)
        self.assertIsNone(websearch_client(replace(account, websearch_enabled=False)))


if __name__ == "__main__":
    unittest.main()
