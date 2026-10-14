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

from config import Settings
from conversation import Conversation
from tests.support import make_settings
from tool_runtime import MAX_RESULT_CHARS, ExecutionContext
from tools import CodingTools
from websearch import WebSearchClient


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
                patch("main.Settings.from_env", return_value=current),
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


if __name__ == "__main__":
    unittest.main()
