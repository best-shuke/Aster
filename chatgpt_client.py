"""ChatGPT account authentication and Codex-specific Responses request policy."""

import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
import uuid
from contextlib import suppress
from typing import Any

from auth import HTTP_TIMEOUT_SECONDS, USER_AGENT, AuthError, LoginExpired, chatgpt_account
from config import auth_file, parse_reasoning_effort
from context_budget import wire_messages
from contracts import Message
from deepseek_client import CompletionError
from http_transport import TransportDeadline
from model_api import protocol_origin
from model_messages import protocol_messages
from model_rate_limit import retry_after_seconds
from responses_client import ResponsesClient
from responses_protocol import responses_input as responses_input
from responses_protocol import responses_tool as responses_tool
from responses_protocol import response_error as response_error
from usage import coarse_duration, number

# Aster's effort names, in order of preference among the levels a model supports.
EFFORT_LEVELS = {
    "off": ("none", "minimal", "low"),
    "low": ("low", "minimal"),
    "high": ("high", "medium"),
    "max": ("max", "xhigh", "high"),
}


def http_error(exc: urllib.error.HTTPError) -> CompletionError:
    retry_after = retry_after_seconds(exc.headers.get("Retry-After")) if exc.headers else None
    payload: Any = {}
    with suppress(OSError, ValueError):
        payload = json.loads(exc.read(16384) or b"{}")
    exc.close()
    error = payload.get("error") if isinstance(payload, dict) else None
    error = error if isinstance(error, dict) else {}
    if exc.code == 401:
        return CompletionError("http_401", "ChatGPT 登录凭据被拒绝")
    if exc.code == 429 and error.get("type") == "usage_limit_reached":
        seconds = error.get("resets_in_seconds")
        if not number(seconds) and number(error.get("resets_at")):
            seconds = max(0, error["resets_at"] - time.time())
        when = f"，约 {coarse_duration(seconds)} 后重置" if number(seconds) else ""
        plan = str(error.get("plan_type") or "").title()
        return CompletionError(
            "usage_limit",
            f"ChatGPT {plan} 用量已达上限{when}；可以等待重置，或用 /login 改用第三方 API",
        )
    # The Codex backend reports request problems (such as an unavailable model) as a
    # short detail; it does not echo credentials, so show it.
    detail = (payload.get("detail") if isinstance(payload, dict) else None) or error.get("message")
    message = f"ChatGPT 返回 HTTP {exc.code}"
    if isinstance(detail, str) and detail.strip():
        message += f"：{detail.strip()[:300]}"
    return CompletionError(
        f"http_{exc.code}",
        message,
        retryable=exc.code in {408, 429} or exc.code >= 500,
        retry_after=retry_after,
    )


def quota_windows(headers: dict[str, str]) -> dict[str, Any] | None:
    """The plan usage windows the backend reports in response headers, if any."""
    values = {key.lower(): value for key, value in headers.items()}

    def window(name: str) -> dict[str, float] | None:
        fields = {}
        for field in ("used-percent", "window-minutes", "reset-after-seconds"):
            with suppress(KeyError, TypeError, ValueError):
                fields[field] = float(values[f"x-codex-{name}-{field}"])
        return fields if "used-percent" in fields else None

    quota: dict[str, Any] = {
        name: data for name in ("primary", "secondary") if (data := window(name)) is not None
    }
    if not quota:
        return None
    quota["observed_at"] = time.time()
    return quota


class ChatGPTClient(ResponsesClient):
    def _minimum_budget_output(self) -> int:
        return 16

    def __init__(self, settings) -> None:
        super().__init__(settings)
        self.account = chatgpt_account(settings.auth_file or auth_file())
        self._local = threading.local()

    def _read_nonstream_response(self, response, observed) -> Message:
        # The account backend always streams, including summary requests.
        return self._read_stream(response, lambda _: None, on_usage=observed.add)

    def _effort(self) -> str:
        effort = parse_reasoning_effort(getattr(self.settings, "reasoning_effort", None))
        levels = self.account.reasoning_levels(self.settings.model)
        wanted = EFFORT_LEVELS[effort]
        if not levels:
            return "low" if effort in {"off", "low"} else "high"
        for level in wanted:
            if level in levels:
                return level
        return levels[0] if effort in {"off", "low"} else levels[-1]

    def _request_body(self, messages, tools, streaming, max_output_tokens) -> dict[str, Any]:
        instructions, items = responses_input(
            protocol_messages(messages, self.settings), origin=protocol_origin(self.settings)
        )
        head = wire_messages(messages[:2])
        body: dict[str, Any] = {
            "model": self.settings.model,
            "instructions": instructions,
            "input": items,
            "store": False,
            "stream": True,
            "include": ["reasoning.encrypted_content"],
            # A stable per-conversation key, also sent as the session_id header, routes
            # requests to the backend's prompt cache; without both, nothing is cached.
            "prompt_cache_key": str(
                uuid.UUID(
                    hashlib.sha256(
                        json.dumps(head, ensure_ascii=False, sort_keys=True).encode()
                    ).hexdigest()[:32]
                )
            ),
        }
        if tools is not None:
            body.update(
                tools=[responses_tool(tool) for tool in tools],
                tool_choice="auto",
                parallel_tool_calls=True,
            )
        self._output_tokens(messages, tools, max_output_tokens)
        if max_output_tokens is not None:
            body["max_output_tokens"] = max_output_tokens
        effort = self._effort()
        body["reasoning"] = (
            {"effort": effort}
            if effort == "none"
            else {
                "effort": effort,
                "summary": "auto",
            }
        )
        return body

    @staticmethod
    def _system_fingerprint(body: dict[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(
                {key: body.get(key) for key in ("model", "instructions", "tools", "reasoning")},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    def _http_request(self, body: dict[str, Any]) -> urllib.request.Request:
        try:
            auth_options = getattr(self._local, "auth_options", {})
            token, account_id = self.account.credentials(**auth_options)
        except LoginExpired as exc:
            raise CompletionError("auth_error", str(exc)) from exc
        except AuthError as exc:
            raise CompletionError("auth_error", str(exc), retryable=True) from exc
        self._local.token = token
        return urllib.request.Request(
            f"{self.settings.base_url}/responses",
            data=json.dumps(body, ensure_ascii=False).encode(),
            headers={
                "Authorization": f"Bearer {token}",
                "ChatGPT-Account-Id": account_id,
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "User-Agent": USER_AGENT,
                "session_id": body.get("prompt_cache_key") or "",
            },
            method="POST",
        )

    def _complete_once(self, messages, tools=None, on_content=None, timeout=180, **kwargs) -> Message:
        generation_timeout = kwargs.get("generation_timeout")
        if generation_timeout is None:
            generation_timeout = getattr(self.settings, "generation_timeout_seconds", None)
        now = time.monotonic()
        generation_deadline = now + generation_timeout if generation_timeout else None
        task_deadline = min(
            (value for value in (kwargs.get("deadline"), self.deadline) if value is not None),
            default=None,
        )
        cancelled = kwargs.get("cancelled") or threading.Event()
        auth_deadline = None

        def check():
            remaining = self._attempt_budget(cancelled, task_deadline, generation_deadline)
            if auth_deadline is not None and time.monotonic() >= auth_deadline:
                raise CompletionError("auth_error", "ChatGPT 登录验证超时", retryable=True)
            return remaining

        def configure_auth():
            nonlocal auth_deadline
            auth_deadline = min(
                value
                for value in (
                    time.monotonic() + min(timeout, HTTP_TIMEOUT_SECONDS),
                    task_deadline,
                    generation_deadline,
                )
                if value is not None
            )
            self._local.auth_options = {"check": check, "deadline": auth_deadline}

        self._local.token = None
        try:
            try:
                configure_auth()
                kwargs["generation_timeout"] = check()
                return super()._complete_once(messages, tools, on_content, timeout, **kwargs)
            except CompletionError as exc:
                rejected = self._local.token
                if exc.code != "http_401" or rejected is None:
                    raise
            try:
                configure_auth()
                check()
                self.account.refresh(rejected, **self._local.auth_options)
            except LoginExpired as exc:
                raise CompletionError("auth_error", str(exc)) from exc
            except AuthError as exc:
                raise CompletionError("auth_error", str(exc), retryable=True) from exc
            kwargs["generation_timeout"] = check()
            return super()._complete_once(messages, tools, on_content, timeout, **kwargs)
        except TransportDeadline as exc:
            check()
            raise CompletionError("auth_error", "ChatGPT 登录验证超时", retryable=True) from exc
        finally:
            self._local.auth_options = {}

    def _http_error(self, exc: urllib.error.HTTPError) -> CompletionError:
        return http_error(exc)

    def _observe_headers(self, headers: dict[str, str]) -> None:
        quota = quota_windows(headers)
        if quota:
            self.account.quota = quota
