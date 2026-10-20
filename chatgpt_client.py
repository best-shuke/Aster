"""Model client for a ChatGPT (Plus / Pro) account, over the Codex Responses backend.

Conversations stay in the chat-completions shape the rest of Aster uses. Requests are
converted to Responses input items here, and stream events are translated back into
chat-completions chunks, so the shared accumulator keeps validating tool calls,
reasoning limits and previews exactly as for any other provider.
"""

import hashlib
import json
import threading
import time
import urllib.error
import urllib.request
import uuid
from contextlib import suppress
from typing import Any

from auth import USER_AGENT, AuthError, LoginExpired, chatgpt_account
from config import auth_file, parse_reasoning_effort
from context_budget import LOCAL_METADATA, wire_messages
from contracts import Message
from deepseek_client import CompletionError, DeepSeekClient
from model_rate_limit import retry_after_seconds
from usage import coarse_duration, number

REASONING_KEY = "responses_reasoning"
DEFAULT_INSTRUCTIONS = "You are a helpful assistant."
# Aster's effort names, in order of preference among the levels a model supports.
EFFORT_LEVELS = {
    "off": ("none", "minimal", "low"),
    "low": ("low", "minimal"),
    "high": ("high", "medium"),
    "max": ("max", "xhigh", "high"),
}


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text") or "" if isinstance(part, dict) else str(part) for part in content
        )
    return "" if content is None else json.dumps(content, ensure_ascii=False)


def _parts(content: Any, kind: str) -> list[dict[str, Any]]:
    if not isinstance(content, list):
        return [{"type": kind, "text": _text(content)}]
    parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "image_url":
            image = part.get("image_url")
            url = image.get("url") if isinstance(image, dict) else image
            parts.append({"type": "input_image", "image_url": url})
        else:
            parts.append({"type": kind, "text": _text([part])})
    return parts


def responses_input(messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
    """Leading system messages become instructions; the backend rejects system items."""
    instructions: list[str] = []
    items: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            text = _text(message.get("content"))
            if not items:
                instructions.append(text)
            else:
                items.append(
                    {"type": "message", "role": "developer", "content": _parts(text, "input_text")}
                )
        elif role == "user":
            items.append(
                {
                    "type": "message",
                    "role": "user",
                    "content": _parts(message.get("content"), "input_text"),
                }
            )
        elif role == "assistant":
            text = _text(message.get("content"))
            metadata = message.get(LOCAL_METADATA)
            # Reasoning is replayed only ahead of the output it led to.
            if isinstance(metadata, dict) and (text or message.get("tool_calls")):
                items.extend(
                    item for item in metadata.get(REASONING_KEY) or [] if isinstance(item, dict)
                )
            if text:
                items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": text}],
                    }
                )
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call.get("id") or "",
                        "name": function.get("name") or "",
                        "arguments": function.get("arguments") or "",
                    }
                )
        elif role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id") or "",
                    "output": _text(message.get("content")),
                }
            )
    return "\n\n".join(part for part in instructions if part) or DEFAULT_INSTRUCTIONS, items


def responses_tool(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function") or {}
    return {
        "type": "function",
        "name": function.get("name"),
        "description": function.get("description") or "",
        "parameters": function.get("parameters") or {"type": "object", "properties": {}},
        # Responses tools default to strict schemas, which Aster's schemas do not follow.
        "strict": False,
    }


def chat_usage(usage: Any) -> dict[str, Any] | None:
    if not isinstance(usage, dict):
        return None
    result: dict[str, Any] = {
        key: usage[source]
        for key, source in (
            ("prompt_tokens", "input_tokens"),
            ("completion_tokens", "output_tokens"),
            ("total_tokens", "total_tokens"),
        )
        if number(usage.get(source))
    }
    cached = (usage.get("input_tokens_details") or {}).get("cached_tokens")
    if number(cached):
        result["prompt_tokens_details"] = {"cached_tokens": cached}
    reasoning = (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
    if number(reasoning):
        result["completion_tokens_details"] = {"reasoning_tokens": reasoning}
    return result or None


def _chunk(finish_reason: str | None = None, usage: Any = None, **delta: Any) -> dict[str, Any]:
    chunk: dict[str, Any] = {"choices": [{"delta": delta, "finish_reason": finish_reason}]}
    if usage:
        chunk["usage"] = usage
    return chunk


def response_error(error: Any) -> CompletionError:
    error = error if isinstance(error, dict) else {"message": str(error)}
    code = str(error.get("code") or error.get("type") or "")
    message = str(error.get("message") or code or "Response failed")[:300]
    if code == "context_length_exceeded":
        return CompletionError("context_overflow", message)
    if code in {"usage_limit_reached", "insufficient_quota"}:
        return CompletionError("usage_limit", f"ChatGPT 用量已达上限：{message}")
    if code in {"rate_limit_exceeded", "rate_limit", "429"}:
        return CompletionError(
            "http_429", "Model service returned a rate-limit error", retryable=True
        )
    return CompletionError("provider_error", message, retryable=True)


class _ResponsesEvents:
    """Translate one Responses event stream into chat-completions chunks."""

    def __init__(self) -> None:
        self.tools: dict[Any, int] = {}
        self.arguments: dict[int, str] = {}
        self.reasoning: list[dict[str, Any]] = []
        self.reasoning_part: Any = None
        self.text_item: Any = None
        self.done = False

    def _tool(self, event: dict[str, Any], item: dict[str, Any] | None = None) -> int:
        key = (item or {}).get("id") or event.get("item_id") or event.get("output_index")
        if key not in self.tools:
            self.tools[key] = len(self.tools)
        return self.tools[key]

    def _arguments(self, index: int, text: str) -> dict[str, Any]:
        self.arguments[index] = self.arguments.get(index, "") + text
        return _chunk(tool_calls=[{"index": index, "function": {"arguments": text}}])

    def chunks(self, event: Any) -> list[dict[str, Any]]:
        if not isinstance(event, dict):
            raise CompletionError(
                "invalid_response", "Stream event is not an object", retryable=True
            )
        kind = event.get("type")
        delta = event.get("delta")
        if kind == "response.output_text.delta":
            item = event.get("item_id")
            separator = "\n\n" if self.text_item not in (None, item) else ""
            self.text_item = item
            return [_chunk(content=separator + delta)] if isinstance(delta, str) else []
        if kind in {"response.reasoning_summary_text.delta", "response.reasoning_text.delta"}:
            part = (event.get("item_id"), event.get("summary_index", event.get("content_index")))
            separator = "\n\n" if self.reasoning_part not in (None, part) else ""
            self.reasoning_part = part
            return [_chunk(reasoning_content=separator + delta)] if isinstance(delta, str) else []
        if kind == "response.function_call_arguments.delta" and isinstance(delta, str):
            return [self._arguments(self._tool(event), delta)]
        item = event.get("item")
        if kind == "response.output_item.added" and isinstance(item, dict):
            if item.get("type") != "function_call":
                return []
            index = self._tool(event, item)
            self.arguments[index] = item.get("arguments") or ""
            call = {
                "index": index,
                "id": item.get("call_id") or "",
                "function": {"name": item.get("name") or "", "arguments": self.arguments[index]},
            }
            return [_chunk(tool_calls=[call])]
        if kind == "response.output_item.done" and isinstance(item, dict):
            if item.get("type") == "reasoning" and item.get("encrypted_content"):
                # Replayed on later requests so the model keeps its own reasoning.
                self.reasoning.append(
                    {
                        "type": "reasoning",
                        "summary": item.get("summary") or [],
                        "encrypted_content": item["encrypted_content"],
                    }
                )
            if item.get("type") != "function_call":
                return []
            known = (item.get("id") or event.get("item_id") or event.get("output_index")) in (
                self.tools
            )
            if not known:
                return self.chunks({**event, "type": "response.output_item.added"})
            index = self._tool(event, item)
            final, received = item.get("arguments") or "", self.arguments.get(index, "")
            if final != received and final.startswith(received):
                return [self._arguments(index, final[len(received) :])]
            return []
        response = event.get("response") if isinstance(event.get("response"), dict) else {}
        if kind == "response.completed":
            self.done = True
            finish = "tool_calls" if self.tools else "stop"
            return [_chunk(finish, chat_usage(response.get("usage")))]
        if kind == "response.incomplete":
            self.done = True
            reason = (response.get("incomplete_details") or {}).get("reason") or "incomplete"
            finish = "length" if reason == "max_output_tokens" else reason
            return [_chunk(finish, chat_usage(response.get("usage")))]
        if kind == "response.failed":
            raise response_error(response.get("error") or {})
        if kind == "error":
            raise response_error(event.get("error") or event)
        return []


class ChatGPTClient(DeepSeekClient):
    def __init__(self, settings) -> None:
        super().__init__(settings)
        self.account = chatgpt_account(settings.auth_file or auth_file())
        self._local = threading.local()

    @classmethod
    def _consume_stream(
        cls, response, state, deadline, generation_deadline, on_usage, metrics=None
    ) -> None:
        events = _ResponsesEvents()
        for raw_line in response:
            cls._check_stream_deadline(deadline, generation_deadline, state.finish_reason)
            line = raw_line.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                state.saw_done = True
                break
            for payload in events.chunks(json.loads(data)):
                if on_usage:
                    on_usage(payload)
                if metrics:
                    metrics.observe(payload)
                state.add_payload(payload)
            if events.reasoning:
                state.metadata[REASONING_KEY] = list(events.reasoning)
            if events.done:
                state.saw_done = True
                break

    def _read_nonstream_response(self, response, observed) -> Message:
        # The backend only streams; callers without a content callback get the whole answer.
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
        instructions, items = responses_input(messages)
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
            token, account_id = self.account.credentials()
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

    def _complete_once(self, messages, *args, **kwargs) -> Message:
        self._local.token = None
        try:
            return super()._complete_once(messages, *args, **kwargs)
        except CompletionError as exc:
            rejected = self._local.token
            if exc.code != "http_401" or rejected is None:
                raise
        try:
            self.account.refresh(rejected)
        except LoginExpired as exc:
            raise CompletionError("auth_error", str(exc)) from exc
        except AuthError as exc:
            raise CompletionError("auth_error", str(exc), retryable=True) from exc
        return super()._complete_once(messages, *args, **kwargs)

    def _http_error(self, exc: urllib.error.HTTPError) -> CompletionError:
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
        detail = (payload.get("detail") if isinstance(payload, dict) else None) or error.get(
            "message"
        )
        message = f"ChatGPT 返回 HTTP {exc.code}"
        if isinstance(detail, str) and detail.strip():
            message += f"：{detail.strip()[:300]}"
        return CompletionError(
            f"http_{exc.code}",
            message,
            retryable=exc.code in {408, 429} or exc.code >= 500,
            retry_after=retry_after,
        )

    def _observe_headers(self, headers: dict[str, str]) -> None:
        values = {key.lower(): value for key, value in headers.items()}

        def window(name: str) -> dict[str, float] | None:
            fields = {}
            for field in ("used-percent", "window-minutes", "reset-after-seconds"):
                with suppress(KeyError, TypeError, ValueError):
                    fields[field] = float(values[f"x-codex-{name}-{field}"])
            return fields if "used-percent" in fields else None

        quota = {
            name: data for name in ("primary", "secondary") if (data := window(name)) is not None
        }
        if quota:
            quota["observed_at"] = time.time()
            self.account.quota = quota
