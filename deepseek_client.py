import hashlib
import http.client
import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

import http_transport
from config import Settings, parse_reasoning_effort
from context_budget import (
    LOCAL_METADATA,
    context_window,
    estimate_context,
    fingerprint,
    request_fingerprint,
    wire_messages,
)
from contracts import (
    ContentCallback as ContentCallback,
)
from contracts import (
    EventRecorder,
    ToolCall,
)
from contracts import (
    Message as Message,
)
from contracts import (
    ProgressCallback as ProgressCallback,
)
from generation_metrics import GenerationMetrics
from model_rate_limit import retry_after_seconds, shared_gate
from model_request import CompletionError as CompletionError
from model_request import ModelRequestRunner
from model_request import has_substantial_progress as has_substantial_progress
from tool_runtime import validate_tool_calls
from usage import CacheStats, UsageAccumulator

# What an OpenAI-compatible service usually means by a status it rejects a request with.
HTTP_ERROR_HINTS = {
    400: "请求被拒绝，常见原因是模型名称、参数或上下文长度不受支持；可用 /login 检查模型",
    401: "API Key 无效或已过期；用 /login 重新填写，或检查 DEEPSEEK_API_KEY",
    402: "账户余额或额度不足",
    403: "API Key 没有访问该模型或服务的权限",
    404: "服务地址或模型不存在；用 /login 检查服务地址与模型名称",
    408: "请求超时",
    413: "请求内容过大",
    422: "请求参数不被支持；可用 /login 检查模型",
    429: "请求过于频繁或额度已用尽",
}


def has_unexecuted_tool_text(text: str) -> bool:
    return bool(re.search(r"<(?!/)[^<>\s]*DSML[^<>\s]*\s+(?:calls|invoke)\b", text))


def clean_completion_suffix(text: str) -> str:
    """Remove orphan closing protocol tags only at the end of ordinary text."""
    if re.search(r"<(?!/)[^<>\s]*DSML[^<>\s]*\s+", text):
        return text
    closing = r"</[^<>\s]*DSML[^<>\s]*\s+(?:parameter|invoke|calls)\s*>"
    return re.sub(rf"{closing}(?:\s*{closing})*\s*$", "", text)


class CompletionInterrupted(KeyboardInterrupt):
    def __init__(self, partial_message: Message | None = None) -> None:
        super().__init__("Generation interrupted by user")
        self.partial_message = partial_message


def provider_error(error, *, partial_message=None):
    code = error.get("code") if isinstance(error, dict) else None
    if code in (429, "429", "rate_limit", "rate_limit_exceeded"):
        return CompletionError(
            "http_429",
            "Model service returned a rate-limit error",
            retryable=True,
            partial_message=partial_message,
        )
    return CompletionError(
        "provider_error", str(error), retryable=True, partial_message=partial_message
    )


class _StreamAccumulator:
    """Build one assistant message while validating incremental provider data."""

    def __init__(
        self,
        on_content: ContentCallback,
        max_reasoning_chars: int | None,
        on_delta: ProgressCallback | None = None,
    ) -> None:
        self.on_content = on_content
        self.on_delta = on_delta
        self.max_reasoning_chars = max_reasoning_chars
        self.content: list[str] = []
        self.tool_calls: dict[int, ToolCall] = {}
        self.reasoning: dict[str, list[str]] = {}
        self.finish_reason: str | None = None
        self.saw_done = False
        self.reasoning_chars = 0
        self.paragraph_tail = ""
        self.paragraphs: dict[str, int] = {}
        # Provider state kept with the message but never sent back as wire content.
        self.metadata: dict[str, Any] = {}

    def snapshot(self) -> Message:
        message: Message = {"role": "assistant", "content": "".join(self.content) or None}
        message.update({key: "".join(parts) for key, parts in self.reasoning.items()})
        if self.tool_calls:
            message["tool_calls"] = [self.tool_calls[index] for index in sorted(self.tool_calls)]
        if self.metadata:
            message[LOCAL_METADATA] = dict(self.metadata)
        return message

    def _append_reasoning(self, delta: dict[str, Any]) -> bool:
        for key in ("reasoning_content", "reasoning"):
            if isinstance(delta.get(key), str):
                self.reasoning.setdefault(key, []).append(delta[key])
        fragment = delta.get("reasoning_content") or delta.get("reasoning") or ""
        if not isinstance(fragment, str):
            raise CompletionError(
                "invalid_response", "Stream reasoning fragment is not text", retryable=True
            )
        self.reasoning_chars += len(fragment)
        self.paragraph_tail += fragment
        repeated = False
        while "\n\n" in self.paragraph_tail:
            paragraph, self.paragraph_tail = self.paragraph_tail.split("\n\n", 1)
            paragraph = " ".join(paragraph.split())
            if len(paragraph) >= 200:
                self.paragraphs[paragraph] = self.paragraphs.get(paragraph, 0) + 1
                repeated |= self.paragraphs[paragraph] >= 3
        self.paragraph_tail = self.paragraph_tail[-8000:]
        return repeated

    def _append_tool_calls(self, parts: Any) -> None:
        if not isinstance(parts, list):
            raise CompletionError(
                "invalid_response", "Stream tool_calls is not a list", retryable=True
            )
        for part in parts:
            if not isinstance(part, dict) or not isinstance(part.get("index"), int):
                raise CompletionError(
                    "invalid_response", "Stream tool call has no valid index", retryable=True
                )
            function = part.get("function") or {}
            if not isinstance(function, dict):
                raise CompletionError(
                    "invalid_response", "Stream tool function is not an object", retryable=True
                )
            fragments = (
                part.get("id", ""),
                function.get("name", ""),
                function.get("arguments", ""),
            )
            if any(not isinstance(fragment, str) for fragment in fragments):
                raise CompletionError(
                    "invalid_response", "Stream tool call fragments must be text", retryable=True
                )
            call = self.tool_calls.setdefault(
                part["index"],
                {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
            )
            call["id"] += fragments[0]
            call["function"]["name"] += fragments[1]
            call["function"]["arguments"] += fragments[2]

    @staticmethod
    def _first_choice(payload: Any) -> dict[str, Any] | None:
        if not isinstance(payload, dict):
            raise CompletionError(
                "invalid_response", "Stream event is not an object", retryable=True
            )
        if payload.get("error"):
            raise provider_error(payload["error"])
        choices = payload.get("choices", [])
        if not isinstance(choices, list):
            raise CompletionError(
                "invalid_response", "Stream choices is not a list", retryable=True
            )
        if not choices:
            return None
        choice = choices[0]
        if not isinstance(choice, dict):
            raise CompletionError(
                "invalid_response", "Stream choice is not an object", retryable=True
            )
        return choice

    @staticmethod
    def _delta(choice: dict[str, Any]) -> dict[str, Any]:
        delta = choice.get("delta") or {}
        if not isinstance(delta, dict):
            raise CompletionError(
                "invalid_response", "Stream delta is not an object", retryable=True
            )
        return delta

    def _append_content(self, delta: dict[str, Any]) -> None:
        text = delta.get("content")
        if text is not None and not isinstance(text, str):
            raise CompletionError(
                "invalid_response", "Stream content fragment is not text", retryable=True
            )
        if text:
            self.content.append(text)
            self.on_content(text)

    def _check_reasoning_limit(self, repeated: bool) -> None:
        if self.finish_reason is not None or self.content or self.tool_calls:
            return
        if repeated:
            raise CompletionError(
                "reasoning_stalled",
                "Repeated long reasoning paragraphs; checkpoint and change approach",
            )
        if (
            self.max_reasoning_chars is not None
            and self.reasoning_chars >= self.max_reasoning_chars
        ):
            raise CompletionError(
                "generation_budget", "Reasoning slice ended; checkpoint before continuing"
            )

    def add_payload(self, payload: Any) -> None:
        choice = self._first_choice(payload)
        if choice is None:
            return
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None and not isinstance(finish_reason, str):
            raise CompletionError(
                "invalid_response", "Stream finish reason is not text", retryable=True
            )
        self.finish_reason = finish_reason or self.finish_reason
        delta = self._delta(choice)
        repeated = self._append_reasoning(delta)
        self._append_content(delta)
        self._append_tool_calls(delta.get("tool_calls", []))
        if self.on_delta:
            content = delta.get("content") or ""
            reasoning = delta.get("reasoning_content") or delta.get("reasoning") or ""
            if content or reasoning:
                self.on_delta(
                    {"phase": "response_delta", "content": content, "reasoning": reasoning}
                )
            if delta.get("tool_calls"):
                latest = self.tool_calls[delta["tool_calls"][-1]["index"]]
                self.on_delta(
                    {
                        "phase": "tool_call_delta",
                        "tool": latest["function"]["name"],
                        "argument_chars": sum(
                            len(call["function"]["arguments"]) for call in self.tool_calls.values()
                        ),
                    }
                )
        if self.finish_reason == "error" and choice.get("error"):
            raise provider_error(choice["error"])
        self._check_reasoning_limit(repeated)

    def finish(self) -> Message:
        message = self.snapshot()
        if not self.saw_done and self.finish_reason is None:
            raise CompletionError(
                "incomplete_stream",
                "Response stream ended before a terminal marker",
                retryable=True,
                partial_message=message,
            )
        DeepSeekClient._validate_completion(message, self.finish_reason)
        return message


class DeepSeekClient(ModelRequestRunner):
    accounts_attempt_tokens = True

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cache_stats = CacheStats()
        self.recorder: EventRecorder | None = None
        self.deadline: float | None = None
        self.request_gate = shared_gate(settings)

    def _resolve_request_gate(self):
        return shared_gate(self.settings)

    @staticmethod
    def _new_generation_metrics(observed, on_progress):
        return GenerationMetrics(observed, on_progress)

    def fork(
        self, *, settings: Settings, cache_stats: CacheStats | None = None
    ) -> "DeepSeekClient":
        client = type(self)(settings)
        client.cache_stats = cache_stats if cache_stats is not None else self.cache_stats
        client.recorder = self.recorder
        client.deadline = self.deadline
        for name in ("retries", "backoff"):
            if hasattr(self, name):
                setattr(client, name, getattr(self, name))
        return client


    def _request_body(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        streaming: bool,
        max_output_tokens: int | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.settings.model,
            "messages": wire_messages(messages),
            "stream": streaming,
        }
        if streaming:
            body["stream_options"] = {"include_usage": True}
        if tools is not None:
            body.update(tools=tools, tool_choice="auto")

        output_tokens = self._output_tokens(messages, tools, max_output_tokens)
        if output_tokens is not None:
            body["max_tokens"] = output_tokens

        self._set_reasoning_parameters(body)
        return body

    def _output_tokens(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        max_output_tokens: int | None,
    ) -> int | None:
        output_tokens = (
            self.settings.max_output_tokens if max_output_tokens is None else max_output_tokens
        )
        if getattr(self.settings, "max_context_chars", 120000) is None:
            available = (
                context_window(self.settings)
                - estimate_context(messages, tools, self.settings)["tokens"]
                - 256
            )
            if available <= 0:
                raise CompletionError(
                    "context_overflow", "Input exceeds the configured context window"
                )
            output_tokens = (
                min(output_tokens, available) if output_tokens is not None else available
            )
        return output_tokens

    def _set_reasoning_parameters(self, body: dict[str, Any]) -> None:
        host = urlsplit(self.settings.base_url).hostname or ""
        deepseek = "deepseek" in self.settings.model.lower() or host == "api.deepseek.com"
        gateway = host in {"api.cline.bot", "openrouter.ai"}
        effort = parse_reasoning_effort(getattr(self.settings, "reasoning_effort", None))
        enabled = effort != "off"

        if deepseek:
            body["thinking"] = {"type": "enabled" if enabled else "disabled"}
        if gateway:
            body["reasoning"] = (
                {"enabled": True, "effort": effort} if enabled else {"enabled": False}
            )
        elif enabled:
            body["reasoning_effort"] = effort
        elif not deepseek:
            body["reasoning_effort"] = "none"


    @staticmethod
    def _read_nonstream_response(response: Any, observed: UsageAccumulator) -> Message:
        payload = json.load(response)
        if not isinstance(payload, dict):
            raise CompletionError(
                "invalid_response", "API response is not an object", retryable=True
            )
        if isinstance(payload.get("data"), dict):
            payload = payload["data"]
        observed.add(payload)
        if payload.get("error"):
            raise provider_error(payload["error"])
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise CompletionError("invalid_response", "API response has no choices", retryable=True)
        choice = choices[0]
        message = choice.get("message")
        if choice.get("finish_reason") == "error" and choice.get("error"):
            raise provider_error(
                choice["error"], partial_message=message if isinstance(message, dict) else None
            )
        if not isinstance(message, dict):
            raise CompletionError("invalid_response", "API response has no message", retryable=True)
        DeepSeekClient._validate_completion(message, choice.get("finish_reason"))
        return message

    def _attach_usage(
        self,
        message: Message,
        input_messages: list[Message],
        tools: list[dict[str, Any]] | None,
        observed: UsageAccumulator,
    ) -> Message:
        usage = observed.result()
        if usage is not None:
            message.setdefault(LOCAL_METADATA, {}).update(
                {
                    "usage": usage,
                    "input_messages": len(input_messages),
                    "input_fingerprint": request_fingerprint(input_messages, tools, self.settings),
                    "response_fingerprint": fingerprint(wire_messages([message])[0]),
                }
            )
        return message


    def _observe_headers(self, headers: dict[str, str]) -> None:
        """Response headers of an accepted request; providers may report quota here."""

    def _http_request(self, body: dict[str, Any]) -> urllib.request.Request:
        return urllib.request.Request(
            f"{self.settings.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode(),
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

    @staticmethod
    def _system_fingerprint(body: dict[str, Any]) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "model": body.get("model"),
                    "system": [
                        message for message in body["messages"] if message.get("role") == "system"
                    ],
                    "tools": body.get("tools"),
                    "reasoning_effort": body.get("reasoning_effort"),
                },
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    def _http_error(self, exc: urllib.error.HTTPError) -> CompletionError:
        retry_after = retry_after_seconds(exc.headers.get("Retry-After")) if exc.headers else None
        exc.close()
        # Provider error bodies may echo credentials. Never publish them; say what the
        # status usually means instead, so a misconfigured endpoint can be fixed.
        hint = HTTP_ERROR_HINTS.get(exc.code) or (
            "模型服务暂时不可用，稍后重试" if exc.code >= 500 else ""
        )
        return CompletionError(
            f"http_{exc.code}",
            f"模型服务返回 HTTP {exc.code}" + (f"：{hint}" if hint else ""),
            retryable=exc.code in {408, 429} or exc.code >= 500,
            retry_after=retry_after,
        )

    def _minimum_budget_output(self) -> int:
        if getattr(self.settings, "resolved_api", None) == "openai-responses":
            return 16
        if getattr(self.settings, "resolved_api", None) == "anthropic-messages":
            mode = getattr(self.settings, "model_compat", None)
            host = urlsplit(self.settings.base_url).hostname
            if (
                host != "api.deepseek.com"
                and not getattr(mode, "adaptive_thinking", False)
                and parse_reasoning_effort(getattr(self.settings, "reasoning_effort", None)) != "off"
            ):
                return 1025
        return 1


    @staticmethod
    def _validate_completion(message: Message, finish_reason: str | None) -> None:
        if finish_reason == "length":
            raise CompletionError(
                "output_limit",
                "Model output reached its token limit; answer is incomplete",
                partial_message=message,
            )
        if finish_reason == "error":
            raise CompletionError(
                "provider_error",
                "Provider ended generation with an error",
                retryable=True,
                partial_message=message,
            )
        if finish_reason not in {None, "stop", "tool_calls", "function_call"}:
            raise CompletionError(
                "unexpected_finish",
                f"Generation ended with {finish_reason!r}",
                partial_message=message,
            )
        if not isinstance(message, dict) or (
            message.get("content") is not None and not isinstance(message["content"], str)
        ):
            raise CompletionError(
                "invalid_response", "Completion content must be text", retryable=True
            )
        if message.get("tool_calls") is not None:
            try:
                validate_tool_calls(message["tool_calls"])
            except ValueError as exc:
                raise CompletionError(
                    "invalid_tool_calls", str(exc), retryable=True, partial_message=message
                ) from exc
        if not (message.get("content") or "").strip() and not message.get("tool_calls"):
            raise CompletionError(
                "empty_completion", "Model returned no answer or tool call", partial_message=message
            )

    @staticmethod
    def _check_stream_deadline(
        deadline: float | None,
        generation_deadline: float | None,
        finish_reason: str | None,
    ) -> None:
        now = time.monotonic()
        if deadline is not None and now >= deadline:
            raise CompletionError(
                "task_deadline", "Task deadline reached while receiving model output"
            )
        if generation_deadline is not None and now >= generation_deadline and finish_reason is None:
            raise CompletionError(
                "generation_budget", "Generation time slice ended; continue from saved progress"
            )

    @staticmethod
    def _consume_stream(
        response: Any,
        state: _StreamAccumulator,
        deadline: float | None,
        generation_deadline: float | None,
        on_usage: Callable[[Any], None] | None,
        metrics: GenerationMetrics | None = None,
    ) -> None:
        for raw_line in response:
            DeepSeekClient._check_stream_deadline(
                deadline, generation_deadline, state.finish_reason
            )
            line = raw_line.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                state.saw_done = True
                break
            payload = json.loads(data)
            if on_usage:
                on_usage(payload)
            if metrics:
                metrics.observe(payload)
            state.add_payload(payload)

    @classmethod
    def _read_stream(
        cls,
        response: Any,
        on_content: ContentCallback,
        *,
        deadline: float | None = None,
        generation_deadline: float | None = None,
        max_reasoning_chars: int | None = None,
        on_usage: Callable[[Any], None] | None = None,
        metrics: GenerationMetrics | None = None,
        on_delta: ProgressCallback | None = None,
    ) -> Message:
        state = _StreamAccumulator(on_content, max_reasoning_chars, on_delta)
        try:
            cls._consume_stream(response, state, deadline, generation_deadline, on_usage, metrics)
        except CompletionError as exc:
            exc.partial_message = state.snapshot()
            raise
        except KeyboardInterrupt as exc:
            raise CompletionInterrupted(state.snapshot()) from exc
        except http_transport.TransportCancelled as exc:
            raise CompletionError(
                "cancelled", "Model request cancelled", partial_message=state.snapshot()
            ) from exc
        except http_transport.TransportDeadline as exc:
            code = (
                "task_deadline"
                if deadline is not None and time.monotonic() >= deadline
                else "generation_budget"
            )
            raise CompletionError(
                code, "Model request deadline reached", partial_message=state.snapshot()
            ) from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CompletionError(
                "invalid_response",
                f"Cannot decode stream: {exc}",
                retryable=True,
                partial_message=state.snapshot(),
            ) from exc
        except (OSError, http.client.HTTPException) as exc:
            raise CompletionError(
                "transport_error",
                f"Stream interrupted: {exc}",
                retryable=True,
                partial_message=state.snapshot(),
            ) from exc
        finally:
            if metrics:
                metrics.publish(finished=True)
        message = state.finish()
        if metrics:
            message.setdefault(LOCAL_METADATA, {})["performance"] = metrics.snapshot()
        return message
