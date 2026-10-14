import hashlib
import http.client
import json
import re
import ssl
import threading
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
    ToolDefinition,
)
from contracts import (
    Message as Message,
)
from contracts import (
    ProgressCallback as ProgressCallback,
)
from execution_trace import (
    TraceContext,
    current_trace,
    display_callback,
    record_event,
    trace_scope,
    traced_progress,
)
from generation_metrics import GenerationMetrics
from model_rate_limit import retry_after_seconds, shared_gate
from tool_runtime import validate_tool_calls
from usage import CacheStats, UsageAccumulator


def has_unexecuted_tool_text(text: str) -> bool:
    return bool(re.search(r"<(?!/)[^<>\s]*DSML[^<>\s]*\s+(?:calls|invoke)\b", text))


def clean_completion_suffix(text: str) -> str:
    """Remove orphan closing protocol tags only at the end of ordinary text."""
    if re.search(r"<(?!/)[^<>\s]*DSML[^<>\s]*\s+", text):
        return text
    closing = r"</[^<>\s]*DSML[^<>\s]*\s+(?:parameter|invoke|calls)\s*>"
    return re.sub(rf"{closing}(?:\s*{closing})*\s*$", "", text)


def has_substantial_progress(message: Message | None) -> bool:
    return bool(
        message
        and sum(len(message.get(k) or "") for k in ("content", "reasoning", "reasoning_content"))
        >= 256
    )


class CompletionError(RuntimeError):
    """An incomplete generation must never become a successful answer or tool call."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        partial_message: Message | None = None,
        retry_after: float | None = None,
        rate_limited: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.partial_message = partial_message
        self.retry_after = retry_after
        self.rate_limited = rate_limited


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

    def snapshot(self) -> Message:
        message: Message = {"role": "assistant", "content": "".join(self.content) or None}
        message.update({key: "".join(parts) for key, parts in self.reasoning.items()})
        if self.tool_calls:
            message["tool_calls"] = [self.tool_calls[index] for index in sorted(self.tool_calls)]
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


class DeepSeekClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.cache_stats = CacheStats()
        self.recorder: EventRecorder | None = None
        self.deadline: float | None = None
        self.request_gate = shared_gate(settings)

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

    @staticmethod
    def _attempt_budget(
        cancelled: threading.Event,
        deadline: float | None,
        generation_deadline: float | None,
    ) -> float | None:
        if cancelled.is_set():
            raise CompletionError("cancelled", "Model request cancelled")
        if deadline is not None and time.monotonic() >= deadline:
            raise CompletionError("task_deadline", "Task deadline reached before model request")
        remaining = (
            generation_deadline - time.monotonic() if generation_deadline is not None else None
        )
        if remaining is not None and remaining <= 0:
            raise CompletionError(
                "generation_budget", "Generation budget reached before another request"
            )
        return remaining

    @staticmethod
    def _retry_delay(
        error: CompletionError,
        attempt: int,
        retries: int,
        backoff: float,
        deadline: float | None,
        recorder: EventRecorder | None,
    ) -> float | None:
        record_event(
            recorder,
            "completion_error",
            attempt=attempt + 1,
            code=error.code,
            retryable=error.retryable,
            error=str(error),
        )
        if has_substantial_progress(error.partial_message):
            record_event(recorder, "request_checkpoint_required", reason=error.code)
            return None
        if not error.retryable or attempt >= retries:
            return None
        delay = min(30, backoff * 2**attempt)
        if error.retry_after is not None:
            delay = max(delay, error.retry_after)
        if deadline is not None and time.monotonic() + delay >= deadline:
            if error.code == "http_429":
                # Let the task retain the rate-limit observation and wait within
                # its remaining budget, rather than reporting an early timeout.
                return None
            raise CompletionError(
                "task_deadline",
                "Insufficient task time for retry backoff",
                partial_message=error.partial_message,
            ) from error
        record_event(
            recorder,
            "request_retry",
            attempt=attempt + 2,
            delay_seconds=delay,
            reason=error.code,
            policy="retry identical uncompleted request; no tool execution",
        )
        return delay

    @staticmethod
    def _wait_before_retry(
        delay: float,
        attempt: int,
        generation_deadline: float | None,
        cancelled: threading.Event,
        on_progress: ProgressCallback | None,
        reason: str | None = None,
    ) -> bool:
        until = time.monotonic() + delay
        if generation_deadline is not None:
            until = min(until, generation_deadline)
        while time.monotonic() < until:
            if on_progress:
                on_progress(
                    {
                        "phase": "rate_limit" if reason == "http_429" else "retry",
                        "attempt": attempt + 2,
                        "delay_seconds": max(0, until - time.monotonic()),
                    }
                )
            if cancelled.wait(min(0.1, max(0, until - time.monotonic()))):
                return False
        return True

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        on_content: ContentCallback | None = None,
        timeout: float = 180,
        *,
        max_output_tokens: int | None = None,
        deadline: float | None = None,
        generation_timeout: float | None = None,
        purpose: str = "solve",
        cancelled: threading.Event | None = None,
        on_progress: ProgressCallback | None = None,
        trace_context: TraceContext | None = None,
    ) -> Message:
        trace = trace_context or current_trace().request()
        if trace.request_id is None:
            trace = trace.request()
        with trace_scope(trace):
            progress = traced_progress(on_progress)
            record_event(self.recorder, "model_request_started", purpose=purpose)
            try:
                answer = self._complete_with_retries(
                    messages,
                    tools,
                    display_callback(on_content, name="content"),
                    timeout,
                    max_output_tokens=max_output_tokens,
                    deadline=deadline,
                    generation_timeout=generation_timeout,
                    purpose=purpose,
                    cancelled=cancelled,
                    on_progress=progress,
                )
            except BaseException as exc:
                record_event(
                    self.recorder,
                    "model_request_failed",
                    purpose=purpose,
                    error_type=type(exc).__name__,
                    code=getattr(exc, "code", None),
                )
                raise
            record_event(self.recorder, "model_request_completed", purpose=purpose)
            return answer

    def _complete_with_retries(
        self,
        messages,
        tools,
        on_content,
        timeout,
        *,
        max_output_tokens,
        deadline,
        generation_timeout,
        purpose,
        cancelled,
        on_progress,
    ) -> Message:
        deadlines = [d for d in (deadline, getattr(self, "deadline", None)) if d is not None]
        deadline = min(deadlines) if deadlines else None
        cancelled = cancelled if cancelled is not None else threading.Event()
        retries = getattr(self, "retries", getattr(self.settings, "request_retries", 3))
        backoff = getattr(self, "backoff", getattr(self.settings, "retry_backoff_seconds", 2))
        recorder = self.recorder
        generation_seconds = (
            generation_timeout
            if generation_timeout is not None
            else getattr(self.settings, "generation_timeout_seconds", None)
        )
        generation_deadline = time.monotonic() + generation_seconds if generation_seconds else None
        preview = on_progress if on_content is not None and purpose in {"solve", "final"} else None
        self.request_gate = shared_gate(self.settings)
        rate_limited = False
        for attempt in range(retries + 1):
            admission = self._wait_for_admission(
                cancelled, deadline, generation_deadline, on_progress, rate_limited=rate_limited
            )
            try:
                generation_remaining = self._attempt_budget(
                    cancelled, deadline, generation_deadline
                )
                chunks: list[str] = []
                if preview:
                    preview({"phase": "response_reset"})
                answer = self._complete_once(
                    messages,
                    tools=tools,
                    on_content=chunks.append if on_content is not None else None,
                    timeout=timeout,
                    max_output_tokens=max_output_tokens,
                    deadline=deadline,
                    generation_timeout=generation_remaining,
                    purpose=purpose,
                    cancelled=cancelled,
                    on_progress=on_progress,
                    _admission=admission,
                )
                self.request_gate.accepted(admission)
            except CompletionError as exc:
                if exc.code == "http_429":
                    rate_limited = True
                    self.request_gate.limited(
                        admission, retry_after=exc.retry_after, backoff=backoff
                    )
                if preview:
                    preview({"phase": "response_reset"})
                delay = self._retry_delay(exc, attempt, retries, backoff, deadline, recorder)
                if delay is None:
                    raise
                if not self._wait_before_retry(
                    delay, attempt, generation_deadline, cancelled, on_progress, exc.code
                ):
                    raise CompletionError(
                        "cancelled", "Model request cancelled during retry"
                    ) from exc
                continue
            finally:
                self.request_gate.release(admission)
            if on_content:
                for chunk in chunks:
                    on_content(chunk)
            return answer
        raise AssertionError("unreachable")

    def _wait_for_admission(
        self, cancelled, deadline, generation_deadline, on_progress, *, rate_limited=False
    ):
        last_update = None
        while True:
            try:
                self._attempt_budget(cancelled, deadline, generation_deadline)
            except CompletionError as exc:
                if exc.code == "generation_budget" and rate_limited:
                    exc.rate_limited = True
                raise
            admission, delay, probe = self.request_gate.try_start()
            if admission is not None:
                return admission
            rate_limited = True
            now = time.monotonic()
            if last_update is None or now - last_update >= 1:
                last_update = now
                record_event(
                    self.recorder, "model_rate_limit_wait", delay_seconds=delay, probe=probe
                )
                if on_progress:
                    on_progress(
                        {
                            "phase": "rate_limit",
                            "delay_seconds": delay,
                            "message": "Waiting for a recovery probe"
                            if probe
                            else "Model service cooldown",
                        }
                    )
            limits = [value - now for value in (deadline, generation_deadline) if value is not None]
            wait = min(0.1, delay, *limits) if limits else min(0.1, delay)
            if cancelled.wait(max(0, wait)):
                raise CompletionError(
                    "cancelled", "Model request cancelled while waiting for rate limit"
                )

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
        if output_tokens is not None:
            body["max_tokens"] = output_tokens

        self._set_reasoning_parameters(body)
        return body

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

    def _request_timing(
        self,
        timeout: float,
        generation_timeout: float | None,
        deadline: float | None,
    ) -> tuple[float | None, float | None, float, float | None]:
        generation_seconds = (
            generation_timeout
            if generation_timeout is not None
            else getattr(self.settings, "generation_timeout_seconds", None)
        )
        generation_deadline = time.monotonic() + generation_seconds if generation_seconds else None
        deadlines = [d for d in (deadline, getattr(self, "deadline", None)) if d is not None]
        effective_deadline = min(deadlines) if deadlines else None
        remaining = min(
            timeout,
            generation_seconds or float("inf"),
            effective_deadline - time.monotonic()
            if effective_deadline is not None
            else float("inf"),
        )
        if remaining <= 0:
            raise CompletionError("task_deadline", "Task deadline reached before model request")
        io_deadlines = [d for d in (effective_deadline, generation_deadline) if d is not None]
        io_deadline = min(io_deadlines) if io_deadlines else None
        return generation_deadline, effective_deadline, remaining, io_deadline

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

    def _perform_request(
        self,
        request: urllib.request.Request,
        body: dict[str, Any],
        tools: list[dict[str, Any]] | None,
        on_content: ContentCallback | None,
        purpose: str,
        observed: UsageAccumulator,
        generation_deadline: float | None,
        effective_deadline: float | None,
        remaining: float,
        io_deadline: float | None,
        cancelled: threading.Event | None,
        on_progress: ProgressCallback | None,
        admission=None,
    ) -> Message:
        metrics = GenerationMetrics(observed, on_progress) if on_content is not None else None
        with http_transport.open_url(
            request,
            timeout=remaining,
            deadline=io_deadline,
            cancelled=cancelled,
            on_progress=on_progress,
        ) as response:
            # Headers confirm acceptance; a long healthy reasoning stream must
            # not hold all other recovery requests behind the probe.
            self.request_gate.accepted(admission)
            if on_content is None:
                message = self._read_nonstream_response(response, observed)
            else:
                message = self._read_stream(
                    response,
                    on_content,
                    deadline=effective_deadline,
                    generation_deadline=generation_deadline,
                    max_reasoning_chars=getattr(self.settings, "max_reasoning_chars", None)
                    if purpose == "solve"
                    else None,
                    on_usage=observed.add,
                    metrics=metrics,
                    on_delta=on_progress if purpose in {"solve", "final"} else None,
                )
        return self._attach_usage(message, body["messages"], tools, observed)

    def _complete_once(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        on_content: ContentCallback | None = None,
        timeout: float = 180,
        *,
        max_output_tokens: int | None = None,
        deadline: float | None = None,
        generation_timeout: float | None = None,
        purpose: str = "solve",
        cancelled: threading.Event | None = None,
        on_progress: ProgressCallback | None = None,
        _admission=None,
    ) -> Message:
        cache_stats = self.cache_stats
        body = self._request_body(messages, tools, on_content is not None, max_output_tokens)
        system_fingerprint = hashlib.sha256(
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
        request = urllib.request.Request(
            f"{self.settings.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode(),
            headers={
                "Authorization": f"Bearer {self.settings.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        generation_deadline, effective_deadline, remaining, io_deadline = self._request_timing(
            timeout, generation_timeout, deadline
        )
        observed = UsageAccumulator()
        try:
            return self._perform_request(
                request,
                body,
                tools,
                on_content,
                purpose,
                observed,
                generation_deadline,
                effective_deadline,
                remaining,
                io_deadline,
                cancelled,
                on_progress,
                _admission,
            )
        except urllib.error.HTTPError as exc:
            retry_after = (
                retry_after_seconds(exc.headers.get("Retry-After")) if exc.headers else None
            )
            exc.close()
            # Provider error bodies may echo credentials. Never publish them.
            raise CompletionError(
                f"http_{exc.code}",
                f"Model API returned HTTP {exc.code}",
                retryable=exc.code in {408, 429} or exc.code >= 500,
                retry_after=retry_after,
            ) from exc
        except http_transport.TransportCancelled as exc:
            raise CompletionError("cancelled", "Model request cancelled") from exc
        except http_transport.TransportDeadline as exc:
            code = (
                "task_deadline"
                if effective_deadline is not None and time.monotonic() >= effective_deadline
                else "generation_budget"
            )
            raise CompletionError(code, "Model request deadline reached") from exc
        except urllib.error.URLError as exc:
            raise CompletionError(
                "connection_error",
                f"Cannot reach DeepSeek API: {exc.reason}",
                retryable=not isinstance(exc.reason, ssl.SSLCertVerificationError),
            ) from exc
        except (TimeoutError, ConnectionError, ssl.SSLError, http.client.HTTPException) as exc:
            raise CompletionError(
                "transport_error",
                f"{type(exc).__name__}: {exc}",
                retryable=not isinstance(exc, ssl.SSLCertVerificationError),
            ) from exc
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise CompletionError(
                "invalid_response", f"Cannot decode API response: {exc}", retryable=True
            ) from exc
        finally:
            usage = observed.result()
            cache_stats.record(usage, purpose, system_fingerprint=system_fingerprint)
            record_event(self.recorder, "model_usage", purpose=purpose, usage=usage)

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

    @staticmethod
    def _read_stream(
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
            DeepSeekClient._consume_stream(
                response, state, deadline, generation_deadline, on_usage, metrics
            )
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
