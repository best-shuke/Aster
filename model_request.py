"""Provider-neutral model request execution, retries and token reservations."""

import http.client
import json
import ssl
import threading
import time
import urllib.error
import urllib.request
from typing import Any

import http_transport
from contracts import ContentCallback, EventRecorder, Message, ProgressCallback, ToolDefinition
from execution_trace import (
    TraceContext, current_trace, display_callback, record_event, trace_scope, traced_progress,
)
from request_budget import (
    TokenBudgetExhausted, current_budgets, mark_dispatched, observe_usage, reserve_completion,
)
from usage import UsageAccumulator

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


class ModelRequestRunner:
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
        self.request_gate = self._resolve_request_gate()
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
                    max_output_tokens=max_output_tokens,
                    on_content=chunks.append if on_content is not None else None,
                    timeout=timeout, deadline=deadline, generation_timeout=generation_remaining,
                    purpose=purpose, cancelled=cancelled, on_progress=on_progress, _admission=admission,
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

    def _perform_request(
        self,
        request: urllib.request.Request,
        input_messages: list[Message],
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
        metrics = self._new_generation_metrics(observed, on_progress) if on_content is not None else None
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
            self._observe_headers(getattr(response, "headers", None) or {})
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
        return self._attach_usage(message, input_messages, tools, observed)

    def _complete_once(self, messages, tools=None, on_content=None, timeout=180, **options):
        if not current_budgets():
            return self._execute_attempt(messages, tools, on_content, timeout, **options)
        limit = self._output_tokens(messages, tools, options.get("max_output_tokens"))
        try:
            with reserve_completion(messages, tools, limit, minimum=self._minimum_budget_output()) as reservation:
                options["max_output_tokens"] = reservation.output_cap
                return self._execute_attempt(messages, tools, on_content, timeout, **options)
        except TokenBudgetExhausted as exc:
            raise CompletionError("budget_exhausted", str(exc)) from exc

    def _execute_attempt(
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
        system_fingerprint = self._system_fingerprint(body)
        request = self._http_request(body)
        generation_deadline, effective_deadline, remaining, io_deadline = self._request_timing(
            timeout, generation_timeout, deadline
        )
        observed = UsageAccumulator()
        answer = None
        try:
            mark_dispatched()
            answer = self._perform_request(
                request,
                messages,
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
            return answer
        except urllib.error.HTTPError as exc:
            raise self._http_error(exc) from exc
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
                f"Cannot reach model API: {exc.reason}",
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
            observe_usage(usage, answer)
            cache_stats.record(usage, purpose, system_fingerprint=system_fingerprint)
            record_event(self.recorder, "model_usage", purpose=purpose, usage=usage)
