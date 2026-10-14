"""Correlate execution events without adding metadata to model messages."""

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager, suppress
from contextvars import ContextVar
from dataclasses import dataclass, fields, replace
from typing import Any, TypedDict
from uuid import uuid4

from contracts import EventRecorder, ProgressCallback


@dataclass(frozen=True)
class TraceContext:
    session_id: str | None = None
    task_id: str | None = None
    agent_id: str | None = None
    request_id: str | None = None
    tool_call_id: str | None = None
    execution_id: str | None = None
    parent_execution_id: str | None = None

    @classmethod
    def task(cls, session_id: str) -> "TraceContext":
        return cls(session_id=session_id, task_id=uuid4().hex)

    def request(self) -> "TraceContext":
        return replace(self, request_id=uuid4().hex)

    def tool(self, call_id: str, execution_id: str | None = None) -> "TraceContext":
        return replace(self, tool_call_id=call_id, execution_id=execution_id or uuid4().hex)

    def agent(self, agent_id: str) -> "TraceContext":
        return replace(
            self,
            agent_id=agent_id,
            request_id=None,
            tool_call_id=None,
            parent_execution_id=self.execution_id,
            execution_id=None,
        )

    def as_dict(self) -> dict[str, str]:
        return {
            item.name: value
            for item in fields(self)
            if (value := getattr(self, item.name)) is not None
        }


class TraceEvent(TypedDict):
    event: str
    timestamp: float
    trace: dict[str, str]
    data: dict[str, Any]


EventSink = Callable[[TraceEvent], None]
_trace: ContextVar[TraceContext | None] = ContextVar("execution_trace", default=None)
_sink: ContextVar[EventSink | None] = ContextVar("execution_event_sink", default=None)


def current_trace() -> TraceContext:
    return _trace.get() or TraceContext()


def current_sink() -> EventSink | None:
    return _sink.get()


@contextmanager
def trace_scope(trace: TraceContext, sink: EventSink | None = None) -> Iterator[None]:
    trace_token = _trace.set(trace)
    sink_token = _sink.set(sink if sink is not None else current_sink())
    try:
        yield
    finally:
        _sink.reset(sink_token)
        _trace.reset(trace_token)


def emit_event(name: str, **details: Any) -> None:
    sink = current_sink()
    if sink is not None:
        with suppress(Exception):
            sink(
                {
                    "event": name,
                    "timestamp": time.time(),
                    "trace": current_trace().as_dict(),
                    "data": details,
                }
            )


def record_event(recorder: EventRecorder | None, name: str, **details: Any) -> None:
    emit_event(name, **details)
    if recorder is not None:
        with suppress(Exception):
            recorder.event(name, **details, trace=current_trace().as_dict())


def display_callback(callback, *, name="progress"):
    """Keep optional display failures separate from execution and interruption."""
    if callback is None:
        return None

    def publish(*args, **kwargs):
        try:
            return callback(*args, **kwargs)
        except Exception as exc:
            emit_event("display_callback_failed", callback=name, error_type=type(exc).__name__)

    return publish


def traced_progress(callback: ProgressCallback | None) -> ProgressCallback | None:
    if callback is None:
        return None

    protected = display_callback(callback)

    def publish(event):
        protected({**event, "trace": event.get("trace", current_trace().as_dict())})

    return publish
