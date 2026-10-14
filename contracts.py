"""Typed contracts at the model, tool, and event boundaries."""

import threading
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypedDict, runtime_checkable

if TYPE_CHECKING:
    from config import Settings
    from execution_trace import TraceContext
    from usage import CacheStats


class ToolFunction(TypedDict):
    name: str
    arguments: str


class ToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: ToolFunction


class MessageFields(TypedDict, total=False):
    content: str | None
    tool_calls: list[ToolCall]
    tool_call_id: str
    reasoning: str
    reasoning_content: str
    _harness: dict[str, Any]


class Message(MessageFields):
    role: Literal["system", "user", "assistant", "tool"]


class FunctionDefinition(TypedDict):
    name: str
    description: str
    parameters: dict[str, Any]


class ToolDefinition(TypedDict):
    type: Literal["function"]
    function: FunctionDefinition


class ErrorOptions(TypedDict, total=False):
    retryable: bool


class ErrorDetails(ErrorOptions):
    code: str
    message: str


class ToolResultPayload(TypedDict):
    status: str
    data: dict[str, Any]
    error: ErrorDetails | None
    truncated: bool
    next_cursor: dict[str, int] | None
    artifact_path: str | None


class ToolRunFields(TypedDict, total=False):
    result: str
    trace: dict[str, str]
    arguments: str
    task_title: str
    started_at: float
    finished_at: float
    duration_seconds: float


class ToolRun(ToolRunFields):
    execution_id: str
    assistant_index: int
    tool_call_id: str
    tool: str
    state: Literal["pending", "running", "completed", "unknown", "not_executed"]


class ProgressFields(TypedDict, total=False):
    trace: dict[str, str]
    request_id: str
    generation_id: str
    finished: bool
    message: str
    content: str
    reasoning: str
    code: str
    mode: str
    title: str
    task: str
    agent_id: str
    state: str
    activity: str
    model_phase: str
    model_requests: int
    tool_calls: int
    tool: str
    arguments: dict[str, Any]
    error: ErrorDetails | None
    tokens: int
    source: str
    generation_seconds: float
    excluded_wait_seconds: float | None
    tps: float | None
    tool_status: str
    duration_seconds: float
    elapsed_seconds: float
    received_bytes: int
    output: str
    artifact_path: str
    argument_chars: int
    attempt: int
    delay_seconds: float


class ProgressEvent(ProgressFields):
    phase: str


class WorkerProgressEvent(ProgressFields):
    pass


ContentCallback = Callable[[str], None]
ProgressCallback = Callable[[ProgressEvent], None]
WorkerProgressCallback = Callable[[WorkerProgressEvent], None]


class EventRecorder(Protocol):
    def event(self, name: str, **details: Any) -> None: ...


class CompletionClient(Protocol):
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
        trace_context: "TraceContext | None" = None,
    ) -> Message: ...


@runtime_checkable
class ConfigurableCompletionClient(CompletionClient, Protocol):
    settings: "Settings"
    cache_stats: "CacheStats"
    recorder: EventRecorder | None
    deadline: float | None

    def fork(
        self,
        *,
        settings: "Settings",
        cache_stats: "CacheStats | None" = None,
    ) -> "ConfigurableCompletionClient": ...
