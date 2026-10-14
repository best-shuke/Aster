"""Shared tool contracts, validation, deadlines, and durable output storage."""

import json
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from contracts import ErrorDetails, ProgressCallback, ToolDefinition, ToolResultPayload
from execution_trace import EventSink, TraceContext, current_sink, current_trace

MAX_RESULT_CHARS = 12_000
PAGE_CHARS = 6_000


@dataclass
class ExecutionContext:
    deadline: float
    cancelled: threading.Event = field(default_factory=threading.Event)
    trace: TraceContext = field(default_factory=current_trace)
    event_sink: EventSink | None = field(default_factory=current_sink)
    on_progress: ProgressCallback | None = None

    def remaining(self) -> float:
        if self.cancelled.is_set():
            raise TimeoutError("Tool execution was cancelled")
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Tool execution deadline reached")
        return remaining

    def child(self, seconds: float) -> "ExecutionContext":
        return self.child_until(time.monotonic() + seconds)

    def child_until(self, deadline: float) -> "ExecutionContext":
        return ExecutionContext(
            min(self.deadline, deadline),
            self.cancelled,
            self.trace,
            self.event_sink,
            self.on_progress,
        )

    def for_agent(self, agent_id: str) -> "ExecutionContext":
        return ExecutionContext(
            self.deadline,
            self.cancelled,
            self.trace.agent(agent_id),
            self.event_sink,
            self.on_progress,
        )

    def report_progress(self, **details: Any) -> None:
        if self.on_progress is not None:
            # An optional display callback must not turn an executed command
            # into a failed operation. User interruption still propagates.
            with suppress(Exception):
                self.on_progress(
                    {"phase": "tool_progress", "trace": self.trace.as_dict(), **details}
                )


@dataclass
class ToolResult:
    status: str = "ok"
    data: dict[str, Any] = field(default_factory=dict)
    error: ErrorDetails | None = None
    truncated: bool = False
    next_cursor: dict[str, int] | None = None
    artifact_path: str | None = None

    def as_dict(self) -> ToolResultPayload:
        return {
            "status": self.status,
            "data": self.data,
            "error": self.error,
            "truncated": self.truncated,
            "next_cursor": self.next_cursor,
            "artifact_path": self.artifact_path,
        }

    @classmethod
    def failure(cls, code: str, message: str, status: str = "error", **kwargs: Any) -> "ToolResult":
        return cls(status=status, error={"code": code, "message": message}, **kwargs)


class ArtifactStore:
    def __init__(self, workspace: Path) -> None:
        self.directory = workspace / ".agent-artifacts"

    def new_path(self, suffix: str = ".txt") -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        return self.directory / f"{uuid4().hex}{suffix}"

    def render(self, result: ToolResult) -> str:
        serialized = json.dumps(result.as_dict(), ensure_ascii=False)
        if len(serialized) <= MAX_RESULT_CHARS:
            return serialized
        # Preserve the entire result before shrinking the envelope. Never cut JSON.
        path = self.new_path(".json")
        path.write_text(serialized, encoding="utf-8")
        summary = ToolResult(
            status=result.status,
            data={"message": "Full result saved to artifact_path. Use read to inspect it."},
            error={
                **result.error,
                "code": result.error["code"][:100],
                "message": result.error["message"][:500],
            }
            if result.error
            else None,
            truncated=True,
            next_cursor=result.next_cursor,
            artifact_path=str(path),
        )
        return json.dumps(summary.as_dict(), ensure_ascii=False)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    properties: dict[str, Any]
    required: tuple[str, ...]
    handler: Callable[..., ToolResult]
    read_only: bool = False

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": self.properties,
            "required": list(self.required),
            "additionalProperties": False,
        }

    def definition(self) -> ToolDefinition:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema,
            },
        }


def validate(value: Any, schema: dict[str, Any], location: str = "arguments") -> None:
    """Validate the JSON Schema subset used by our tool registry, including nested inputs."""
    kind = schema["type"]
    types = {"object": dict, "array": list, "string": str, "integer": int, "boolean": bool}
    if type(value) is not types[kind]:
        raise ValueError(f"{location} must be {kind}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{location} must be one of {schema['enum']}")
    if kind == "object":
        properties = schema["properties"]
        for name in schema.get("required", []):
            if name not in value:
                raise ValueError(f"{location}.{name} is required")
        for name, item in value.items():
            if name not in properties:
                additional = schema.get("additionalProperties", False)
                if additional is True:
                    continue
                if isinstance(additional, dict):
                    validate(item, additional, f"{location}.{name}")
                    continue
                raise ValueError(f"Unknown argument: {location}.{name}")
            validate(item, properties[name], f"{location}.{name}")
    elif kind == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", float("inf")):
            raise ValueError(f"{location} has an invalid number of items")
        for index, item in enumerate(value):
            validate(item, schema["items"], f"{location}[{index}]")
    elif kind == "integer":
        if not schema.get("minimum", -float("inf")) <= value <= schema.get("maximum", float("inf")):
            raise ValueError(f"{location} is outside the allowed range")
    elif kind == "string":
        if len(value) < schema.get("minLength", 0):
            raise ValueError(f"{location} must not be empty")
        if len(value) > schema.get("maxLength", float("inf")):
            raise ValueError(f"{location} exceeds the maximum length {schema['maxLength']}")


def validate_tool_calls(calls: Any) -> None:
    if not isinstance(calls, list):
        raise ValueError("tool_calls must be an array")
    identities = set()
    for call in calls:
        if not isinstance(call, dict):
            raise ValueError("Each tool call must be an object")
        identity = call.get("id")
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("Each tool call must have a nonempty string id")
        if identity in identities:
            raise ValueError("Tool call ids must be unique within a response")
        identities.add(identity)
        if call.get("type") != "function":
            raise ValueError("Tool call type must be function")
        function = call.get("function")
        if not isinstance(function, dict):
            raise ValueError("Tool call function must be an object")
        name = function.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Tool call function name must be a nonempty string")
        if not isinstance(function.get("arguments"), str):
            raise ValueError("Tool call arguments must be a JSON string")
