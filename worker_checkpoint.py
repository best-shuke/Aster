"""Retain worker state locally without changing normal model requests."""

import json
import threading
import time
from pathlib import Path

from file_state import atomic_write_text
from tool_runtime import ArtifactStore, ToolResult


class WorkerCheckpoint:
    def __init__(self, workspace: Path):
        self.artifacts = ArtifactStore(workspace)
        self._lock = threading.Lock()
        self._state = None
        self._frozen = None
        self._parts = {"content": [], "reasoning": []}
        self._previous_partial = {}
        self._observations = []
        self._inflight = None
        self.path = None
        self.started = time.monotonic()

    def capture(self, system, messages, data, trace):
        # Accepted messages are append-only objects. Copy the changing lists,
        # without serializing the whole history on every successful tool call.
        with self._lock:
            if self._frozen is not None:
                return
            self._state = {
                "version": 1,
                "trace": trace.as_dict(),
                "system": system,
                "messages": list(messages),
                "data": {**data, "checks": list(data.get("checks", []))},
            }

    def begin_generation(self):
        with self._lock:
            if self._frozen is None:
                self._parts = {"content": [], "reasoning": []}
                self._previous_partial = {}

    def tool_started(self, call):
        with self._lock:
            if self._frozen is None:
                self._inflight = call["id"]

    def tool_completed(self, call, output, trace):
        with self._lock:
            if self._frozen is None:
                self._observations.append(
                    {"call": call, "result": output, "trace": trace.as_dict()}
                )
                self._inflight = None

    def progress(self, event):
        with self._lock:
            if self._frozen is not None:
                return
            if self._state is not None and event.get("model_requests") is not None:
                self._state["data"]["model_requests"] = event["model_requests"]
            if event.get("phase") == "response_reset":
                partial = {key: "".join(parts) for key, parts in self._parts.items() if parts}
                if partial:
                    self._previous_partial = partial
                self._parts = {"content": [], "reasoning": []}
            elif event.get("phase") == "response_delta":
                for key in self._parts:
                    if isinstance(event.get(key), str) and event[key]:
                        self._parts[key].append(event[key])

    def preserve_partial(self, message):
        if not isinstance(message, dict) or not any(
            message.get(key) for key in ("content", "reasoning", "reasoning_content", "tool_calls")
        ):
            return
        with self._lock:
            if self._frozen is None:
                self._previous_partial = dict(message)
                self._parts = {"content": [], "reasoning": []}

    def failure(self, code, message, *, status="timeout"):
        with self._lock:
            if self._frozen is None:
                state = self._state or {"version": 1, "messages": [], "data": {}}
                partial = {
                    key: "".join(parts) for key, parts in self._parts.items() if parts
                } or self._previous_partial
                completed = [
                    item["tool_call_id"]
                    for item in state["messages"]
                    if item.get("role") == "tool" and item.get("tool_call_id")
                ]
                current_completed = set()
                unknown = [self._inflight] if self._inflight else []
                for observation in self._observations:
                    identity = observation["call"]["id"]
                    completed.append(identity)
                    if observation["trace"].get("request_id") == state.get("trace", {}).get(
                        "request_id"
                    ):
                        current_completed.add(identity)
                    try:
                        result = json.loads(observation["result"])
                    except (TypeError, ValueError):
                        result = {}
                    if not isinstance(result, dict) or result.get("status") in {None, "unknown"}:
                        unknown.append(identity)
                last_assistant = next(
                    (
                        index
                        for index in range(len(state["messages"]) - 1, -1, -1)
                        if state["messages"][index].get("role") == "assistant"
                    ),
                    None,
                )
                group = state["messages"][last_assistant:] if last_assistant is not None else []
                group_completed = {
                    item.get("tool_call_id") for item in group if item.get("role") == "tool"
                } | current_completed
                pending = [
                    call["id"]
                    for item in group[:1]
                    for call in item.get("tool_calls") or []
                    if call.get("id") and call["id"] not in group_completed
                ]
                self._frozen = {
                    **state,
                    "unfinished_response": partial,
                    "unfinished_response_verified": False,
                    "completed_tool_call_ids": list(dict.fromkeys(completed)),
                    "pending_tool_call_ids": [
                        identity for identity in pending if identity != self._inflight
                    ],
                    "unknown_tool_call_ids": list(dict.fromkeys(unknown)),
                    "tool_observations": list(self._observations),
                }
            state = self._frozen
            path = self.path
        data = {
            **state["data"],
            "report": state["data"].get("report") or "Investigation did not finish.",
            "duration_seconds": round(time.monotonic() - self.started, 3),
            "tool_calls": max(len(state["tool_observations"]), state["data"].get("tool_calls", 0)),
            "recovery_guidance": (
                "Continue from the checkpoint's observations and remaining assignment. "
                "Unfinished output and pending tool calls are unverified. "
                "Do not replay completed calls; inspect actual state for operations with unknown effects."
            ),
        }
        partial = state["unfinished_response"]
        if partial.get("content"):
            data.update(report=partial["content"], report_verified=False)
        try:
            if path is None:
                path = self.artifacts.new_path(".worker-checkpoint.json")
                atomic_write_text(path, json.dumps(state, ensure_ascii=False, indent=2))
                with self._lock:
                    self.path = path
            data["checkpoint_path"] = str(path)
        except OSError as exc:
            # A failed recovery write must not erase the in-memory evidence or
            # turn a completed operation into something to execute again.
            data.update(checkpoint=state, checkpoint_error=str(exc))
        return ToolResult.failure(code, message, status=status, data=data)
