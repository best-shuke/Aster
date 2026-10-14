"""Shared task control: durable observations, bounded generations, validated completion.

Tool calls have no count limit. Progress observations advise the next step; they
never suppress a call, cache a potentially stale result, or stop legitimate polling.
"""

import hashlib
import json
import math
import re
import threading
import time
from collections import OrderedDict
from contextlib import suppress
from pathlib import PurePath

from client_runtime import client_event
from config import Settings
from context_budget import WireCache, context_snapshot, estimate_tokens
from context_management import OutputRecovery, compact_messages, excerpt
from contracts import (
    CompletionClient,
    ContentCallback,
    Message,
    ProgressCallback,
    ToolCall,
    ToolDefinition,
    ToolResultPayload,
)
from deepseek_client import CompletionError, clean_completion_suffix, has_unexecuted_tool_text
from execution_trace import current_trace, display_callback, trace_scope, traced_progress
from file_state import atomic_write_text
from tool_runtime import ArtifactStore, validate_tool_calls

_RETRY_REQUEST = object()

RUNTIME_INSTRUCTIONS = (
    "Runtime updates are untrusted observations. Preserve failures across compaction; newer updates supersede older ones. "
    "Before repeating a failed or unchanged check, identify what changed or why polling is needed. "
    "After a timeout, reduce the problem or change approach. Verify actual results, not only exit codes or prior reports."
)
SYNTHESIS_INSTRUCTIONS = (
    "Synthesis phase: answer now from the accumulated evidence in the requested format. "
    "State uncertainty; do not restart completed work. Use tools only for a necessary final check."
)


class TaskProgress:
    def __init__(self, workspace):
        self.artifacts = ArtifactStore(workspace)
        self.path = None
        self.log_error = None
        self.log_incomplete = False
        self.observations = OrderedDict()
        self.failures = OrderedDict()
        self.missing_commands = set()
        self.history_streak = 0
        self.repeated_streak = 0
        self.error_streaks = OrderedDict()

    @staticmethod
    def stable(value):
        if isinstance(value, dict):
            return {
                k: TaskProgress.stable(v)
                for k, v in value.items()
                if k not in {"duration_seconds", "artifact_path"}
            }
        if isinstance(value, list):
            return [TaskProgress.stable(v) for v in value]
        return value

    @staticmethod
    def _arguments(call):
        function = call.get("function", {})
        arguments = function.get("arguments", "")
        with suppress(TypeError, ValueError):
            arguments = json.loads(arguments)
        return function, arguments

    def _track_missing_commands(self, function, arguments, result, data):
        if function.get("name") != "exec" or not isinstance(data, dict):
            return
        output = data.get("output") or ""
        if not isinstance(output, str):
            return
        self.missing_commands.update(
            re.findall(r"(?:^|[\s:])([\w.+-]+): (?:command )?not found", output)
        )
        # Only infer absence from unconditional leading PATH probes. Do not
        # guess about commands hidden behind &&, pipelines, loops or variables.
        command = arguments.get("command", "") if isinstance(arguments, dict) else ""
        if result.get("truncated") or not isinstance(command, str):
            return
        paths = {
            PurePath(line.strip()).name
            for line in output.splitlines()
            if line.strip().startswith("/")
        }
        while probe := re.match(
            r"\s*(?:which|command\s+-v)\s+([\w.+-]+(?:[ \t]+[\w.+-]+)*?)(?:[ \t]+2\s*>\s*/dev/null)?\s*(?:;|$)",
            command,
        ):
            for name in probe[1].split():
                if name in paths:
                    self.missing_commands.discard(name)
                else:
                    self.missing_commands.add(name)
            command = command[probe.end() :]

    def _update_streaks(self, arguments, same):
        argument_text = json.dumps(arguments, ensure_ascii=False)
        history_markers = (".agent-artifacts", ".unfinished.json", ".output-limit.json")
        self.history_streak = (
            self.history_streak + 1
            if any(marker in argument_text for marker in history_markers)
            else 0
        )
        self.repeated_streak = self.repeated_streak + 1 if same > 1 else 0

    def _store(self, signature, record, failed):
        self.observations[signature] = record
        self.failures.pop(signature, None)
        if failed:
            self.failures[signature] = record
        for mapping, limit in ((self.observations, 128), (self.failures, 32)):
            while len(mapping) > limit:
                mapping.popitem(last=False)
        return self._write_log(record)

    def _write_log(self, record):
        line = json.dumps(record, ensure_ascii=False) + "\n"
        previous_error = self.log_error
        had_path = self.path is not None
        recreated = False
        try:
            for attempt in range(2):
                try:
                    if self.path is None:
                        self.path = self.artifacts.new_path(".progress.jsonl")
                    elif had_path and not self.path.exists():
                        recreated = self.log_incomplete = True
                    self.path.parent.mkdir(parents=True, exist_ok=True)
                    with self.path.open("a", encoding="utf-8") as stream:
                        stream.write(line)
                    break
                except FileNotFoundError:
                    if had_path:
                        recreated = self.log_incomplete = True
                    if attempt:
                        raise
        except OSError as exc:
            self.log_error = str(exc)
            self.log_incomplete = True
            if self.log_error != previous_error:
                return {
                    "phase": "warning",
                    "code": "progress_log_unavailable",
                    "message": "进度日志暂时无法写入，工具结果保留，任务继续执行："
                    + excerpt(self.log_error, 500),
                }
            return None
        self.log_error = None
        if previous_error is not None or recreated:
            return {
                "phase": "warning",
                "code": "progress_log_restored" if previous_error else "progress_log_recreated",
                "message": "进度日志已恢复写入，任务继续；此前日志可能缺失。"
                if previous_error
                else "进度日志目录或文件消失，已自动重建；任务继续，旧日志可能缺失。",
            }
        return None

    def observe(self, call: ToolCall, result: ToolResultPayload):
        function, arguments = self._arguments(call)
        signature = json.dumps(
            [function.get("name"), arguments], sort_keys=True, ensure_ascii=False
        )
        data = result.get("data") or {}
        normalized = self.stable(result)
        if result.get("truncated"):
            normalized["full_result_ref"] = result.get("artifact_path")
        fingerprint = hashlib.sha256(
            json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        previous = self.observations.pop(signature, None)
        same = (
            previous["same_result_count"] + 1
            if previous and previous["result_hash"] == fingerprint
            else 1
        )
        text = json.dumps(data, ensure_ascii=False)
        self._track_missing_commands(function, arguments, result, data)
        self._update_streaks(arguments, same)
        record = {
            "trace": current_trace().as_dict(),
            "tool": function.get("name"),
            "arguments": excerpt(json.dumps(arguments, ensure_ascii=False), 500),
            "status": result.get("status"),
            "error": result.get("error"),
            "result": excerpt(text, 650),
            "result_hash": fingerprint,
            "same_result_count": same,
            "artifact_path": result.get("artifact_path"),
        }
        name = function.get("name")
        error = result.get("error") or {}
        if result.get("status") == "error" and error.get("code"):
            family = (name, error["code"])
            count = self.error_streaks.pop(family, 0) + 1
            self.error_streaks[family] = count
            while len(self.error_streaks) > 32:
                self.error_streaks.popitem(last=False)
        elif result.get("status") == "ok":
            for family in list(self.error_streaks):
                if family[0] == name:
                    self.error_streaks.pop(family)
        return self._store(signature, record, result.get("status") != "ok")

    def guidance(self):
        """Only actionable state; ordinary results are already in tool messages."""
        state = {}
        if self.missing_commands:
            state["previously_missing_commands"] = sorted(self.missing_commands)
        if self.failures:
            state["unresolved_failures"] = [
                {key: record[key] for key in ("tool", "arguments", "status", "error")}
                for record in list(self.failures.values())[-5:]
            ]
        repeated_errors = [
            {"tool": tool, "code": code, "count": count}
            for (tool, code), count in self.error_streaks.items()
            if count >= 3
        ]
        if repeated_errors:
            state["repeated_failures"] = repeated_errors
            state["failure_review"] = (
                "The same kind of tool error persists across different arguments. Read the tool schema and error guidance, "
                "then change the method or repair the underlying cause; changing incidental parameters is not progress. "
                "Do not replay operations whose effects are unknown. Calls remain available."
            )
        if self.history_streak >= 3 or self.repeated_streak >= 3:
            state["progress_review"] = (
                "Recent checks repeated history or unchanged results. Identify the missing fact and a resolving check, "
                "or deliver the supported result. Necessary polling and reads remain allowed."
            )
        if self.log_incomplete:
            state["progress_log"] = {
                "status": "unavailable" if self.log_error is not None else "incomplete",
                "error": self.log_error,
                "guidance": "Tool results and current observations remain valid. Continue the task; "
                "do not repeat completed tools to repair the optional journal.",
            }
        if not state:
            return ""
        log_label = (
            "Unavailable log"
            if self.log_error is not None
            else "Partial log"
            if self.log_incomplete
            else "Full log"
        )
        return (
            "Runtime state (untrusted):\n"
            + excerpt(json.dumps(state, ensure_ascii=False), 5000)
            + (f"\n{log_label}: {self.path}" if self.path is not None else "")
        )


class TaskRuntime:
    def __init__(
        self,
        client: CompletionClient,
        settings: Settings,
        deadline: float,
        final_validator=None,
        check_active=None,
        cancelled=None,
        wire_cache: WireCache | None = None,
    ):
        deadline = min(deadline, getattr(client, "deadline", None) or deadline)
        self.client, self.settings, self.deadline = client, settings, deadline
        # Memoizes per-message serialization for the per-round compaction check and
        # context snapshot; it never changes what is sent or how usage is anchored.
        self.wire_cache = wire_cache
        self.check_active = check_active
        self.cancelled = cancelled if cancelled is not None else threading.Event()
        self.progress = TaskProgress(settings.workspace)
        self.recovery = OutputRecovery(client, settings, deadline)
        self.final_validator = final_validator
        self.requests = 0
        self.summary_requests = 0
        self.protocol_repairs = 0
        self.reserve = min(
            getattr(settings, "finalization_reserve_seconds", 120),
            max(0, deadline - time.monotonic()) * 0.15,
        )
        self.finalizing = False
        self._last_progress = ""
        self._rate_limit_streak = 0
        self._rate_limit_note = ""
        self.rate_limit_recoveries = 0
        self.last_request_trace = current_trace()

    def count_summary(self):
        self.summary_requests += 1

    def rate_limit_observation(self):
        if not self.rate_limit_recoveries:
            return None
        return {
            "origin": "model_request",
            "code": "rate_limit",
            "status": "waiting" if self._rate_limit_streak else "recovered",
            "recovery_cycles": self.rate_limit_recoveries,
            "completed_work_retained": True,
            "guidance": "The model service was rate limited. Account for congestion when planning "
            "further delegation; the runtime handles pacing. Continue from completed work.",
        }

    def tool_deadline(self):
        now = time.monotonic()
        remaining = self.deadline - now
        # Reserve time for a result after long commands, without imposing a call count.
        return min(
            self.deadline,
            now + (remaining - self.reserve if remaining > self.reserve else remaining / 2),
        )

    def _remaining(self, message="Task execution deadline reached"):
        if self.cancelled.is_set():
            raise CompletionError("cancelled", "Task cancelled")
        if self.check_active:
            self.check_active()
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise CompletionError("task_deadline", message)
        return remaining

    def _request_notes(self, repair, remaining):
        notes = []
        progress = self.progress.guidance()
        if progress != self._last_progress:
            notes.append(progress or "Runtime state: previous warnings cleared.")
            self._last_progress = progress

        if remaining <= self.reserve and not self.finalizing:
            self.finalizing = True
            client_event(self.client, "finalization_started", remaining_seconds=remaining)
            notes.append(SYNTHESIS_INSTRUCTIONS)

        state = progress or (
            "Runtime state: previous warnings cleared." if self.progress.observations else ""
        )
        snapshot = [
            part
            for part in (
                state,
                self._rate_limit_note,
                SYNTHESIS_INSTRUCTIONS if self.finalizing else "",
                repair,
            )
            if part
        ]
        if repair:
            notes.append(repair)
        return notes, snapshot

    def _compact_request(
        self,
        messages,
        current_system,
        definitions,
        notes,
        snapshot,
        on_progress,
    ):
        # Reintroduce actionable observations only after compaction, even if
        # the summarizer omitted them. Reserve their space before compacting.
        reserved = max(
            len(json.dumps({"role": "system", "content": "\n\n".join(parts)}, ensure_ascii=False))
            if parts
            else 0
            for parts in (notes, snapshot)
        )
        reserved_tokens = max(
            estimate_tokens({"role": "system", "content": "\n\n".join(parts)}) if parts else 0
            for parts in (notes, snapshot)
        )
        compacted = compact_messages(
            messages,
            current_system,
            definitions,
            self.client,
            self.settings,
            self.deadline,
            self.count_summary,
            self.cancelled,
            on_progress,
            reserve_chars=reserved + 2 if reserved else 0,
            reserve_tokens=reserved_tokens + 2 if reserved_tokens else 0,
            cache=self.wire_cache,
        )
        selected_notes = snapshot if compacted else notes
        if selected_notes:
            messages.append({"role": "system", "content": "\n\n".join(selected_notes)})

    def _generation_time(self, remaining):
        generation_time = min(
            getattr(self.settings, "generation_timeout_seconds", None) or remaining, remaining
        )
        if remaining > self.reserve:
            generation_time = min(generation_time, remaining - self.reserve)
        return max(0.001, generation_time)

    def _recover_rate_limit(self, error, messages, on_progress):
        # A model API rejection is an execution observation, not a failed tool
        # or a completed answer. Keep one note per congestion episode so a long
        # outage cannot grow the prompt on every probe. Never rewrite history.
        base = max(1, getattr(self.settings, "retry_backoff_seconds", 2))
        delay = min(60, base * 2 ** min(self._rate_limit_streak, 6))
        retry_after = error.retry_after
        if retry_after is not None and math.isfinite(retry_after) and retry_after >= 0:
            delay = max(delay, retry_after)
        if error.partial_message:
            self.recovery.preserve_terminal(error.partial_message, messages, error.code)
        if not self._rate_limit_streak:
            observation = {
                "origin": "model_request",
                "status": "error",
                "error": {"code": error.code, "retryable": True, "rate_limited": True},
                "recovery": {"action": "wait_then_retry", "initial_delay_seconds": delay},
                "tool_calls_executed_by_this_request": False,
            }
            self._rate_limit_note = (
                "Runtime execution result:\n"
                + json.dumps(observation, ensure_ascii=False)
                + "\nThe previous model request could not complete because of service rate limiting. "
                "The runtime coordinates cooldown and recovery probes within the original task deadline. "
                "Earlier completed tool results remain valid; unfinished response calls were not executed. "
                "Continue from that evidence. Account for service congestion when planning further delegation; "
                "do not restart completed tasks or replay operations with unknown effects."
            )
            messages.append({"role": "system", "content": self._rate_limit_note})
        self._rate_limit_streak = min(7, self._rate_limit_streak + 1)
        self.rate_limit_recoveries += 1
        client_event(
            self.client,
            "model_rate_limit_recovery",
            code=error.code,
            recovery=self._rate_limit_streak,
            delay_seconds=delay,
        )
        until = time.monotonic() + delay
        last_update = None
        while True:
            remaining = self._remaining(
                "Task deadline reached while recovering from model rate limiting"
            )
            now = time.monotonic()
            if now >= until:
                return
            if on_progress and (last_update is None or now - last_update >= 1):
                last_update = now
                on_progress(
                    {
                        "phase": "rate_limit",
                        "delay_seconds": until - now,
                        "message": "Model service recovery; completed work retained",
                    }
                )
            self.cancelled.wait(min(0.1, remaining, until - now))

    def _request_model(
        self,
        messages,
        current_system,
        definitions,
        remaining,
        on_progress,
    ):
        try:
            # Publish only accepted responses, so a partial answer or textual
            # tool call cannot masquerade as a completed response in the UI.
            self.last_request_trace = current_trace().request()
            with trace_scope(self.last_request_trace):
                client_event(
                    self.client,
                    "model_call_started",
                    purpose="final" if self.finalizing else "solve",
                )
                try:
                    response = self.client.complete(
                        [current_system, *messages],
                        tools=definitions or None,
                        on_content=lambda _: None,
                        timeout=min(self.settings.request_timeout_seconds, remaining),
                        deadline=self.deadline,
                        generation_timeout=self._generation_time(remaining),
                        max_output_tokens=8192
                        if self.finalizing
                        else self.settings.max_output_tokens,
                        purpose="final" if self.finalizing else "solve",
                        cancelled=self.cancelled,
                        on_progress=traced_progress(on_progress),
                        trace_context=self.last_request_trace,
                    )
                except BaseException as exc:
                    client_event(self.client, "model_call_failed", error_type=type(exc).__name__)
                    raise
                client_event(self.client, "model_call_completed")
                self._rate_limit_streak = 0
                return response
        except KeyboardInterrupt as exc:
            self.cancelled.set()
            with trace_scope(self.last_request_trace):
                self.recovery.preserve_interruption(getattr(exc, "partial_message", None), messages)
            raise
        except CompletionError as exc:
            if self.check_active:
                try:
                    self.check_active()
                except TimeoutError as stopped:
                    self.recovery.preserve_terminal(exc.partial_message, messages, "task_deadline")
                    # Keep the original response available to callers even
                    # when the shared execution deadline raises first.
                    stopped.partial_message = exc.partial_message
                    raise
            with trace_scope(self.last_request_trace):
                if (exc.code == "http_429" and exc.retryable) or (
                    exc.code == "generation_budget" and exc.rate_limited and not exc.partial_message
                ):
                    try:
                        self._recover_rate_limit(exc, messages, on_progress)
                    except KeyboardInterrupt:
                        self.cancelled.set()
                        raise
                    return _RETRY_REQUEST
                if self.recovery.recover(exc, messages):
                    return _RETRY_REQUEST
            if (
                exc.code == "generation_budget"
                and 0 < self.deadline - time.monotonic() <= self.reserve
                and not self.finalizing
            ):
                return _RETRY_REQUEST
            raise

    def _response_problem(self, response, definitions):
        if not isinstance(response, dict):
            return "The response must be an object."
        if response.get("content") is not None and not isinstance(response["content"], str):
            return "Response content must be text."
        content = response.get("content") or ""
        if response.get("tool_calls") is not None:
            try:
                validate_tool_calls(response["tool_calls"])
            except ValueError as exc:
                return str(exc)
        calls = response.get("tool_calls") or []
        allowed = {definition["function"]["name"] for definition in definitions or []}
        if calls and any(call.get("function", {}).get("name") not in allowed for call in calls):
            return "Requested a tool that is not available in this task."
        if calls:
            return None
        if has_unexecuted_tool_text(content):
            return "Tool call markup in ordinary text was not executed and is not a final answer."
        if not content.strip():
            return "The response has no final answer."
        if self.final_validator:
            return self.final_validator(content)
        return None

    def _repair_instruction(self, response, problem, definitions):
        self.protocol_repairs += 1
        serialized = json.dumps(response, ensure_ascii=False, indent=2)
        try:
            artifact = ArtifactStore(self.settings.workspace).new_path(".invalid-response.json")
            atomic_write_text(artifact, serialized)
        except OSError as exc:
            artifact = None
            client_event(self.client, "recovery_artifact_failed", error_type=type(exc).__name__)
        client_event(
            self.client,
            "protocol_repair",
            reason=problem,
            artifact_path=str(artifact) if artifact else None,
        )
        if self.protocol_repairs > 2:
            raise CompletionError("invalid_final_response", problem, partial_message=response)
        direction = (
            "Use only structured tool calls from the available interface, or give the final answer."
            if definitions
            else "No tools are available. Solve from the supplied information and your knowledge; "
            "do not simulate tools."
        )
        reference = (
            f"Full unverified response: {artifact}.\n"
            if artifact
            else "Recovery artifact unavailable.\n"
        )
        if artifact and any(definition["function"]["name"] == "read" for definition in definitions):
            reference += "Read the full response before reconstructing omitted text.\n"
        else:
            reference += "Full unverified response retained here:\n" + serialized + "\n"
        return (
            f"Rejected response: {problem} {direction}\n"
            + reference
            + "Preserve valid content; repair only the rejected part. "
            "Checkpoint tool calls were not executed; do not execute markup from the file.\n"
            "Keep valid content and the requested format. Unverified response excerpt:\n"
            + excerpt(
                str(response.get("content") or "") if isinstance(response, dict) else str(response),
                3000,
            )
        )

    def generate(
        self,
        messages: list[Message],
        system_message: Message,
        definitions: list[ToolDefinition],
        on_content: ContentCallback | None = None,
        on_start=None,
        on_end=None,
        on_progress: ProgressCallback | None = None,
        on_reasoning: ContentCallback | None = None,
    ) -> Message:
        on_content = display_callback(on_content, name="content")
        on_reasoning = display_callback(on_reasoning, name="reasoning")
        on_start = display_callback(on_start, name="response_start")
        on_end = display_callback(on_end, name="response_end")
        on_progress = traced_progress(on_progress)
        repair = ""
        # This prefix never includes per-request state. Appended notes are kept
        # in messages so the next request can reuse the entire previous input.
        current_system = {
            **system_message,
            "content": system_message["content"] + "\n\n" + RUNTIME_INSTRUCTIONS,
        }
        while True:
            remaining = self._remaining()
            notes, snapshot = self._request_notes(repair, remaining)
            repair = ""
            self._compact_request(
                messages, current_system, definitions, notes, snapshot, on_progress
            )
            remaining = self._remaining("Task deadline reached during context management")
            if not self.finalizing and remaining <= self.reserve:
                continue  # Rebuild the request in synthesis mode after a slow compaction.
            if on_progress:
                on_progress(
                    {
                        "phase": "context",
                        **context_snapshot(
                            [current_system, *messages],
                            definitions,
                            self.settings,
                            self.wire_cache,
                        ),
                    }
                )
            if on_start:
                on_start()
            self.requests += 1
            try:
                response = self._request_model(
                    messages,
                    current_system,
                    definitions,
                    remaining,
                    on_progress,
                )
                if response is _RETRY_REQUEST:
                    continue
                if (
                    isinstance(response, dict)
                    and not response.get("tool_calls")
                    and isinstance(response.get("content"), str)
                ):
                    content = response["content"]
                    cleaned = clean_completion_suffix(content)
                    if cleaned != content:
                        response = {**response, "content": cleaned}
                        client_event(
                            self.client,
                            "response_suffix_normalized",
                            removed_chars=len(content) - len(cleaned),
                        )
                if time.monotonic() >= self.deadline:
                    raise CompletionError(
                        "task_deadline",
                        "Task deadline reached before accepting the response",
                        partial_message=response,
                    )
                problem = self._response_problem(response, definitions)
                if problem:
                    with trace_scope(self.last_request_trace):
                        repair = self._repair_instruction(response, problem, definitions)
                    continue
                if time.monotonic() >= self.deadline:
                    raise CompletionError(
                        "task_deadline",
                        "Task deadline reached during final acceptance",
                        partial_message=response,
                    )
                self.protocol_repairs = 0
                content = response.get("content") or ""
                reasoning = response.get("reasoning_content") or response.get("reasoning") or ""
                if on_reasoning and reasoning:
                    on_reasoning(reasoning)
                if on_content and content:
                    on_content(content)
                return response
            except CompletionError as exc:
                if exc.code in {"task_deadline", "cancelled"}:
                    self.recovery.preserve_terminal(exc.partial_message, messages, exc.code)
                raise
            finally:
                if on_end:
                    on_end()
