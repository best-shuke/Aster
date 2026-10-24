import base64
import io
import json
import os
import re
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager, nullcontext, suppress
from copy import deepcopy
from dataclasses import dataclass, replace
from itertools import pairwise
from pathlib import Path
from typing import Any, TextIO
from uuid import uuid4

from client_runtime import snapshot_client
from config import MAX_SUBAGENTS
from contracts import CompletionClient, ProgressCallback, ToolCall, ToolDefinition
from execution_trace import display_callback, emit_event, trace_scope
from file_changes import MAX_AUTO_MERGE_CHARS, EditLedger, FileTransaction, text_digest
from file_state import FileChangedError, atomic_write_text, file_digest, file_lock
from mcp_client import McpManager
from process_monitor import ProcessMonitor
from reasoning import ReasoningTools
from tool_runtime import (
    DELEGATE_RESULT_CHARS,
    MAX_RESULT_CHARS,
    PAGE_CHARS,
    READ_PAGE_CHARS,
    ArtifactStore,
    ExecutionContext,
    ToolResult,
    ToolSpec,
    command_environment,
    validate,
)
from websearch import ChatGPTWebSearchClient, WebSearchClient

TEXT = {"type": "string"}
PATH = {"type": "string", "minLength": 1}
TIMEOUT = {
    "type": "integer",
    "minimum": 1,
    "maximum": 3600,
    "description": "Maximum elapsed seconds; also bounded by the task deadline.",
}
OFFSET = {
    "type": "integer",
    "minimum": 1,
    "default": 1,
    "description": "1-based position. Follow next_cursor to continue.",
}
LIMIT = {
    "type": "integer",
    "minimum": 1,
    "maximum": 2000,
    "description": "Maximum items or lines; the output size budget may return fewer.",
}
CURSOR = {
    "type": "object",
    "properties": {"offset": OFFSET},
    "required": ["offset"],
    "additionalProperties": False,
    "description": "The next_cursor object returned by the previous page; keep other query parameters unchanged.",
}
READ_CURSOR = {
    **CURSOR,
    "properties": {"offset": OFFSET, "column": {"type": "integer", "minimum": 0}},
}

TOOL_NAMES = frozenset(
    {
        "read",
        "write",
        "edit",
        "grep",
        "find",
        "ls",
        "exec",
        "monitor",
        "update_plan",
        "websearch",
        "mcp",
        "ask_user",
        "delegate",
        "team",
    }
)

# Subagents may never use these, even when an environment grants extra worker tools:
# workers must not delegate, ask the user, rewrite the parent task plan, or join a team.
WORKER_EXCLUDED = frozenset({"delegate", "ask_user", "update_plan", "monitor", "team"})
MERGED_NOTE = (
    "Another agent changed this file since you read it; your change was merged with theirs. "
    "Read the file before relying on its exact contents."
)


@dataclass(frozen=True)
class ToolPolicy:
    allow: frozenset[str] | None = None
    deny: frozenset[str] = frozenset()

    def permits(self, name: str) -> bool:
        return name not in self.deny and (self.allow is None or name in self.allow)

    def narrowed(self, other: "ToolPolicy") -> "ToolPolicy":
        allow = other.allow if self.allow is None else self.allow
        if self.allow is not None and other.allow is not None:
            allow = self.allow & other.allow
        return ToolPolicy(allow, self.deny | other.deny)


@dataclass
class ProcessResult:
    path: Path
    exit_code: int
    timed_out: bool
    duration_seconds: float


def run_process(
    command: str | list[str],
    cwd: Path,
    artifacts: ArtifactStore,
    context: ExecutionContext,
    shell: bool = False,
    *,
    tool: str = "exec",
) -> ProcessResult:
    """Spool complete output to disk; enforce the shared deadline and cancellation."""
    context.remaining()
    path = artifacts.new_path(".log")
    started = time.monotonic()
    with path.open("xb") as output:
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                shell=shell,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env=command_environment(),
            )
        except BaseException:
            path.unlink(missing_ok=True)
            raise

        def kill() -> None:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()

        timed_out = False
        next_progress = started + 1
        try:
            while True:
                try:
                    remaining = context.remaining()
                except TimeoutError:
                    timed_out = True
                    kill()
                    break
                try:
                    process.wait(timeout=min(remaining, 0.1))
                    kill()
                    break
                except subprocess.TimeoutExpired:
                    now = time.monotonic()
                    if context.on_progress is not None and now >= next_progress:
                        size, tail = 0, ""
                        with suppress(OSError):
                            size = os.fstat(output.fileno()).st_size
                            with path.open("rb") as preview:
                                preview.seek(max(0, size - 2048))
                                tail = preview.read(2048).decode("utf-8", errors="replace")
                        context.report_progress(
                            tool=tool,
                            elapsed_seconds=now - started,
                            received_bytes=size,
                            output="\n".join(tail.splitlines()[-3:]),
                            artifact_path=str(path),
                        )
                        next_progress = now + 1
        except BaseException:
            kill()
            raise
    return ProcessResult(path, process.returncode, timed_out, round(time.monotonic() - started, 3))


def _preview(path: Path) -> tuple[str, bool]:
    with path.open(encoding="utf-8", errors="replace", newline="") as stream:
        content = stream.read(PAGE_CHARS + 1)
    if len(content) <= PAGE_CHARS:
        return content, False
    with path.open("rb") as stream:
        stream.seek(-min(path.stat().st_size, PAGE_CHARS // 2), os.SEEK_END)
        tail = stream.read().decode("utf-8", errors="replace")
    return content[
        : PAGE_CHARS // 2
    ] + "\n[... full output saved in artifact_path ...]\n" + tail, True


def _null_records(path: Path, context: ExecutionContext) -> Iterator[str]:
    pending = b""
    with path.open("rb") as stream:
        while chunk := stream.read(65536):
            context.remaining()
            parts = (pending + chunk).split(b"\0")
            pending = parts.pop()
            for part in parts:
                yield os.fsdecode(part)
    if pending:
        raise ValueError("Incomplete filename record from ripgrep")


def _numbered(content: str, first_line: int) -> str:
    # Lines end at "\n" only, as read's offsets count them.
    if not content:
        return content
    lines = content.split("\n")
    end = "\n" if lines[-1] == "" else ""
    if end:
        lines.pop()
    return "\n".join(f"{number:>6}\t{line}" for number, line in enumerate(lines, first_line)) + end


def _line_of(text: str, index: int) -> int:
    return text.count("\n", 0, index) + 1


def _edit_mismatch(original: str, old_text: str, number: int) -> str:
    """Why edits[number].oldText does not match exactly once, and where to look instead."""
    message = f"Each oldText must match exactly once; edits[{number}].oldText "
    count = original.count(old_text)
    if count > 1:
        starts, index = [], original.find(old_text)
        while index != -1 and len(starts) < 5:
            starts.append(_line_of(original, index))
            index = original.find(old_text, index + 1)
        return (
            message + f"matches {count} places (lines {', '.join(map(str, starts))}"
            f"{', ...' if count > len(starts) else ''}). Include nearby unique lines."
        )
    message += "matches nothing in the current file."
    wanted = [line.strip() for line in old_text.split("\n") if line.strip()]
    if not wanted:
        return message + " Read the file and copy the exact text."
    lines = original.split("\n")
    stripped = [line.strip() for line in lines]
    # A block equal apart from indentation or trailing spaces is the usual cause.
    for start in range(len(lines)):
        if stripped[start] != wanted[0]:
            continue
        found, position = 0, start
        while position < len(lines) and found < len(wanted):
            if stripped[position] == wanted[found]:
                found += 1
            elif stripped[position]:
                break
            position += 1
        if found == len(wanted):
            return (
                message + f" Ignoring whitespace it matches lines {start + 1}-{position}: copy "
                "their exact indentation, blank lines and trailing spaces (read with "
                "line_numbers to see them; the prefixes are not file content)."
            )
    partial = next(
        (
            (index, line)
            for index, line in enumerate(wanted)
            if stripped.count(line) == 1 and len(line) > 3
        ),
        None,
    )
    if partial is not None:
        index, line = partial
        return (
            message + f" Its line {index + 1} ({line[:80]!r}) is at line "
            f"{stripped.index(line) + 1}, but the lines around it differ; read that range again."
        )
    return message + " None of its lines is unique in the file; read the current text first."


def _rg_text(value: dict[str, str]) -> str:
    if "text" in value:
        return value["text"]
    return base64.b64decode(value["bytes"]).decode("utf-8", errors="replace")


class WorkerStatus:
    def __init__(self, heartbeat_seconds: float = 10) -> None:
        self.lock = threading.RLock()
        self.running = 0
        self.started_total = 0
        self.finished_total = 0
        self.limit = MAX_SUBAGENTS
        self.path: Path | None = None
        self.generation = uuid4().hex
        self.owner = uuid4().hex
        self.last_error: str | None = None
        self.heartbeat_seconds = heartbeat_seconds
        self.stop = threading.Event()

    def _write(self) -> None:
        if self.path is None:
            return
        try:
            atomic_write_text(
                self.path,
                json.dumps(
                    {
                        "running": self.running,
                        "max": self.limit,
                        "updated_at": time.time(),
                        "started_total": self.started_total,
                        "finished_total": self.finished_total,
                        "generation": self.generation,
                        "pid": os.getpid(),
                        "owner": self.owner,
                    }
                ),
            )
            self.last_error = None
        except OSError as exc:
            self.last_error = f"子代理状态文件写入失败：{exc}"

    def status_path(self, base: Path) -> Path:
        return base.with_name(f"{base.stem}.{os.getpid()}.{self.owner}{base.suffix}")

    def configure(self, path: Path | None, limit: int) -> None:
        path = self.status_path(path) if path is not None else None
        with self.lock:
            if path == self.path and limit == self.limit and not self.stop.is_set():
                return
            self.stop.set()
            if self.path is not None and self.path != path:
                try:
                    if self.path.exists():
                        value = json.loads(self.path.read_text(encoding="utf-8"))
                        if (
                            isinstance(value, dict)
                            and value.get("owner") == self.owner
                            and value.get("pid") == os.getpid()
                        ):
                            self.path.unlink(missing_ok=True)
                except (OSError, ValueError) as exc:
                    self.last_error = f"子代理状态文件清理失败：{exc}"
            self.path = path
            self.limit = limit
            self.generation = uuid4().hex
            self.stop = threading.Event()
            if path is not None:
                self._write()
                try:
                    thread = threading.Thread(
                        target=self._heartbeat,
                        args=(self.stop,),
                        name="worker-status",
                        daemon=True,
                    )
                    thread.start()
                except RuntimeError as exc:
                    self.stop.set()
                    self.last_error = f"子代理状态心跳启动失败：{exc}"

    def _heartbeat(self, stop: threading.Event) -> None:
        while not stop.wait(self.heartbeat_seconds):
            with self.lock:
                if stop is not self.stop or stop.is_set():
                    return
                self._write()

    def started(self) -> None:
        with self.lock:
            self.running += 1
            self.started_total += 1
            self._write()

    def finished(self) -> None:
        with self.lock:
            self.running -= 1
            self.finished_total += 1
            self._write()

    def close(self) -> None:
        self.configure(None, self.limit)


class CodingTools:
    def __init__(
        self,
        workspace: Path,
        on_tool: Callable[[str, dict[str, Any]], None] | None = None,
        on_task_update: Callable[[dict[str, Any]], None] | None = None,
        on_execution: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.workspace = workspace.resolve()
        self.on_tool = on_tool
        self.on_task_update = on_task_update
        self.on_execution = on_execution
        self.task_title = ""
        self.task_plan: dict[str, Any] | None = None
        self.artifacts = ArtifactStore(self.workspace)
        self.processes = ProcessMonitor(self.artifacts)
        self._file_versions = threading.local()
        # Definitions, validation, dispatch and worker access all use these same specs.
        specs = [
            ToolSpec(
                "read",
                "Read a local UTF-8 text file with exact line/column ranges; URLs are not file paths. Pass the returned next_cursor object with the same path, or merge its offset/column fields, to continue without gaps. Set line_numbers=true when you need to cite line numbers.",
                {
                    "path": PATH,
                    "offset": OFFSET,
                    "limit": {**LIMIT, "default": 400},
                    "column": {
                        "type": "integer",
                        "minimum": 0,
                        "default": 0,
                        "description": "0-based character position within the starting line; used for very long lines.",
                    },
                    "line_numbers": {
                        "type": "boolean",
                        "default": False,
                        "description": "Prefix each line with its number and a tab, like cat -n. The prefixes are not file content: never copy them into edit oldText.",
                    },
                    "next_cursor": READ_CURSOR,
                },
                ("path",),
                self.read,
                True,
            ),
            ToolSpec(
                "write",
                "Create or overwrite a UTF-8 file, creating parent directories. Read existing files before overwriting.",
                {
                    "path": PATH,
                    "content": TEXT,
                },
                ("path", "content"),
                self.write,
            ),
            ToolSpec(
                "edit",
                "Apply exact, unique, non-overlapping replacements against the original UTF-8 file. All replacements must match before any write.",
                {
                    "path": PATH,
                    "edits": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "oldText": {"type": "string", "minLength": 1},
                                "newText": TEXT,
                            },
                            "required": ["oldText", "newText"],
                            "additionalProperties": False,
                        },
                    },
                },
                ("path", "edits"),
                self.edit,
            ),
            ToolSpec(
                "grep",
                "Search text with ripgrep. Returns complete match/context records with line numbers and paths (relative to the workspace when inside it). Repeat the same query with the returned next_cursor object to continue; results are ordered by path. Use literal=true for fixed strings. For minified files, avoid enormous regex repetition bounds; search a short literal, then read the matching range. An explicit positive glob can override ignore files.",
                {
                    "pattern": TEXT,
                    "path": PATH,
                    "glob": TEXT,
                    "ignoreCase": {"type": "boolean", "default": False},
                    "literal": {"type": "boolean", "default": False},
                    "context": {"type": "integer", "minimum": 0, "maximum": 100},
                    "offset": OFFSET,
                    "limit": {**LIMIT, "default": 100},
                    "next_cursor": CURSOR,
                },
                ("pattern",),
                self.grep,
                True,
            ),
            ToolSpec(
                "find",
                "Find files by ripgrep glob; explicit positive globs can override ignore files. Returns complete relative paths in sorted order. Repeat the same query with next_cursor to continue.",
                {
                    "pattern": TEXT,
                    "path": PATH,
                    "offset": OFFSET,
                    "limit": {**LIMIT, "default": 1000},
                    "next_cursor": CURSOR,
                },
                ("pattern",),
                self.find,
                True,
            ),
            ToolSpec(
                "ls",
                "List entries including dotfiles, sorted by name; directories end in /. Pass next_cursor with the same path to continue.",
                {
                    "path": PATH,
                    "offset": OFFSET,
                    "limit": {**LIMIT, "default": 500},
                    "next_cursor": CURSOR,
                },
                (),
                self.ls,
                True,
            ),
            ToolSpec(
                "exec",
                "Execute a shell command. Nonzero exit codes are errors. For human interaction, long-running tasks or long-lived programs, set background=true: returns session_id immediately, keeps the process alive across paused turns, and uses timeout as its independent lifetime. Use monitor to wait for output/exit, send stdin or stop it. Complete stdout/stderr is retained at artifact_path.",
                {
                    "command": {"type": "string", "minLength": 1},
                    "cwd": PATH,
                    "timeout": {
                        **TIMEOUT,
                        "default": 120,
                        "description": "Foreground commands are bounded by the task deadline. For background=true, this is an independent process lifetime in seconds.",
                    },
                    "background": {"type": "boolean", "default": False},
                },
                ("command",),
                self.run_command,
            ),
            ToolSpec(
                "monitor",
                "Wait for user actions, background tasks or external state. For exec background sessions use list, status, wait, input, stop. wait returns on literal wait_for output or exit; an empty wait_for waits for exit. Reuse next_offset for new output. For files use watch with path and condition=exists/contains/changed; contains needs wait_for, changed returns baseline to reuse across calls. Each wait returns after wait_seconds without killing any process. event=waiting is not completion; keep waiting or check actual state. Sessions survive turn cancellation but end on their own timeout, environment switch or application exit. Main agent only; requires exec permission.",
                {
                    "action": {
                        "type": "string",
                        "enum": ["list", "status", "wait", "input", "stop", "watch"],
                    },
                    "session_id": PATH,
                    "offset": {"type": "integer", "minimum": 0, "default": 0},
                    "wait_seconds": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 60,
                        "default": 30,
                    },
                    "wait_for": {**TEXT, "maxLength": 1000},
                    "input": {**TEXT, "maxLength": 1000},
                    "path": PATH,
                    "condition": {
                        "type": "string",
                        "enum": ["exists", "contains", "changed"],
                        "default": "exists",
                    },
                    "baseline": {**TEXT, "maxLength": 512},
                },
                ("action",),
                self.monitor,
            ),
            ToolSpec(
                "update_plan",
                "Set the task title and replace the visible Todo list for multi-step work. Keep steps current; mark completed only after verification. At most one step may be in_progress. Do not create a plan for simple questions.",
                {
                    "title": {"type": "string", "minLength": 1, "maxLength": 120},
                    "steps": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 20,
                        "items": {
                            "type": "object",
                            "properties": {
                                "step": {"type": "string", "minLength": 1, "maxLength": 240},
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                },
                            },
                            "required": ["step", "status"],
                            "additionalProperties": False,
                        },
                    },
                },
                ("title", "steps"),
                self.update_plan,
            ),
        ]
        self.registry = {spec.name: spec for spec in specs}
        self.policy = ToolPolicy()
        # Extra tools an environment grants to subagents (empty means read-only).
        self.worker_allow: frozenset[str] = frozenset()
        self.worker_roles: dict[str, str] = {}
        self.worker_skills = ""
        self.worker_instructions = ""
        self.cache_scope = "default"
        self._global_worker_slots = None
        self._global_worker_limit = None
        self.worker_status = WorkerStatus()
        self.role_loader = None
        # Team members get private session state hidden and, optionally, write scopes.
        self.protected_paths: tuple[Path, ...] = ()
        self.write_roots: tuple[Path, ...] | None = None
        # Shared by a team's agents: stale edits merge with teammates' writes.
        self.edit_ledger: EditLedger | None = None

    def _registry_for(
        self,
        owner: "CodingTools",
        registry: dict[str, ToolSpec] | None = None,
    ) -> dict[str, ToolSpec]:
        return {
            name: replace(spec, handler=getattr(owner, spec.handler.__name__))
            if getattr(spec.handler, "__self__", None) is self
            else spec
            for name, spec in (self.registry if registry is None else registry).items()
        }

    def fork(self, *, registry: dict[str, ToolSpec] | None = None) -> "CodingTools":
        snapshot = CodingTools(self.workspace, self.on_tool, self.on_task_update)
        snapshot.on_execution = self.on_execution
        snapshot.artifacts = self.artifacts
        snapshot.processes = self.processes
        snapshot._file_versions = self._file_versions
        snapshot.worker_status = self.worker_status
        snapshot.edit_ledger = self.edit_ledger
        snapshot.apply_configuration(self)
        if registry is not None:
            snapshot.registry = self._registry_for(snapshot, registry)
        return snapshot

    def apply_configuration(self, other: "CodingTools") -> None:
        self.registry = other._registry_for(self)
        self.policy = other.policy
        self.task_title = other.task_title
        self.task_plan = deepcopy(other.task_plan)
        self.worker_allow = other.worker_allow
        self.worker_roles = dict(other.worker_roles)
        self.worker_skills = other.worker_skills
        self.worker_instructions = other.worker_instructions
        self.cache_scope = other.cache_scope
        self.role_loader = other.role_loader
        self._global_worker_slots = other._global_worker_slots
        self._global_worker_limit = other._global_worker_limit
        self.protected_paths = other.protected_paths
        self.write_roots = other.write_roots

    @property
    def file_versions(self) -> dict[Path, str | None]:
        attribute = (
            "worker_versions" if getattr(self._file_versions, "worker", False) else "versions"
        )
        if not hasattr(self._file_versions, attribute):
            setattr(self._file_versions, attribute, {})
        return getattr(self._file_versions, attribute)

    @property
    def worker_changes(self) -> FileTransaction | None:
        return getattr(self._file_versions, "transaction", None)

    @contextmanager
    def worker_session(self, context: ExecutionContext):
        previous = self.worker_changes
        previous_versions = getattr(self._file_versions, "worker_versions", None)
        previous_worker = getattr(self._file_versions, "worker", False)
        writable = any(
            name in self.worker_allow and self.policy.permits(name) for name in ("write", "edit")
        )
        ledger = self.edit_ledger
        changes = (
            FileTransaction(self.artifacts, ledger.record_write if ledger else None)
            if writable
            else None
        )
        self._file_versions.transaction = changes
        self._file_versions.worker_versions = {}
        self._file_versions.worker = True
        try:
            context.remaining()
            yield changes
        finally:
            self._file_versions.transaction = previous
            self._file_versions.worker = previous_worker
            if previous_versions is None:
                del self._file_versions.worker_versions
            else:
                self._file_versions.worker_versions = previous_versions

    @property
    def definitions(self) -> list[ToolDefinition]:
        return [
            spec.definition() for spec in self.registry.values() if self.tool_available(spec.name)
        ]

    @property
    def worker_definitions(self) -> list[ToolDefinition]:
        return [
            spec.definition()
            for spec in self.registry.values()
            if self.tool_available(spec.name, worker=True)
        ]

    def tool_available(self, name: str, *, worker: bool = False) -> bool:
        spec = self.registry.get(name)
        return bool(
            spec is not None
            and self.policy.permits(name)
            and (name != "monitor" or self.policy.permits("exec"))
            and (
                not worker
                or (name not in WORKER_EXCLUDED and (spec.read_only or name in self.worker_allow))
            )
        )

    def detach(self, name: str) -> None:
        if name not in {"websearch", "mcp", "delegate", "ask_user", "team"}:
            raise ValueError(f"Cannot detach built-in tool {name!r}")
        self.registry.pop(name, None)

    def reset_task(self, title: str = "") -> None:
        self.task_title = title
        self.task_plan = None
        if "delegate" in self.registry:
            self.registry["delegate"].handler.__self__.agent_sequence = 0

    def update_plan(
        self, title: str, steps: list[dict[str, str]], *, _context: ExecutionContext
    ) -> ToolResult:
        _context.remaining()
        if not title.strip() or any(not item["step"].strip() for item in steps):
            return ToolResult.failure("invalid_arguments", "Title and steps must not be blank")
        if len(title) > 120 or any(len(item["step"]) > 240 for item in steps):
            return ToolResult.failure("invalid_arguments", "Title or step exceeds the length limit")
        if sum(item["status"] == "in_progress" for item in steps) > 1:
            return ToolResult.failure("invalid_arguments", "Only one step may be in_progress")
        self.task_title = title.strip()
        self.task_plan = {"title": self.task_title, "steps": [dict(item) for item in steps]}
        if self.on_task_update:
            display_callback(self.on_task_update, name="plan")({"phase": "plan", **self.task_plan})
        return ToolResult(data=self.task_plan)

    def attach_websearch(self, client: WebSearchClient | ChatGPTWebSearchClient) -> None:
        self.registry["websearch"] = ToolSpec(
            "websearch",
            client.description,
            {
                "query": {"type": "string", "minLength": 1},
                "allowed_domains": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "description": "Only search these domains, e.g. python.org. Do not combine with blocked_domains.",
                },
                "blocked_domains": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "description": "Exclude these domains. Do not combine with allowed_domains.",
                },
                "timeout": {**TIMEOUT, "default": client.default_timeout},
            },
            ("query",),
            client.search,
            True,
        )

    def attach_mcp(self, manager: McpManager) -> None:
        self.registry["mcp"] = ToolSpec(
            "mcp",
            "Discover and call tools from configured MCP servers. The application starts connecting enabled servers in the background at startup. Use status to inspect servers, search to find a tool, describe to inspect its schema, then call it. Search query '*' lists every tool; other queries rank tools matching any of their words, and a search without matches lists the available tool paths. Search/describe/call wait for a server that is still connecting and reconnect automatically if needed; do not ask the user to connect manually.",
            {
                "action": {
                    "type": "string",
                    "enum": ["status", "connect", "search", "describe", "call"],
                },
                "server": {"type": "string", "minLength": 1},
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Words matched against tool names and descriptions, or '*' for every tool.",
                },
                "tool": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Qualified server/tool path returned by search, or an unambiguous tool name.",
                },
                "arguments": {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": True,
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 12},
            },
            ("action",),
            manager.execute,
        )

    def attach_user_input(
        self,
        handler: Callable[[str, list[dict[str, str]]], dict[str, Any]],
    ) -> None:
        def ask_user(
            question: str,
            options: list[dict[str, str]],
            *,
            _context: ExecutionContext,
        ) -> ToolResult:
            _context.remaining()
            labels = [option["label"].strip().casefold() for option in options]
            if len(labels) != len(set(labels)):
                raise ValueError("ask_user option labels must be unique")
            reserved = {"其他", "其它", "自定义", "other", "custom"}
            reserved_prefixes = ("其他（", "其它（", "自定义", "other ", "custom ")
            if any(label in reserved or label.startswith(reserved_prefixes) for label in labels):
                raise ValueError("Do not provide a custom option; the UI adds it automatically")
            selection = handler(question, options)
            _context.remaining()
            if not isinstance(selection, dict) or not isinstance(selection.get("answer"), str):
                raise TypeError("User input handler returned an invalid selection")
            return ToolResult(data={"question": question, **selection})

        self.registry["ask_user"] = ToolSpec(
            "ask_user",
            "Ask the user one necessary multiple-choice question. Provide 2-5 mutually exclusive options, each with a concise label and a concrete explanation of its impact or tradeoff. The UI automatically adds a final custom-answer option, so do not add Other yourself. Put the recommended option first when there is a clear recommendation and explain why. Do not use this tool when the request already provides enough information to proceed safely.",
            {
                "question": {"type": "string", "minLength": 1},
                "options": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 5,
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string", "minLength": 1},
                            "description": {"type": "string", "minLength": 1},
                        },
                        "required": ["label", "description"],
                        "additionalProperties": False,
                    },
                },
            },
            ("question", "options"),
            ask_user,
        )

    def environment_instructions(self) -> str:
        instructions = (
            "Execution environment: /bin/sh. Use exec.timeout, keep scratch files in the workspace, retain missing-"
            "dependency results, and test expensive work at small scale first. Calls are unlimited but deadlines apply."
        )
        if not self.policy.permits("exec"):
            instructions = "Use only the listed tools; never simulate or bypass disabled tools. Calls are unlimited but deadlines apply."
        if self.tool_available("monitor"):
            instructions += (
                " For human interaction or long-running work, start the program with exec background=true "
                "and a sufficient independent timeout, tell the user what to do, then use monitor "
                "wait for a real output marker or exit and continue. A wait returning event=waiting "
                "does not kill the process. Use monitor watch to wait for a file to exist, contain a "
                "real completion marker, or change (reuse baseline). Do not finish the task merely "
                "to wait for a user action or task completion. "
                "After a paused turn, inspect the existing session_id instead of restarting."
            )
        if "ask_user" in self.registry and self.policy.permits("ask_user"):
            instructions += " Use ask_user only when a necessary choice cannot be inferred safely."
        return instructions

    def attach_reasoning(self, client: CompletionClient, max_workers: int = MAX_SUBAGENTS) -> None:
        workers = max(1, min(MAX_SUBAGENTS, max_workers))
        if self._global_worker_slots is None:
            self._global_worker_slots = threading.BoundedSemaphore(workers)
            self._global_worker_limit = workers
        snapshot = self.fork()
        worker_client = snapshot_client(client)
        reasoning = ReasoningTools(
            worker_client,
            snapshot.execute_workspace,
            self.worker_definitions,
            workers,
            workspace=self.workspace,
            on_progress=self.on_task_update,
            global_slots=self._global_worker_slots,
            worker_roles=self.worker_roles,
            worker_skills=self.worker_skills,
            worker_session=snapshot.worker_session,
            role_loader=self.role_loader,
            on_worker_start=self.worker_status.started,
            on_worker_finish=self.worker_status.finished,
        )
        reasoning.runner.environment_instructions = self.worker_instructions
        reasoning.runner.cache_scope = self.cache_scope
        self.registry["delegate"] = ToolSpec(
            "delegate",
            "Delegate bounded tasks when independent investigation, parallel work, focused verification, or other useful division of work improves efficiency or quality; task complexity is not the criterion. There is no tool-call count limit. The task count limit is capacity, not a target: choose the smallest useful number of workers, avoid delegation with no clear benefit, and sequence dependent work. Use mode=reason for reasoning/recall without tools; mode=inspect to use the granted tools. Do not repeat an assignment without new evidence or a distinct question. Workers are read-only unless the active environment grants extra worker tools; they can never delegate. Writable workers use private snapshots for read/write/edit, then successful drafts are merged in task order. Non-overlapping changes can merge; conflicts preserve patches and never overwrite existing changes. Check file_changes before claiming edits are applied, and validate the integrated result. grep/find/ls and external tools still see the shared workspace, not drafts. exec is blocked while a worker has staged changes; run tests after integration. Some environments provide worker roles: pass role to load that role's system instructions. The parent synthesizes their evidence; status=ok only means a report was returned.",
            {
                "tasks": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": workers,
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string", "minLength": 1, "maxLength": 120},
                            "task": {"type": "string", "minLength": 1},
                            "context": TEXT,
                            "expected_output": {"type": "string", "minLength": 1},
                            "role": {
                                "type": "string",
                                "minLength": 1,
                                "description": "Optional worker role defined by the active environment (loads that role's instructions as the subagent system prompt).",
                            },
                            "mode": {
                                "type": "string",
                                "enum": ["auto", "reason", "inspect"],
                                "default": "auto",
                            },
                        },
                        "required": ["task", "expected_output"],
                        "additionalProperties": False,
                    },
                },
                "context": TEXT,
                "timeout": {**TIMEOUT, "default": 300},
            },
            ("tasks",),
            reasoning.delegate,
            result_chars=DELEGATE_RESULT_CHARS,
        )

    def path(self, value: str = ".") -> Path:
        path = Path(value).expanduser()
        resolved = path.resolve() if path.is_absolute() else (self.workspace / path).resolve()
        if self._protected(resolved):
            raise PermissionError(f"{resolved} holds private session state and is not accessible")
        return resolved

    def _protected(self, path: Path) -> bool:
        for root in self.protected_paths:
            if path == root or path.is_relative_to(root):
                return True
            # atomic_write_text stages content in ".<name>.<id>.tmp" next to the target.
            if (
                path.parent == root.parent
                and path.name.startswith(f".{root.name}.")
                and path.name.endswith(".tmp")
            ):
                return True
        return False

    def _writable(self, path: Path) -> None:
        if self.write_roots is not None and not any(
            path == root or path.is_relative_to(root) for root in self.write_roots
        ):
            allowed = ", ".join(str(root) for root in self.write_roots) or "none"
            raise PermissionError(f"Writing {path} is outside this agent's write scope ({allowed})")

    def _workspace_relative(self, value: str) -> str:
        # Repeated absolute paths would take much of a search page.
        path = Path(value)
        return (
            path.relative_to(self.workspace).as_posix()
            if path.is_relative_to(self.workspace)
            else value
        )

    def _protected_globs(self, root: Path) -> list[str]:
        globs = []
        for protected in self.protected_paths:
            if protected.is_relative_to(root) and protected != root:
                relative = protected.relative_to(root)
                temporary = (relative.parent / f".{relative.name}.*.tmp").as_posix()
                globs += ["--glob", f"!{relative.as_posix()}", "--glob", f"!{relative.as_posix()}/**"]
                globs += ["--glob", f"!{temporary}"]
        return globs

    def execute(
        self,
        tool_call: ToolCall,
        max_timeout: float | None = None,
        *,
        deadline: float | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> str:
        context = ExecutionContext(
            deadline
            if deadline is not None
            else time.monotonic() + (max_timeout if max_timeout is not None else 3600),
            on_progress=on_progress,
        )
        return self._execute(tool_call, context, worker=False)

    def execute_in(self, tool_call: ToolCall, context: ExecutionContext) -> str:
        """Run as a full agent under the caller's deadline and cancellation (team members)."""
        return self._execute(tool_call, context, worker=False)

    def execute_workspace(self, tool_call: ToolCall, context: ExecutionContext) -> str:
        previous = getattr(self._file_versions, "worker", False)
        self._file_versions.worker = True
        try:
            return self._execute(tool_call, context, worker=True)
        finally:
            self._file_versions.worker = previous

    def _execute(self, tool_call: ToolCall, context: ExecutionContext, worker: bool) -> str:
        call_id = tool_call.get("id", "")
        trace = context.trace
        if trace.tool_call_id != call_id or trace.execution_id is None:
            trace = trace.tool(call_id)
        context = replace(context, trace=trace)
        name = tool_call.get("function", {}).get("name")
        with trace_scope(trace, context.event_sink):
            started = time.monotonic()
            self._execution_update("started", tool_call, context)
            emit_event("tool_started", tool=name, worker=worker)
            try:
                output = self._execute_registered(tool_call, context, worker)
            except BaseException as exc:
                self._execution_update(
                    "interrupted", tool_call, context, duration_seconds=time.monotonic() - started
                )
                emit_event("tool_interrupted", tool=name, error_type=type(exc).__name__)
                raise
            result = json.loads(output)
            self._execution_update(
                "completed",
                tool_call,
                context,
                result=output,
                duration_seconds=time.monotonic() - started,
            )
            emit_event("tool_completed", tool=name, status=result["status"], worker=worker)
            return output

    def _execution_update(self, state, call, context, **details) -> None:
        # Full inputs/results belong to the session inspector, not telemetry events.
        if self.on_execution is not None:
            with suppress(Exception):
                self.on_execution(
                    {
                        "state": state,
                        "call": call,
                        "trace": context.trace.as_dict(),
                        "timestamp": time.time(),
                        **details,
                    }
                )

    def _execute_registered(
        self, tool_call: ToolCall, context: ExecutionContext, worker: bool
    ) -> str:
        spec = None
        try:
            context.remaining()
            function = tool_call["function"]
            name = function["name"]
            spec = self.registry.get(name)
            if not self.tool_available(name, worker=worker):
                return self.artifacts.render(
                    ToolResult.failure(
                        "unavailable_tool",
                        f"Tool {name!r} is unavailable{' to subagents' if worker else ''}",
                    )
                )
            if worker and name == "exec" and self.worker_changes and self.worker_changes.changes:
                return self.artifacts.render(
                    ToolResult.failure(
                        "staged_changes_pending",
                        "File changes are still private drafts; exec would test the unchanged shared "
                        "workspace. Return your report so the parent can merge and test the result.",
                    )
                )
            try:
                arguments = json.loads(function.get("arguments") or "{}")
                if isinstance(arguments, dict) and name in {"read", "grep", "find", "ls"}:
                    cursor = arguments.get("next_cursor")
                    # Some providers double-encode cursor objects. Decode only
                    # this documented field; never guess arbitrary arguments.
                    if isinstance(cursor, str):
                        cursor = arguments["next_cursor"] = json.loads(cursor)
                    if isinstance(cursor, dict):
                        for key in ("offset", "column"):
                            if key in arguments and key in cursor and arguments[key] != cursor[key]:
                                raise ValueError(
                                    f"arguments.{key} conflicts with next_cursor.{key}"
                                )
                validate(arguments, spec.schema)
                if name == "read" and re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", arguments["path"]):
                    raise ValueError(
                        "read.path must be a local file, not a URL; use a web or browser tool for URLs"
                    )
            except (ValueError, TypeError) as exc:
                return self.artifacts.render(
                    ToolResult.failure(
                        "invalid_arguments",
                        str(exc),
                        data={
                            "tool": name,
                            "guidance": "Correct the arguments using this schema and send valid JSON. Do not repeat the unchanged call.",
                            "input_schema": spec.schema,
                        },
                    )
                )
            if worker and name == "exec" and arguments.get("background"):
                return self.artifacts.render(
                    ToolResult.failure(
                        "unavailable_tool",
                        "Background command sessions are only available to the main agent",
                    )
                )
            if self.on_tool and not worker:
                display_callback(self.on_tool, name="tool_start")(name, arguments)
            result = spec.handler(**arguments, _context=context)
        except FileChangedError as exc:
            result = ToolResult.failure(
                "file_changed",
                str(exc),
                data={
                    "guidance": "Read the current file again, then rebase the intended edit on its exact contents."
                },
            )
        except TimeoutError as exc:
            result = ToolResult.failure("deadline_exceeded", str(exc), status="timeout")
        except Exception as exc:
            result = ToolResult.failure(type(exc).__name__, str(exc))
        return self.artifacts.render(
            result, spec.result_chars if spec is not None else MAX_RESULT_CHARS
        )

    def read(
        self,
        path: str,
        offset: int = 1,
        limit: int = 400,
        column: int = 0,
        line_numbers: bool = False,
        next_cursor: dict[str, int] | None = None,
        *,
        _context: ExecutionContext,
    ) -> ToolResult:
        if next_cursor is not None:
            offset = next_cursor["offset"]
            column = next_cursor.get("column", 0)
        file_path = self.path(path)
        changes = self.worker_changes
        try:
            if changes is not None:
                content = changes.read(file_path, _context.remaining)
                version = text_digest(content)
                source = io.StringIO(content, newline="")
            else:
                if not file_path.is_file():
                    raise FileNotFoundError(f"Not a regular file: {file_path}")
                whole = self._mergeable_text(file_path, _context.remaining)
                if whole is not None:
                    # One read serves the page, its version and the team's merge base.
                    version = self.edit_ledger.remember(whole)
                    source = io.StringIO(whole, newline="")
                else:
                    version = file_digest(file_path, _context.remaining)
                    source = file_path.open(encoding="utf-8", newline="")
            with source as stream:
                current_line = 1
                while current_line < offset:
                    _context.remaining()
                    chunk = stream.readline(PAGE_CHARS)
                    if not chunk:
                        break
                    if chunk.endswith("\n"):
                        current_line += 1
                if current_line < offset:
                    raise ValueError("offset is beyond the end of the file")
                left = column
                while left:
                    _context.remaining()
                    chunk = stream.readline(min(left, PAGE_CHARS))
                    if not chunk or chunk.endswith("\n"):
                        raise ValueError("column is beyond the end of the line")
                    left -= len(chunk)
                content, next_line, next_column, eof = self._read_page(
                    stream, offset, column, limit, _context
                )
        except UnicodeDecodeError:
            # PDFs, Word files and images are common research inputs; say what to do.
            return ToolResult.failure(
                "not_text",
                f"{file_path} is not UTF-8 text (for example a PDF, Word or image file). "
                "Extract its text first with a command such as `pdftotext FILE OUT.txt`, "
                "or ask a teammate who can run commands to do it, then read the text file.",
            )
        if changes is None and file_digest(file_path, _context.remaining) != version:
            raise FileChangedError(f"File changed while reading: {file_path}. Read it again.")
        self.file_versions[file_path] = version
        return ToolResult(
            data={
                "path": str(file_path),
                "content": _numbered(content, offset) if line_numbers else content,
                "start_line": offset,
                "start_column": column,
                "end_line": next_line,
                "end_column": next_column,
                "eof": eof,
                "worker_snapshot": changes is not None,
            },
            truncated=not eof,
            next_cursor=None if eof else {"offset": next_line, "column": next_column},
        )

    @staticmethod
    def _read_page(
        stream: TextIO, line: int, column: int, limit: int, context: ExecutionContext
    ) -> tuple[str, int, int, bool]:
        chunks = []
        size = 0
        completed = 0
        while size < READ_PAGE_CHARS and completed < limit:
            context.remaining()
            chunk = stream.readline(READ_PAGE_CHARS - size)
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if chunk.endswith("\n"):
                line += 1
                column = 0
                completed += 1
            else:
                column += len(chunk)
        return "".join(chunks), line, column, stream.read(1) == ""

    def write(self, path: str, content: str, *, _context: ExecutionContext) -> ToolResult:
        file_path = self.path(path)
        self._writable(file_path)
        _context.remaining()
        if self.worker_changes is not None:
            if (
                file_path not in self.file_versions
                and self.worker_changes.capture(file_path, _context.remaining).base is not None
            ):
                return ToolResult.failure(
                    "read_required", "Read the existing file before staging a full-file overwrite."
                )
            self.worker_changes.write(file_path, content, _context.remaining)
            self.file_versions[file_path] = text_digest(content)
            return ToolResult(
                data={
                    "path": str(file_path),
                    "bytes_written": len(content.encode("utf-8")),
                    "staged": True,
                    "commit_required": True,
                }
            )
        version = (
            self.file_versions[file_path]
            if file_path in self.file_versions
            else file_digest(file_path, _context.remaining)
        )
        if file_path not in self.file_versions and version is not None:
            return ToolResult.failure(
                "read_required", "Read the existing file before a full-file overwrite."
            )
        merged = False
        with self._merge_lock(file_path, _context.remaining):
            current = self._stale_content(file_path, version, _context.remaining)
            if current is not None:
                content = self.edit_ledger.merge(file_path, version, current, content)
                version, merged = text_digest(current), True
            atomic_write_text(file_path, content, expected=version, check=_context.remaining)
            # Inside the lock: a teammate merging next must already see this as agent-written.
            self._wrote(file_path, content)
        return ToolResult(
            data={
                "path": str(file_path),
                "bytes_written": len(content.encode("utf-8")),
                **({"merged": MERGED_NOTE} if merged else {}),
            }
        )

    def edit(
        self, path: str, edits: list[dict[str, str]], *, _context: ExecutionContext
    ) -> ToolResult:
        file_path = self.path(path)
        self._writable(file_path)
        changes = self.worker_changes
        merged = False
        with self._merge_lock(file_path, _context.remaining) if changes is None else nullcontext():
            if changes is not None:
                original = changes.read(file_path, _context.remaining)
                version = text_digest(original)
            else:
                version = file_digest(file_path, _context.remaining)
                with file_path.open(encoding="utf-8", newline="") as stream:
                    original = stream.read()
            seen = self.file_versions.get(file_path, version)
            base = original
            if seen != version:
                stale = changes is None and self.edit_ledger is not None
                base = self.edit_ledger.content(seen) if stale else None
                if base is None:
                    raise FileChangedError(
                        f"File changed since it was read: {file_path}. Read it again."
                    )
            # Edits apply to the content this agent saw; a teammate's newer write is merged.
            updated = self._apply_edits(base, edits, _context.remaining)
            if base is not original:
                updated = self.edit_ledger.merge(file_path, seen, original, updated)
                merged = True
            _context.remaining()
            if changes is not None:
                changes.write(file_path, updated, _context.remaining)
                self.file_versions[file_path] = text_digest(updated)
            else:
                atomic_write_text(file_path, updated, expected=version, check=_context.remaining)
                self._wrote(file_path, updated)
        return ToolResult(
            data={
                "path": str(file_path),
                "edits_applied": len(edits),
                "bytes_written": len(updated.encode("utf-8")),
                "staged": changes is not None,
                "commit_required": changes is not None,
                **({"merged": MERGED_NOTE} if merged else {}),
            }
        )

    @staticmethod
    def _apply_edits(original: str, edits: list[dict[str, str]], check) -> str:
        replacements = []
        for number, item in enumerate(edits):
            check()
            old_text, new_text = item["oldText"], item["newText"]
            if original.count(old_text) != 1:
                raise ValueError(_edit_mismatch(original, old_text, number))
            start = original.index(old_text)
            replacements.append((start, start + len(old_text), new_text))
        replacements.sort()
        if any(current[0] < previous[1] for previous, current in pairwise(replacements)):
            raise ValueError("Edits must not overlap")
        updated = original
        for start, end, replacement in reversed(replacements):
            updated = updated[:start] + replacement + updated[end:]
        return updated

    def _merge_lock(self, path: Path, check):
        # Read, merge and write under one lock so concurrent teammates serialize.
        return file_lock(path, check) if self.edit_ledger is not None else nullcontext()

    def _stale_content(self, path: Path, seen: str | None, check) -> str | None:
        """The current content when the file changed since this agent's read and can merge.

        None when it did not change (or is gone: the write then reports the conflict).
        Only content a teammate's file tool wrote is mergeable, and the ledger holds it, so
        nothing else is read or decoded here; EditLedger.merge checks the rest.
        """
        ledger = self.edit_ledger
        if ledger is None or seen is None:
            return None
        current = file_digest(path, check)
        if current in {None, seen}:
            return None
        content = ledger.content(current)
        if content is None:
            raise FileChangedError(f"File changed since it was read: {path}. Read it again.")
        return content

    def _mergeable_text(self, path: Path, check) -> str | None:
        """The whole file when a team may merge later edits based on this read, else None.

        Larger or non-UTF-8 files are never merged, so their reads stream as usual.
        """
        if self.edit_ledger is None:
            return None
        check()
        try:
            # UTF-8 needs at most 4 bytes per character.
            if path.stat().st_size > 4 * MAX_AUTO_MERGE_CHARS:
                return None
            with path.open(encoding="utf-8", newline="") as stream:
                content = stream.read(MAX_AUTO_MERGE_CHARS + 1)
        except (OSError, UnicodeDecodeError):
            return None
        return content if len(content) <= MAX_AUTO_MERGE_CHARS else None

    def _wrote(self, path: Path, content: str) -> None:
        self.file_versions[path] = text_digest(content)
        if self.edit_ledger is not None:
            self.edit_ledger.record_write(path, content)

    def run_command(
        self,
        command: str,
        cwd: str = ".",
        timeout: int = 120,
        background: bool = False,
        *,
        _context: ExecutionContext,
    ) -> ToolResult:
        directory = self.path(cwd)
        if background:
            _context.remaining()
            return self.processes.start(command, directory, timeout)
        process = run_process(
            command, directory, self.artifacts, _context.child(timeout), shell=True
        )
        output, truncated = _preview(process.path)
        result = ToolResult(
            data={
                "cwd": str(directory),
                "exit_code": process.exit_code,
                "timed_out": process.timed_out,
                "duration_seconds": process.duration_seconds,
                "output": output,
            },
            truncated=truncated,
            artifact_path=str(process.path),
        )
        missing = sorted(set(re.findall(r"(?:^|[\s:])([\w.+-]+): (?:command )?not found", output)))
        if missing:
            result.data["missing_commands"] = missing
            result.data["guidance"] = (
                "These commands were unavailable. Retain this fact until the environment changes. "
                "Use exec.timeout for time limits; a shell pipeline's zero exit code does not prove all steps succeeded."
            )
        if "Operation not permitted" in output or "Permission denied" in output:
            result.data["permission_diagnostic"] = (
                "Inspect the denied path; use the allowed workspace and its scratch directory."
            )
        if process.timed_out:
            result.status = "timeout"
            result.error = {
                "code": "deadline_exceeded",
                "message": "Command exceeded its timeout or task deadline",
            }
            result.data["guidance"] = (
                "Reduce the input or change algorithm before rerunning; validate a small case and estimate growth."
            )
        elif process.exit_code:
            result.status = "error"
            result.error = {
                "code": "nonzero_exit",
                "message": f"Command exited with code {process.exit_code}",
            }
        return result

    def monitor(
        self,
        action: str,
        session_id: str = "",
        offset: int = 0,
        wait_seconds: int = 30,
        wait_for: str = "",
        input: str = "",
        path: str = "",
        condition: str = "exists",
        baseline: str = "",
        *,
        _context: ExecutionContext,
    ) -> ToolResult:
        _context.remaining()
        if action == "watch":
            if not path:
                return ToolResult.failure("invalid_arguments", "watch requires a path")
            return self.processes.watch_file(
                self.path(path), condition, wait_for, baseline, wait_seconds, _context
            )
        if action == "list":
            with self.processes.lock:
                sessions = [session.metadata() for session in self.processes.sessions.values()]
            return ToolResult(data={"sessions": sessions})
        if not session_id:
            return ToolResult.failure("invalid_arguments", "session_id is required for this action")
        session = self.processes.lookup(session_id)
        if session is None:
            return ToolResult.failure(
                "unknown_session",
                "Session not found in this application instance; do not rerun the command without checking its effects",
            )
        if offset > session.path.stat().st_size:
            return ToolResult.failure("invalid_arguments", "offset exceeds the current output size")
        if action == "wait":
            return self.processes.wait(session, offset, wait_seconds, wait_for, _context)
        if action == "input":
            if not input:
                return ToolResult.failure(
                    "invalid_arguments", "input must not be empty; send \\n for Enter"
                )
            return self.processes.send_input(session, input, offset)
        if action == "stop":
            session.stop()
        return self.processes.result(session, offset)

    def close(self) -> None:
        self.processes.close()
        self.worker_status.close()

    @staticmethod
    def _rg_options(glob: str | None = None) -> list[str]:
        options = [
            "rg",
            "--no-config",
            "--no-require-git",
            "--hidden",
            "--sort",
            "path",
            "--color",
            "never",
        ]
        if glob is not None:
            options += ["--glob", glob]
        return [*options, "--glob", "!**/.git/**", "--glob", "!**/.agent-artifacts/**"]

    def _search(
        self, command: list[str], context: ExecutionContext, *, tool: str = "grep"
    ) -> tuple[ProcessResult, ToolResult | None]:
        process = run_process(
            command, self.workspace, self.artifacts, context.child(120), tool=tool
        )
        if process.timed_out or process.exit_code not in (0, 1):
            output, truncated = _preview(process.path)
            code = "deadline_exceeded" if process.timed_out else "search_failed"
            message = "Search timed out" if process.timed_out else "ripgrep failed; inspect output"
            guidance = "Inspect the error and correct the query before retrying."
            if not process.timed_out and "compiled regex exceeds size limit" in output:
                code = "regex_too_large"
                message = "Regex exceeds ripgrep's compiled size limit"
                guidance = "Use literal=true with a short search string, then read the matching range. Remove enormous repetition bounds; do not retry variations of the same oversized regex."
            elif not process.timed_out and "regex parse error" in output:
                code = "invalid_regex"
                guidance = "Correct the regular expression, or use literal=true if searching for literal text."
            return process, ToolResult.failure(
                code,
                message,
                status="timeout" if process.timed_out else "error",
                data={"exit_code": process.exit_code, "output": output, "guidance": guidance},
                truncated=truncated,
                artifact_path=str(process.path),
            )
        return process, None

    @staticmethod
    def _page(
        items: Iterable[Any], offset: int, limit: int, context: ExecutionContext
    ) -> tuple[list[Any], int | None]:
        page = []
        size = 0
        for index, item in enumerate(items, 1):
            context.remaining()
            if index < offset:
                continue
            item_size = len(json.dumps(item, ensure_ascii=False))
            if page and (len(page) >= limit or size + item_size > PAGE_CHARS):
                return page, index
            page.append(item)
            size += item_size
        return page, None

    def grep(
        self,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        ignoreCase: bool = False,
        literal: bool = False,
        context: int = 0,
        offset: int = 1,
        limit: int = 100,
        next_cursor: dict[str, int] | None = None,
        *,
        _context: ExecutionContext,
    ) -> ToolResult:
        if next_cursor is not None:
            offset = next_cursor["offset"]
        command = [*self._rg_options(glob), "--json"]
        if ignoreCase:
            command.append("--ignore-case")
        if literal:
            command.append("--fixed-strings")
        if context:
            command += ["--context", str(context)]
        root = self.path(path)
        command += [*self._protected_globs(root if root.is_dir() else root.parent)]
        command += ["-e", pattern, "--", str(root)]
        process, error = self._search(command, _context)
        if error:
            return error

        def matches() -> Iterator[dict[str, Any]]:
            with process.path.open(encoding="utf-8") as stream:
                for line in stream:
                    _context.remaining()
                    event = json.loads(line)
                    if event["type"] in ("match", "context"):
                        data = event["data"]
                        yield {
                            "path": self._workspace_relative(_rg_text(data["path"])),
                            "line": data["line_number"],
                            "text": _rg_text(data["lines"]),
                            "kind": event["type"],
                        }

        records, next_offset = self._page(matches(), offset, limit, _context)
        return ToolResult(
            data={"matches": records},
            truncated=next_offset is not None,
            next_cursor={"offset": next_offset} if next_offset is not None else None,
            artifact_path=str(process.path),
        )

    def find(
        self,
        pattern: str,
        path: str = ".",
        offset: int = 1,
        limit: int = 1000,
        next_cursor: dict[str, int] | None = None,
        *,
        _context: ExecutionContext,
    ) -> ToolResult:
        if next_cursor is not None:
            offset = next_cursor["offset"]
        root = self.path(path)
        if not root.is_dir():
            raise NotADirectoryError(root)
        process, error = self._search(
            [
                *self._rg_options(pattern),
                *self._protected_globs(root),
                "--files",
                "--null",
                "--",
                str(root),
            ],
            _context,
            tool="find",
        )
        if error:
            return error
        names = (
            str(Path(item).relative_to(root)) for item in _null_records(process.path, _context)
        )
        paths, next_offset = self._page(names, offset, limit, _context)
        return ToolResult(
            data={"root": str(root), "paths": paths},
            truncated=next_offset is not None,
            next_cursor={"offset": next_offset} if next_offset is not None else None,
            artifact_path=str(process.path),
        )

    def ls(
        self,
        path: str = ".",
        offset: int = 1,
        limit: int = 500,
        next_cursor: dict[str, int] | None = None,
        *,
        _context: ExecutionContext,
    ) -> ToolResult:
        if next_cursor is not None:
            offset = next_cursor["offset"]
        root = self.path(path)
        entries = sorted(root.iterdir(), key=lambda entry: entry.name)
        names = (entry.name + ("/" if entry.is_dir() else "") for entry in entries)
        page, next_offset = self._page(names, offset, limit, _context)
        return ToolResult(
            data={"path": str(root), "entries": page},
            truncated=next_offset is not None,
            next_cursor={"offset": next_offset} if next_offset is not None else None,
        )
