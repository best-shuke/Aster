"""Managed command sessions that survive a paused model turn.

Only explicitly backgrounded commands use this lifetime. Each session has its
own timeout and is cleaned up when the owning application closes.
"""

import codecs
import json
import os
import signal
import subprocess
import threading
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from stat import S_ISREG
from uuid import uuid4

from tool_runtime import (
    PAGE_CHARS,
    ArtifactStore,
    ExecutionContext,
    ToolResult,
    command_environment,
)


@dataclass
class ProcessSession:
    session_id: str
    command: str
    cwd: Path
    path: Path
    process: subprocess.Popen
    started: float
    timeout: int
    state: str = "running"
    ended: float | None = None
    done: threading.Event = field(default_factory=threading.Event)
    lock: threading.RLock = field(default_factory=threading.RLock)
    watcher: threading.Thread | None = None

    def terminate(self) -> None:
        with suppress(ProcessLookupError):
            os.killpg(self.process.pid, signal.SIGKILL)

    def watch(self) -> None:
        try:
            remaining = max(0, self.started + self.timeout - time.monotonic())
            try:
                self.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                with self.lock:
                    if self.state == "running":
                        self.state = "timed_out"
                    self.terminate()
                self.process.wait()
        finally:
            # A shell exiting must not leave descendants using the session log.
            self.terminate()
            self.process.wait()
            with self.lock:
                if self.process.stdin is not None:
                    self.process.stdin.close()
                if self.state == "running":
                    self.state = "completed" if self.process.returncode == 0 else "failed"
                self.ended = time.monotonic()
                self.done.set()

    def stop(self) -> None:
        with self.lock:
            if not self.done.is_set() and self.process.poll() is None:
                self.state = "stopped"
                self.terminate()
        self.done.wait()

    def metadata(self) -> dict:
        with self.lock:
            return {
                "session_id": self.session_id,
                "pid": self.process.pid,
                "command": self.command[:240],
                "command_truncated": len(self.command) > 240,
                "cwd": str(self.cwd),
                "state": self.state,
                "exit_code": self.process.returncode if self.done.is_set() else None,
                "timed_out": self.state == "timed_out",
                "duration_seconds": round((self.ended or time.monotonic()) - self.started, 3),
                "timeout_seconds": self.timeout,
                "artifact_path": str(self.path),
            }


class ProcessMonitor:
    def __init__(self, artifacts: ArtifactStore) -> None:
        self.artifacts = artifacts
        self.sessions: dict[str, ProcessSession] = {}
        self.lock = threading.RLock()
        self.closed = False

    def start(self, command: str, cwd: Path, timeout: int) -> ToolResult:
        with self.lock:
            if self.closed:
                return ToolResult.failure("monitor_closed", "Process monitor is closed")
            path = self.artifacts.new_path(".log")
            started = time.monotonic()
            process = None
            try:
                with path.open("xb") as output:
                    process = subprocess.Popen(
                        command,
                        cwd=cwd,
                        shell=True,
                        stdin=subprocess.PIPE,
                        stdout=output,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                        env=command_environment(),
                    )
            except BaseException:
                if process is not None:
                    with suppress(ProcessLookupError):
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                    process.stdin.close()
                path.unlink(missing_ok=True)
                raise
            session = ProcessSession(uuid4().hex, command, cwd, path, process, started, timeout)
            try:
                self.sessions[session.session_id] = session
                os.set_blocking(process.stdin.fileno(), False)
                session.watcher = threading.Thread(
                    target=session.watch, name=f"process-{session.session_id[:8]}", daemon=True
                )
                session.watcher.start()
            except BaseException:
                session.terminate()
                process.wait()
                process.stdin.close()
                self.sessions.pop(session.session_id, None)
                raise
        return self.result(session)

    def lookup(self, session_id: str) -> ProcessSession | None:
        with self.lock:
            return self.sessions.get(session_id)

    @staticmethod
    def read_output(session: ProcessSession, offset: int) -> tuple[str, int, bool]:
        with session.path.open("rb") as stream:
            if offset > os.fstat(stream.fileno()).st_size:
                raise ValueError("offset exceeds the current output size")
            stream.seek(offset)
            chunk = stream.read(PAGE_CHARS)
            while True:
                decoder = codecs.getincrementaldecoder("utf-8")("replace")
                output = decoder.decode(
                    chunk,
                    final=session.done.is_set()
                    and offset + len(chunk) >= os.fstat(stream.fileno()).st_size,
                )
                pending, _ = decoder.getstate()
                if len(json.dumps(output, ensure_ascii=False)) <= PAGE_CHARS:
                    break
                # Escaped control characters also count against the JSON budget.
                chunk = chunk[: len(chunk) // 2]
            next_offset = offset + len(chunk) - len(pending)
            more = next_offset < os.fstat(stream.fileno()).st_size - len(pending)
        return output, next_offset, more

    def result(self, session: ProcessSession, offset: int = 0, **details) -> ToolResult:
        output, next_offset, more = self.read_output(session, offset)
        data = session.metadata()
        data.update(output=output, next_offset=next_offset, has_more_output=more, **details)
        result = ToolResult(
            data=data,
            truncated=more,
            next_cursor={"offset": next_offset} if more else None,
            artifact_path=str(session.path),
        )
        if data["state"] == "timed_out":
            result.status = "timeout"
            result.error = {"code": "process_timeout", "message": "Background command timed out"}
        elif data["state"] == "failed":
            result.status = "error"
            result.error = {
                "code": "nonzero_exit",
                "message": f"Command exited with code {data['exit_code']}",
            }
        return result

    def wait(
        self,
        session: ProcessSession,
        offset: int,
        seconds: int,
        wait_for: str,
        context: ExecutionContext,
    ) -> ToolResult:
        until = time.monotonic() + seconds
        needle = wait_for.encode("utf-8")
        # Include a partial marker consumed by a previous output page/wait.
        position = max(0, offset - len(needle) + 1) if needle else offset
        tail = b""
        matched = False
        with session.path.open("rb") as stream:
            if offset > os.fstat(stream.fileno()).st_size:
                raise ValueError("offset exceeds the current output size")
            stream.seek(position)
            check_limit = os.fstat(stream.fileno()).st_size if seconds == 0 else None
            while True:
                remaining = context.remaining()
                if needle:
                    chunk = stream.read(
                        min(65536, max(0, check_limit - position))
                        if check_limit is not None
                        else 65536
                    )
                    position += len(chunk)
                    text = tail + chunk
                    search_from = max(0, offset - (position - len(text)) - len(needle) + 1)
                    if needle in text[search_from:]:
                        matched = True
                        break
                    tail = text[-(len(needle) - 1) :] if len(needle) > 1 else b""
                    if chunk and (seconds == 0 or time.monotonic() < until):
                        continue
                if session.done.is_set():
                    # Drain bytes written between the last read and process exit.
                    if (
                        needle
                        and position < os.fstat(stream.fileno()).st_size
                        and time.monotonic() < until
                    ):
                        continue
                    break
                left = until - time.monotonic()
                if left <= 0:
                    break
                session.done.wait(min(0.1, left, remaining))
        event = "matched" if matched else "exited" if session.done.is_set() else "waiting"
        return self.result(session, offset, event=event, matched=matched)

    @staticmethod
    def watch_file(
        path: Path,
        condition: str,
        wait_for: str,
        baseline: str,
        seconds: int,
        context: ExecutionContext,
    ) -> ToolResult:
        """Wait for a local state change without running a model for every poll."""
        if condition == "contains" and not wait_for:
            return ToolResult.failure("invalid_arguments", "contains requires a nonempty wait_for")
        until = time.monotonic() + seconds
        initial = baseline
        last_checked = None
        matched = False
        needle = wait_for.encode("utf-8")
        while True:
            remaining = context.remaining()
            try:
                stat = path.stat()
                version = f"{stat.st_dev}:{stat.st_ino}:{stat.st_size}:{stat.st_mtime_ns}:{stat.st_ctime_ns}"
                exists = True
            except FileNotFoundError:
                version, exists = "missing", False
            if not initial:
                initial = version
            if condition == "exists":
                matched = exists
            elif condition == "changed":
                matched = version != initial
            elif exists and version != last_checked:
                if not S_ISREG(stat.st_mode):
                    return ToolResult.failure(
                        "invalid_arguments", "contains requires a regular file"
                    )
                try:
                    with path.open("rb") as stream:
                        tail = b""
                        check_limit = os.fstat(stream.fileno()).st_size if seconds == 0 else None
                        while chunk := stream.read(
                            min(65536, max(0, check_limit - stream.tell()))
                            if check_limit is not None
                            else 65536
                        ):
                            context.remaining()
                            text = tail + chunk
                            if needle in text:
                                matched = True
                                break
                            tail = text[-(len(needle) - 1) :] if len(needle) > 1 else b""
                            if seconds and time.monotonic() >= until:
                                break
                except FileNotFoundError:
                    # Writers may replace/delete the file between stat and open.
                    pass
                last_checked = version
            if matched or time.monotonic() >= until:
                return ToolResult(
                    data={
                        "path": str(path),
                        "condition": condition,
                        "state": "ready" if matched else "waiting",
                        "event": "matched" if matched else "waiting",
                        "matched": matched,
                        "baseline": initial,
                        "current_version": version,
                    }
                )
            # Cancellation affects this wait, never an independently running program.
            context.cancelled.wait(min(0.2, remaining, max(0, until - time.monotonic())))

    def send_input(self, session: ProcessSession, text: str, offset: int) -> ToolResult:
        with session.lock:
            if session.done.is_set() or session.process.poll() is not None:
                return ToolResult.failure("process_exited", "The command is no longer running")
            try:
                written = os.write(session.process.stdin.fileno(), text.encode("utf-8"))
            except BlockingIOError:
                return ToolResult.failure(
                    "stdin_busy", "Command stdin is full; wait before retrying"
                )
            except BrokenPipeError:
                return ToolResult.failure("stdin_closed", "Command stdin is closed")
        result = self.result(session, offset, bytes_written=written)
        if written < len(text.encode("utf-8")):
            result.status = "partial"
            result.error = {
                "code": "partial_input",
                "message": "Only bytes_written bytes were sent; do not resend the complete input",
            }
        return result

    def stop_all(self) -> int:
        with self.lock:
            sessions = list(self.sessions.values())
        active = [session for session in sessions if not session.done.is_set()]
        for session in active:
            session.stop()
        return len(active)

    def close(self) -> None:
        with self.lock:
            self.closed = True
        self.stop_all()
