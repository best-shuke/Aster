import fcntl
import hashlib
import json
import re
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from context_budget import LOCAL_METADATA
from file_state import atomic_write_text, file_digest
from goals import GoalState
from tool_runtime import validate_tool_calls


class SessionConflictError(RuntimeError):
    pass


def displayed_text(message: dict) -> object:
    """A message as the user wrote it: a /skill: command rather than the skill it loaded."""
    metadata = message.get(LOCAL_METADATA)
    display = metadata.get("display") if isinstance(metadata, dict) else None
    return display if isinstance(display, str) else message.get("content")


@dataclass(frozen=True)
class SavedSession:
    session_id: str
    title: str
    updated_at: datetime
    message_count: int


class SessionStore:
    def __init__(self, active_file: Path) -> None:
        self.active_file = active_file
        self.directory = active_file.with_name(f"{active_file.stem}.sessions")
        self.diagnostics: list[str] = []
        self._revision = file_digest(active_file)

    def snapshot_path(self, session_id: str) -> Path:
        if not isinstance(session_id, str) or re.fullmatch(r"[0-9a-f]{32}", session_id) is None:
            raise ValueError("invalid session ID")
        return self.directory / f"{session_id}.json"

    @staticmethod
    def _validate_messages(messages: object) -> None:
        if not isinstance(messages, list) or any(
            not isinstance(message, dict) or not isinstance(message.get("role"), str)
            for message in messages
        ):
            raise ValueError("invalid messages in session")
        for message in messages:
            calls = message.get("tool_calls")
            if calls is not None:
                validate_tool_calls(calls)
            if message["role"] == "tool" and not isinstance(message.get("tool_call_id", ""), str):
                raise ValueError("invalid tool result in session")

    @staticmethod
    def _validate_environment(data: dict) -> None:
        environment_id = data.get("environment_id")
        if environment_id is not None and (
            not isinstance(environment_id, str) or not environment_id or len(environment_id) > 100
        ):
            raise ValueError("invalid environment ID")
        reference = data.get("environment_ref")
        if reference is not None and (
            environment_id is None
            or not isinstance(reference, dict)
            or not isinstance(reference.get("path"), str)
            or not isinstance(reference.get("fingerprint"), str)
            or re.fullmatch(r"[0-9a-f]{64}", reference["fingerprint"]) is None
        ):
            raise ValueError("invalid environment reference")

    def _validate(self, data: dict) -> None:
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("unsupported session format")
        self._validate_messages(data["messages"])
        journal = data.get("tool_runs", [])
        if not isinstance(journal, list) or any(not isinstance(run, dict) for run in journal):
            raise ValueError("invalid tool execution journal")
        state = data.get("task_state")
        if state is not None and not isinstance(state, dict):
            raise ValueError("invalid task state")
        outcome = data.get("last_outcome")
        if outcome is not None and not isinstance(outcome, dict):
            raise ValueError("invalid task outcome")
        session_id = data.get("session_id")
        if session_id is not None:
            self.snapshot_path(session_id)
        self._validate_environment(data)
        if data.get("goal") is not None:
            GoalState.from_dict(data["goal"])

    def read(self, path: Path | None = None) -> dict:
        path = self.active_file if path is None else path
        try:
            content = path.read_bytes()
            data = json.loads(content)
            self._validate(data)
            if path == self.active_file:
                self._revision = hashlib.sha256(content).hexdigest()
            return data
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            raise RuntimeError(f"Cannot load session file {path}: {exc}") from exc

    def _write(self, path: Path, content: str) -> None:
        if path.parent == self.directory and not (self.directory / ".gitignore").exists():
            # Saved conversations stay out of `git status` in the user's repository.
            self.directory.mkdir(parents=True, exist_ok=True)
            with suppress(OSError):
                (self.directory / ".gitignore").write_text("# Aster sessions\n*\n", "utf-8")
        atomic_write_text(path, content)

    @contextmanager
    def _locked(self):
        self.active_file.parent.mkdir(parents=True, exist_ok=True)
        lock_file = self.active_file.with_name(f"{self.active_file.name}.lock")
        with lock_file.open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                if file_digest(self.active_file) != self._revision:
                    raise SessionConflictError(
                        "会话已被另一个实例更新，未覆盖任何记录。请重启本实例后继续，"
                        "或使用不同的 DEEPSEEK_SESSION_FILE。"
                    )
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def remove_active(self) -> None:
        with self._locked():
            self.active_file.unlink(missing_ok=True)
            self._revision = None

    def save(self, data: dict) -> None:
        snapshot = self.snapshot_path(data["session_id"])
        # Saved after every tool step: unindented JSON uses the C encoder, ~2x faster.
        content = json.dumps(data, ensure_ascii=False)
        with self._locked():
            if data["messages"] or data.get("goal") is not None:
                self._write(snapshot, content)
            self._write(self.active_file, content)
            self._revision = hashlib.sha256(content.encode("utf-8")).hexdigest()

    def transition(self, previous: dict, data: dict) -> None:
        content = json.dumps(data, ensure_ascii=False)
        snapshots = {}
        for item in (previous, data):
            if item["messages"] or item.get("goal") is not None:
                snapshots[self.snapshot_path(item["session_id"])] = json.dumps(
                    item, ensure_ascii=False
                )
        with self._locked():
            originals = {
                path: path.read_bytes().decode("utf-8") if path.exists() else None
                for path in snapshots
            }
            try:
                for path, snapshot in snapshots.items():
                    self._write(path, snapshot)
                self._write(self.active_file, content)
            except BaseException:
                for path, original in originals.items():
                    if original is None:
                        path.unlink(missing_ok=True)
                    else:
                        atomic_write_text(path, original)
                raise
            self._revision = hashlib.sha256(content.encode("utf-8")).hexdigest()

    def list(self) -> list[SavedSession]:
        self.diagnostics.clear()
        sessions = []
        for path in self.directory.glob("*.json"):
            try:
                data = self.read(path)
                session_id = data["session_id"]
                if path != self.snapshot_path(session_id):
                    raise ValueError("session ID does not match file name")
                messages = data["messages"]
                count = sum(message["role"] in {"user", "assistant"} for message in messages)
                if not count and data.get("goal") is None:
                    continue
                title = next(
                    (
                        " ".join(text.split())[:80]
                        for message in messages
                        if message["role"] == "user"
                        and isinstance(text := displayed_text(message), str)
                        and text.strip()
                    ),
                    (data.get("goal") or {}).get("objective", "未命名会话"),
                )
                sessions.append(
                    SavedSession(
                        session_id,
                        title,
                        datetime.fromisoformat(data["updated_at"]).astimezone(),
                        count,
                    )
                )
            except (RuntimeError, KeyError, TypeError, ValueError) as exc:
                self.diagnostics.append(f"已跳过无法读取的历史会话 {path.name}: {exc}")
        return sorted(sessions, key=lambda session: session.updated_at, reverse=True)
