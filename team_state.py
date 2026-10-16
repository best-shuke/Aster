"""Environment session state for a multi-agent team.

Shared state (the board: messages, tasks, groups, artifacts, decisions, protocols,
mailboxes) lives in one JSON file; each agent's private context lives in its own file.
Files sit under the environment state directory, outside the workspace, so they are not
part of what agents browse. A session directory is keyed by environment and session id;
one process holds it at a time.
"""

import fcntl
import hashlib
import json
import os
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

from file_state import atomic_write_text

VERSION = 1
MAIN = "main"
EVENT_TAIL = 300
WAKE_KINDS = frozenset({"message", "request", "reply", "task", "task_update", "protocol", "review"})
TERMINAL_TASK = frozenset({"completed", "failed", "cancelled"})


class TeamError(Exception):
    """A rejected team action; reported to the calling agent as a tool error."""

    def __init__(self, code: str, message: str, **data: Any):
        super().__init__(message)
        self.code = code
        self.data = data


def environment_key(entry) -> str:
    digest = hashlib.sha256(str(entry.path).encode()).hexdigest()[:12]
    return f"{entry.source}-{entry.name}-{digest}"


def now() -> float:
    return round(time.time(), 3)


def new_state(environment, session_id: str, workspace: Path) -> dict:
    return {
        "version": VERSION,
        "environment": {
            "id": environment.environment_id,
            "path": str(environment.entry.path),
        },
        "session_id": session_id,
        "workspace": str(workspace),
        "created_at": now(),
        "updated_at": now(),
        "sequence": {},
        "turn": {"number": 0, "messages": 0},
        "agents": {},
        "mailboxes": {MAIN: []},
        "messages": {},
        "tasks": {},
        "groups": {},
        "artifacts": {},
        "decisions": [],
        "protocols": {},
        "events": [],
    }


def agent_record(*, dynamic: dict | None = None) -> dict:
    return {
        "status": "idle",
        "activations_total": 0,
        "turn": {"activations": 0, "tool_calls": 0, "model_requests": 0, "tokens": 0},
        "inflight": [],
        "last_report": None,
        "last_error": None,
        "exhausted_notice": False,
        **({"dynamic": dynamic} if dynamic is not None else {}),
    }


class TeamStore:
    """Files and the cross-process lock for one environment session."""

    def __init__(self, state_dir: Path, environment, session_id: str):
        if not session_id or not all(character in "0123456789abcdef" for character in session_id):
            raise ValueError("invalid session id for team state")
        self.environment = environment
        self.session_id = session_id
        self.directory = state_dir / "sessions" / environment_key(environment.entry) / session_id
        self.state_file = self.directory / "state.json"
        self.events_file = self.directory / "events.jsonl"
        self.agents_dir = self.directory / "agents"
        self._lock_stream = None

    def open(self, workspace: Path) -> tuple[dict, list[str]]:
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.agents_dir.mkdir(exist_ok=True, mode=0o700)
        stream = (self.directory / "lock").open("a+b")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            stream.close()
            raise RuntimeError(
                "该环境会话正被另一个进程使用；请在那个进程中继续，或开启新会话。"
            ) from exc
        self._lock_stream = stream
        try:
            if self.state_file.exists():
                state = json.loads(self.state_file.read_text(encoding="utf-8"))
                self._validate(state)
                notices = recover(state)
            else:
                state = new_state(self.environment, self.session_id, workspace)
                notices = []
            return state, notices
        except BaseException:
            self.close()
            raise

    def _validate(self, state: dict) -> None:
        if not isinstance(state, dict) or state.get("version") != VERSION:
            raise RuntimeError(f"团队状态格式不受支持：{self.state_file}")
        owner = state.get("environment", {})
        # Sessions never cross environments: a state file belongs to one definition path.
        if owner.get("path") != str(self.environment.entry.path):
            raise RuntimeError("团队状态属于另一个环境，拒绝加载")
        if state.get("session_id") != self.session_id:
            raise RuntimeError("团队状态与会话 ID 不匹配")

    def save(self, state: dict) -> None:
        state["updated_at"] = now()
        atomic_write_text(self.state_file, json.dumps(state, ensure_ascii=False))
        with suppress(OSError):
            self.state_file.chmod(0o600)

    def append_event(self, event: dict) -> None:
        descriptor = os.open(self.events_file, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")

    def context_path(self, agent: str) -> Path:
        return self.agents_dir / f"{agent}.json"

    def load_context(self, agent: str) -> list[dict]:
        path = self.context_path(agent)
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        messages = data.get("messages") if isinstance(data, dict) else None
        if not isinstance(messages, list):
            raise RuntimeError(f"成员 {agent} 的私有上下文已损坏：{path}")
        return messages

    def save_context(self, agent: str, messages: list[dict]) -> None:
        path = self.context_path(agent)
        atomic_write_text(path, json.dumps({"version": VERSION, "messages": messages}))
        with suppress(OSError):
            path.chmod(0o600)

    def close(self) -> None:
        stream, self._lock_stream = self._lock_stream, None
        if stream is not None:
            with suppress(OSError):
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            stream.close()


def recover(state: dict) -> list[str]:
    """Return interrupted work to the mailboxes; completed work stays as recorded."""
    notices = []
    for name, agent in state.get("agents", {}).items():
        inflight = agent.get("inflight") or []
        if inflight:
            for item in inflight:
                item["redelivered"] = True
            state["mailboxes"].setdefault(name, [])[:0] = inflight
            agent["inflight"] = []
            notices.append(f"{name} 有 {len(inflight)} 项中断的工作已重新排队")
        if agent.get("status") in {"running", "queued", "stopping"}:
            agent["status"] = "idle"
    background = state.get("background")
    if isinstance(background, dict) and background.get("status") == "running":
        # The process ended while members worked in the background.
        background.update(status="stopped", ended_at=now())
        notices.append("上次进程退出时团队仍在后台工作，未完成的部分已重新排队")
    pending = sum(len(items) for items in state.get("mailboxes", {}).values())
    if pending:
        notices.append(f"恢复团队会话：{pending} 项待处理")
    return notices


class TeamState:
    """Pure state operations; the runtime calls them while holding its lock."""

    def __init__(self, data: dict):
        self.data = data

    # Identity and bookkeeping -------------------------------------------------------

    def next_id(self, prefix: str) -> str:
        sequence = self.data["sequence"]
        sequence[prefix] = sequence.get(prefix, 0) + 1
        return f"{prefix}{sequence[prefix]}"

    @property
    def agents(self) -> dict[str, dict]:
        return self.data["agents"]

    def mailbox(self, name: str) -> list[dict]:
        return self.data["mailboxes"].setdefault(name, [])

    def sync_members(self, names: list[str]) -> None:
        for name in names:
            self.agents.setdefault(name, agent_record())
            self.mailbox(name)
        for name, record in self.agents.items():
            if name not in names and "dynamic" not in record:
                record["status"] = "removed"

    def event(self, event_type: str, actor: str, **data: Any) -> dict:
        event = {"id": self.next_id("e"), "at": now(), "type": event_type, "actor": actor, **data}
        events = self.data["events"]
        events.append(event)
        del events[:-EVENT_TAIL]
        return event

    # Messages -----------------------------------------------------------------------

    def add_message(self, **fields: Any) -> dict:
        message = {"id": self.next_id("m"), "created_at": now(), "replied_by": [], **fields}
        self.data["messages"][message["id"]] = message
        self.data["turn"]["messages"] += 1
        return message

    def visible_message(self, message_id: str, agent: str) -> dict | None:
        message = self.data["messages"].get(message_id)
        if message is None:
            return None
        if agent != message["sender"] and agent not in message["recipients"]:
            return None
        return message

    # Board views --------------------------------------------------------------------

    def group(self, group_id: str) -> dict | None:
        return self.data["groups"].get(group_id)

    def artifact_visible(self, artifact: dict, agent: str) -> bool:
        if artifact["author"] == agent or artifact["visibility"] == "team":
            return True
        if artifact["visibility"] == "group":
            group = self.group(artifact.get("group") or "")
            return group is not None and agent in group["members"]
        return False

    def protocol_visible(self, protocol: dict, agent: str) -> bool:
        return agent in {protocol["initiator"], protocol["facilitator"], *protocol["participants"]}

    def view(self, agent: str) -> dict:
        """What one agent may see: never another agent's private context or others' DMs."""
        data = self.data
        return {
            "you": agent,
            "turn": data["turn"]["number"],
            "members": {
                name: {
                    "status": record["status"],
                    "activations_this_turn": record["turn"]["activations"],
                }
                for name, record in data["agents"].items()
                if record["status"] != "removed"
            },
            "tasks": [
                {
                    key: task.get(key)
                    for key in (
                        "id",
                        "title",
                        "kind",
                        "creator",
                        "assignee",
                        "status",
                        "depends_on",
                    )
                }
                | ({"result": excerpt(task.get("result"), 400)} if task.get("result") else {})
                for task in data["tasks"].values()
            ],
            "groups": [group for group in data["groups"].values() if agent in group["members"]],
            "artifacts": [
                {key: artifact[key] for key in ("id", "title", "author", "visibility")}
                for artifact in data["artifacts"].values()
                if self.artifact_visible(artifact, agent)
            ],
            "decisions": data["decisions"][-20:],
            "protocols": [
                {
                    key: protocol.get(key)
                    for key in ("id", "protocol", "topic", "status", "round", "rounds")
                }
                | {"participant": self.protocol_visible(protocol, agent)}
                for protocol in data["protocols"].values()
            ],
            "unread": len(data["mailboxes"].get(agent, [])),
        }


def excerpt(value: Any, limit: int) -> Any:
    if not isinstance(value, str) or len(value) <= limit:
        return value
    return value[: limit - 20] + f"… [{len(value) - limit + 20} more chars]"
