"""Scheduler and shared-state service for one multi-agent environment session.

Members run as independent activations on worker threads, each with its own private
context and model client. Delivering an item to an idle member schedules it; items for a
busy member wait in its mailbox and are coalesced into its next activation. Team activity
is bound to the main agent's turn: begin_turn resumes pending work and end_turn pauses
it (running activations are cancelled and their items requeued). Every change is
persisted, so a later turn, /resume, or a new process continues where work stopped.
"""

import threading
import time
from collections.abc import Callable
from contextlib import contextmanager, suppress
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

from environment_spec import AgentSpec, parse_agent
from skills import SkillCatalog, skill_block
from team_protocols import ProtocolEngine
from team_state import (
    MAIN,
    TERMINAL_TASK,
    WAKE_KINDS,
    TeamError,
    TeamState,
    TeamStore,
    agent_record,
    excerpt,
    now,
)
from team_tools import activation_text, activation_title
from tool_runtime import ExecutionContext

ITEMS_PER_ACTIVATION = 12
GROUP_LIMIT = 20
STOP_GRACE_SECONDS = 10


class MainOrigin:
    """The main agent acts at the root of every chain: hop 0, task depth 0."""

    hop = 0
    depth = 0
    agent = MAIN

    def __init__(self):
        self.replied: set[str] = set()
        self.contributed: set[str] = set()
        self.updated: set[str] = set()


class Activation:
    def __init__(self, agent: str, items: list[dict], deadline: float, session: str | None):
        self.id = uuid4().hex[:12]
        self.session = session
        self.agent = agent
        self.items = items
        self.deadline = deadline
        self.cancelled = threading.Event()
        self.thread: threading.Thread | None = None
        self.started = time.monotonic()
        self.hop = max((item.get("hop", 0) for item in items), default=0)
        self.depth = max((item.get("depth", 0) for item in items), default=0)
        self.replied: set[str] = set()
        self.contributed: set[str] = set()
        self.updated: set[str] = set()


@dataclass
class ActivationResult:
    status: str
    final: str = ""
    error: str | None = None


class TeamRuntime:
    def __init__(
        self,
        environment,
        *,
        settings,
        client,
        base_tools,
        mcp_manager=None,
        skills: SkillCatalog | None = None,
        state_dir: Path,
        protected_paths: tuple[Path, ...] = (),
        on_progress: Callable[[dict], None] | None = None,
        member_factory=None,
    ):
        from team_agent import MemberAgent

        self.environment = environment
        self.team = environment.team
        self.limits = environment.team.limits
        self.settings = settings
        self.client = client
        self.base_tools = base_tools
        self.mcp_manager = mcp_manager
        self.skills = skills
        self.state_dir = state_dir
        self.protected_paths = protected_paths
        self.on_progress = on_progress
        self.member_factory = member_factory or MemberAgent
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.store: TeamStore | None = None
        self.state: TeamState | None = None
        self.session_id: str | None = None
        self.protocols = ProtocolEngine(self)
        self.members: dict[str, Any] = {}
        self.activations: dict[str, Activation] = {}
        self.accepting = False
        self.turn_deadline = float("inf")
        self.nudges = 0
        self.notices: list[str] = []
        self.main_origin = MainOrigin()

    # Members -------------------------------------------------------------------------

    def member_specs(self) -> list[AgentSpec]:
        specs = list(self.team.agents)
        if self.state is not None:
            for name, record in self.state.agents.items():
                dynamic = record.get("dynamic")
                if dynamic is not None and record["status"] != "removed":
                    specs.append(self._dynamic_spec(name, dynamic))
        return specs

    def member_spec(self, name: str) -> AgentSpec | None:
        return next((spec for spec in self.member_specs() if spec.name == name), None)

    def _dynamic_spec(self, name: str, data: dict) -> AgentSpec:
        spec = parse_agent(
            name,
            {key: value for key, value in data.items() if key != "spawned_by"},
            read_prompt=None,
            tool_names=frozenset(data.get("tools", [])),
        )
        return replace(spec, dynamic=True)

    def member_tool_names(self, spec: AgentSpec) -> frozenset[str]:
        from team_agent import allowed_tools

        return allowed_tools(self, spec)

    def member(self, name: str):
        with self.lock:
            member = self.members.get(name)
            if member is None:
                spec = self.member_spec(name)
                if spec is None:
                    raise TeamError("unknown_member", f"Unknown member {name!r}")
                member = self.members[name] = self.member_factory(spec, self)
            return member

    def require_member(self, name: str, *, allow_main: bool = True) -> None:
        if name == MAIN:
            if allow_main:
                return
            raise TeamError("invalid_member", "main cannot take part here; choose members")
        spec = self.member_spec(name)
        record = self.state.agents.get(name)
        if spec is None or record is None or record["status"] == "removed":
            names = ", ".join(spec.name for spec in self.member_specs() if spec.enabled)
            raise TeamError("unknown_member", f"Unknown member {name!r}; members: {names}")
        if not spec.enabled:
            raise TeamError("member_disabled", f"Member {name!r} is disabled in this environment")

    def skill_names(self, spec: AgentSpec) -> list[str]:
        if self.skills is None or not self.skills.enabled:
            return []
        active = [skill.name for skill in self.skills.skills]
        if spec.skills is None:
            return active
        return [name for name in spec.skills if name in active]

    # Session lifecycle --------------------------------------------------------------

    def open_session(self, session_id: str) -> list[str]:
        with self.lock:
            self.accepting = False
        # Workers need the lock to finish, so never wait for them while holding it.
        self._stop_activations()
        with self.lock:
            self._close_store()
            store = TeamStore(self.state_dir, self.environment, session_id)
            data, notices = store.open(self.settings.workspace)
            self.store, self.state, self.session_id = store, TeamState(data), session_id
            self.members = {}
            self.state.sync_members([spec.name for spec in self.team.agents])
            for spec in self.team.agents:
                if not spec.enabled:
                    self.state.agents[spec.name]["status"] = "disabled"
                elif self.state.agents[spec.name]["status"] == "disabled":
                    self.state.agents[spec.name]["status"] = "idle"
            self.notices = notices
            self._event("session_opened", MAIN, recovered=notices)
            self._persist()
            return notices

    def close(self) -> None:
        with self.lock:
            self.accepting = False
        self._stop_activations()
        with self.lock:
            self._close_store()

    def _close_store(self) -> None:
        if self.store is not None:
            try:
                self._persist()
            finally:
                self.store.close()
                self.store = None

    @contextmanager
    def turn(self, deadline: float):
        self.begin_turn(deadline)
        interrupted = True
        try:
            yield
            interrupted = False
        finally:
            self.end_turn(interrupted=interrupted)

    def begin_turn(self, deadline: float) -> None:
        with self.lock:
            self._require_open()
            self.accepting = True
            self.turn_deadline = deadline
            self.nudges = 0
            data = self.state.data
            data["turn"] = {"number": data["turn"]["number"] + 1, "messages": 0}
            for record in self.state.agents.values():
                record["turn"] = {
                    "activations": 0,
                    "tool_calls": 0,
                    "model_requests": 0,
                    "tokens": 0,
                }
                record["exhausted_notice"] = False
                if record["status"] == "exhausted":
                    record["status"] = "idle"
            self._event("turn_started", MAIN, turn=data["turn"]["number"])
            self._drain()
            self._persist()

    def end_turn(self, *, interrupted: bool = False) -> int:
        with self.lock:
            if self.state is None:
                return 0
            self.accepting = False
            paused = len(self.activations)
        self._stop_activations()
        with self.lock:
            if self.state is not None:
                self._event("turn_ended", MAIN, paused=paused, interrupted=interrupted)
                self._persist()
            self.changed.notify_all()
        return paused

    def _stop_activations(self) -> None:
        with self.lock:
            running = list(self.activations.values())
            for activation in running:
                activation.cancelled.set()
        deadline = time.monotonic() + STOP_GRACE_SECONDS
        for activation in running:
            thread = activation.thread
            if thread is not None and thread is not threading.current_thread():
                thread.join(max(0, deadline - time.monotonic()))

    # Scheduling ---------------------------------------------------------------------

    def deliver(self, name: str, item: dict) -> None:
        """Append to a mailbox and wake its owner (caller holds the lock)."""
        self.state.mailbox(name).append(item)
        if name == MAIN:
            self.changed.notify_all()
            return
        self._schedule(name)

    def _wakeable(self, name: str) -> list[dict]:
        return [item for item in self.state.mailbox(name) if item["kind"] in WAKE_KINDS]

    def _exhausted(self, name: str) -> str | None:
        turn = self.state.agents[name]["turn"]
        limits = self.limits
        if turn["activations"] >= limits.max_activations_per_agent:
            return f"{name} reached its limit of {limits.max_activations_per_agent} activations this turn"
        if turn["tokens"] >= limits.token_budget_per_agent:
            return f"{name} exhausted its token budget for this turn"
        return None

    def _schedule(self, name: str) -> None:
        if not self.accepting or name in self.activations:
            return
        record = self.state.agents.get(name)
        if record is None or record["status"] in {"disabled", "removed"}:
            return
        if not self._wakeable(name):
            record["status"] = "idle"
            return
        reason = self._exhausted(name)
        if reason is not None:
            self._exhaust(name, reason)
            return
        if len(self.activations) >= self.limits.max_concurrent_agents:
            record["status"] = "queued"
            self._progress(name, state="pending", activity="queued")
            return
        self._start(name)

    def _exhaust(self, name: str, reason: str) -> None:
        record = self.state.agents[name]
        record["status"] = "exhausted"
        if not record["exhausted_notice"]:
            record["exhausted_notice"] = True
            self._event("budget_exhausted", name, reason=reason)
            self._progress(name, state="blocked", activity="warning", warning=reason)
        # Nobody may wait forever on an agent that cannot run: settle its pending work.
        data = self.state.data
        owed, record["owed"] = record.get("owed", []), []
        for message_id in owed:
            message = data["messages"].get(message_id)
            if message is not None and name not in message["replied_by"]:
                self._post(
                    name,
                    [message["sender"]],
                    "(automatic, budget exhausted) Latest report: "
                    + (record.get("last_report") or reason),
                    kind="reply",
                    reply_to=message_id,
                    hop=message.get("hop", 0) + 1,
                    depth=message.get("depth", 0),
                    auto=True,
                )
        owed_tasks, record["owed_tasks"] = record.get("owed_tasks", []), []
        for task_id in owed_tasks:
            task = data["tasks"].get(task_id)
            if task is not None and task["status"] not in TERMINAL_TASK:
                self._task_status(
                    task, name, "blocked", f"{reason}. {record.get('last_report') or ''}"
                )
        # Settling can deliver new items to this mailbox, so loop until nothing is left.
        mailbox = self.state.mailbox(name)
        while True:
            pending = [
                item
                for item in mailbox
                if item["kind"] == "protocol"
                or (item["kind"] == "request" and not item.get("auto_replied"))
            ]
            if not pending:
                return
            for item in pending:
                if item["kind"] == "protocol":
                    mailbox[:] = [other for other in mailbox if other is not item]
                    self.protocols.absent(name, item, reason)
                    continue
                item["auto_replied"] = True
                self._post(
                    name,
                    [self.state.data["messages"][item["ref"]]["sender"]],
                    f"(automatic) I cannot answer now: {reason}. Your request stays queued "
                    "until my next turn.",
                    kind="reply",
                    reply_to=item["ref"],
                    hop=item.get("hop", 0) + 1,
                    depth=item.get("depth", 0),
                    auto=True,
                )

    def _drain(self) -> None:
        waiting = sorted(
            (name for name in self.state.agents if name != MAIN and self._wakeable(name)),
            key=lambda name: self._wakeable(name)[0].get("created_at", 0),
        )
        for name in waiting:
            if len(self.activations) >= self.limits.max_concurrent_agents:
                for rest in waiting:
                    if rest not in self.activations and self.state.agents[rest]["status"] == "idle":
                        self.state.agents[rest]["status"] = "queued"
                break
            self._schedule(name)

    def _start(self, name: str) -> None:
        mailbox = self.state.mailbox(name)
        items, rest = mailbox[:ITEMS_PER_ACTIVATION], mailbox[ITEMS_PER_ACTIVATION:]
        mailbox[:] = rest
        deadline = min(
            self.turn_deadline, time.monotonic() + self.limits.activation_timeout_seconds
        )
        activation = Activation(name, items, deadline, self.session_id)
        record = self.state.agents[name]
        record["status"] = "running"
        record["inflight"] = items
        record["turn"]["activations"] += 1
        record["activations_total"] += 1
        self.activations[name] = activation
        title = activation_title(self, items)
        self._event(
            "activation_started", name, activation=activation.id, items=len(items), title=title
        )
        self._persist()
        self._progress(
            name, state="running", activity="model", model_phase="connecting", title=title
        )
        thread = threading.Thread(
            target=self._run, args=(activation,), name=f"team-{name}", daemon=True
        )
        activation.thread = thread
        thread.start()

    def _run(self, activation: Activation) -> None:
        try:
            member = self.member(activation.agent)
            result = member.run(activation)
        except BaseException as exc:
            result = ActivationResult("error", error=f"{type(exc).__name__}: {exc}")
        with self.lock:
            try:
                if activation.session == self.session_id and self.state is not None:
                    self._finish(activation, result)
            finally:
                if self.activations.get(activation.agent) is activation:
                    del self.activations[activation.agent]
                if self.state is not None and activation.session == self.session_id:
                    self._drain()
                    self._persist()
                self.changed.notify_all()

    def _finish(self, activation: Activation, result: ActivationResult) -> None:
        if self.state is None:
            return
        name = activation.agent
        record = self.state.agents[name]
        record["inflight"] = []
        title = activation_title(self, activation.items)
        if result.status == "cancelled" or (
            activation.cancelled.is_set() and result.status != "ok"
        ):
            # Paused, not failed: the work returns to the mailbox for the next turn.
            for item in activation.items:
                item["redelivered"] = True
            self.state.mailbox(name)[:0] = activation.items
            record["status"] = "idle"
            self._event("activation_cancelled", name, activation=activation.id)
            self._progress(name, state="paused", activity="finished", title=title)
            return
        ok = result.status == "ok"
        record["status"] = "idle"
        record["last_report"] = excerpt(result.final, 4000) if result.final else None
        record["last_error"] = result.error
        final = result.final.strip() if ok and result.final else ""
        fallback = (
            final or f"(no report: {result.status}{': ' + result.error if result.error else ''})"
        )
        succeeded = ok and bool(final)
        for item in activation.items:
            if item["kind"] == "protocol":
                self._settle_protocol(name, activation, item, fallback, succeeded)
        self._settle_obligations(name, activation, fallback, succeeded)
        self._event(
            "activation_finished" if ok else "activation_failed",
            name,
            activation=activation.id,
            status=result.status,
            error=result.error,
            report=excerpt(result.final, 600) if result.final else None,
        )
        self._progress(
            name,
            state="completed" if ok else result.status,
            activity="finished",
            title=title,
            error={"code": result.status, "message": result.error} if result.error else None,
        )

    def waits_for(self, name: str) -> set[str]:
        """Agents this one waits on: unanswered requests, delegated tasks, its protocols."""
        data = self.state.data
        waits: set[str] = set()
        for message in data["messages"].values():
            if message["sender"] == name and message["kind"] == "request":
                waits.update(
                    recipient
                    for recipient in message["recipients"]
                    if recipient not in message["replied_by"]
                )
        for task in data["tasks"].values():
            if (
                task["creator"] == name
                and task["status"] not in TERMINAL_TASK | {"blocked"}
                and task["assignee"] not in {None, name}
            ):
                waits.add(task["assignee"])
        for record in data["protocols"].values():
            if record["initiator"] == name and record["status"] in {"running", "concluding"}:
                waits.update(record["participants"])
                waits.add(record["facilitator"])
        waits.discard(name)
        return waits

    def _closes_cycle(self, name: str, waits: set[str]) -> bool:
        """True when someone this agent waits on is (transitively) waiting on it."""
        seen, pending = set(), list(waits)
        while pending:
            other = pending.pop()
            if other in seen:
                continue
            seen.add(other)
            others = self.waits_for(other)
            if name in others:
                return True
            pending.extend(others - seen)
        return False

    def _settle_obligations(self, name: str, activation: Activation, text: str, succeeded: bool):
        """Answer requests and close tasks the agent did not settle explicitly.

        A member that ends its turn while waiting on others (it asked someone, delegated a
        task or started a protocol) still owes its answer: the obligation carries over to
        the activation in which it is no longer waiting. Failures answer at once so nobody
        waits on a broken agent.
        """
        data = self.state.data
        record = self.state.agents[name]
        owed = list(record.get("owed", []))
        owed += [item["ref"] for item in activation.items if item["kind"] == "request"]
        owed_tasks = list(record.get("owed_tasks", []))
        owed_tasks += [
            item["ref"] for item in activation.items if item["kind"] in {"task", "review"}
        ]
        owed = [
            message_id
            for message_id in dict.fromkeys(owed)
            if message_id not in activation.replied
            and name not in data["messages"].get(message_id, {"replied_by": [name]})["replied_by"]
        ]
        owed_tasks = [
            task_id
            for task_id in dict.fromkeys(owed_tasks)
            if task_id not in activation.updated
            and data["tasks"].get(task_id, {}).get("status") not in TERMINAL_TASK | {None}
            and data["tasks"][task_id]["assignee"] == name
        ]
        waits = self.waits_for(name) if succeeded else set()
        # Unread items mean another activation follows at once; settle after reading them.
        unread = succeeded and bool(self._wakeable(name))
        if waits and not unread and (owed or owed_tasks) and self._closes_cycle(name, waits):
            # Deferring would deadlock (A waits on B while B waits on A): answer now.
            self._event("wait_cycle_broken", name, waiting_on=sorted(waits))
        elif waits or unread:
            record["owed"], record["owed_tasks"] = owed, owed_tasks
            if owed or owed_tasks:
                self._event("obligations_deferred", name, requests=owed, tasks=owed_tasks)
            return
        record["owed"], record["owed_tasks"] = [], []
        for message_id in owed:
            message = data["messages"][message_id]
            self._post(
                name,
                [message["sender"]],
                text,
                kind="reply",
                reply_to=message_id,
                hop=message.get("hop", 0) + 1,
                depth=message.get("depth", 0),
                auto=True,
            )
        for task_id in owed_tasks:
            task = data["tasks"][task_id]
            if succeeded:
                self._task_status(task, name, "completed", text, auto=True)
            else:
                self._task_status(task, name, "blocked", text, auto=True)

    def _settle_protocol(
        self, name: str, activation: Activation, item: dict, text: str, succeeded: bool
    ):
        """Protocol rounds never wait: the final message is the agent's contribution."""
        key = f"{item['ref']}:{item.get('round')}:{item.get('step')}"
        record = self.state.data["protocols"].get(item["ref"])
        if key in activation.contributed or record is None:
            return
        step = item.get("step")
        if step == "contribute" and record["status"] == "running":
            if item.get("round") == record["round"]:
                self.protocols.contribute(
                    name,
                    record["id"],
                    text if succeeded else None,
                    auto=True,
                    absent=None if succeeded else text,
                )
        elif step == "conclude" and record["status"] == "concluding":
            self.protocols.conclude(name, record["id"], text, [], auto=True)

    # Shared helpers -----------------------------------------------------------------

    def _require_open(self) -> None:
        if self.state is None:
            raise TeamError("team_unavailable", "The team session is not open")

    def _event(self, event_type: str, actor: str, **data: Any) -> None:
        event = self.state.event(event_type, actor, **data)
        if self.store is not None:
            # The event tail in the state file stays authoritative if the log is unwritable.
            with suppress(OSError):
                self.store.append_event(event)

    def _persist(self) -> None:
        if self.store is not None and self.state is not None:
            self.store.save(self.state.data)

    def _progress(self, name: str, **event: Any) -> None:
        if self.on_progress is None:
            return
        # Display callbacks never change execution, as with display_callback elsewhere.
        with suppress(Exception):
            self.on_progress(
                {
                    "phase": "subagent",
                    "agent_id": name,
                    "title": event.pop("title", None) or name,
                    "task": name,
                    "mode": "team",
                    **event,
                }
            )

    def progress(self, name: str, **event: Any) -> None:
        """Model and tool activity reported by a running member (any thread)."""
        activation = self.activations.get(name)
        if activation is not None and activation.cancelled.is_set():
            return
        self._progress(name, state="running", **event)

    def _origin(self, actor: str, activation):
        return activation if activation is not None else self.main_origin

    def _post(
        self,
        sender,
        recipients,
        content,
        *,
        kind,
        reply_to=None,
        group=None,
        hop,
        depth,
        auto=False,
    ):
        message = self.state.add_message(
            sender=sender,
            recipients=list(recipients),
            group=group,
            kind=kind,
            content=content,
            reply_to=reply_to,
            hop=hop,
            depth=depth,
            auto=auto,
        )
        if reply_to is not None:
            original = self.state.data["messages"].get(reply_to)
            if original is not None and sender not in original["replied_by"]:
                original["replied_by"].append(sender)
        self._event(
            "message_sent",
            sender,
            message_id=message["id"],
            to=list(recipients),
            kind=kind,
            reply_to=reply_to,
            auto=auto,
            preview=excerpt(content, 160),
        )
        for name in recipients:
            self.deliver(
                name,
                {
                    "kind": kind,
                    "ref": message["id"],
                    "sender": sender,
                    "hop": hop,
                    "depth": depth,
                    "created_at": now(),
                },
            )
        return message

    def _recipients(self, actor: str, to: list[str]) -> tuple[list[str], str | None]:
        names: list[str] = []
        group_id = None
        for target in to:
            if target == "all":
                if not self.team.collaboration.broadcast:
                    raise TeamError("broadcast_disabled", "Broadcast is disabled; address members")
                names.extend(
                    spec.name
                    for spec in self.member_specs()
                    if spec.enabled
                    and self.state.agents.get(spec.name, {}).get("status") != "removed"
                )
                continue
            if target.startswith("group:") or target in self.state.data["groups"]:
                group_id = target.removeprefix("group:")
                group = self.state.group(group_id)
                if group is None:
                    raise TeamError("unknown_group", f"Unknown group {group_id!r}")
                if actor not in group["members"]:
                    raise TeamError("not_in_group", f"{actor} is not a member of {group_id}")
                names.extend(group["members"])
                continue
            self.require_member(target)
            names.append(target)
        recipients = [name for name in dict.fromkeys(names) if name != actor]
        if not recipients:
            raise TeamError("invalid_arguments", "No recipients other than yourself")
        return recipients, group_id

    def _task_status(self, task: dict, actor: str, status: str, text: str | None, *, auto=False):
        task["status"] = status
        task["updated_at"] = now()
        if status == "completed":
            task["result"] = text
            task["auto_completed"] = auto
        elif text:
            task["notes"].append(text)
        self._event("task_updated", actor, task_id=task["id"], status=status, auto=auto)
        if status in TERMINAL_TASK | {"blocked"} and task["creator"] != actor:
            self.deliver(
                task["creator"],
                {
                    "kind": "task_update",
                    "ref": task["id"],
                    "sender": actor,
                    "hop": task.get("hop", task["depth"]),
                    "depth": task["depth"],
                    "created_at": now(),
                },
            )
        if status in TERMINAL_TASK:
            self._release_dependents(task)

    def _release_dependents(self, finished: dict) -> None:
        tasks = self.state.data["tasks"]
        for task in tasks.values():
            if task["status"] != "waiting" or finished["id"] not in task["depends_on"]:
                continue
            states = [tasks[dependency]["status"] for dependency in task["depends_on"]]
            if any(state in {"failed", "cancelled"} for state in states):
                self._task_status(
                    task, "runtime", "blocked", f"Dependency {finished['id']} {finished['status']}"
                )
            elif all(state == "completed" for state in states):
                self._assign(task)

    def _assign(self, task: dict) -> None:
        task["status"] = "assigned" if task["assignee"] else "open"
        if task["assignee"]:
            self.deliver(
                task["assignee"],
                {
                    "kind": "review" if task["kind"] == "review" else "task",
                    "ref": task["id"],
                    "sender": task["creator"],
                    "hop": task.get("hop", task["depth"]),
                    "depth": task["depth"],
                    "created_at": now(),
                },
            )

    def _message_budget(self) -> None:
        if self.state.data["turn"]["messages"] >= self.limits.max_messages_per_turn:
            raise TeamError(
                "message_budget",
                f"The team reached {self.limits.max_messages_per_turn} messages this turn. "
                "Summarize your results in your final message instead of sending more.",
            )

    # Budgets called by members ------------------------------------------------------

    def charge_tokens(self, name: str, tokens: int, *, before: bool = False) -> None:
        with self.lock:
            record = self.state.agents[name]["turn"]
            if before:
                if record["tokens"] >= self.limits.token_budget_per_agent:
                    raise TeamError("token_budget", f"{name} exhausted its token budget this turn")
                return
            record["tokens"] += max(0, int(tokens))
            record["model_requests"] += 1

    def charge_tool_call(self, name: str) -> None:
        with self.lock:
            record = self.state.agents[name]["turn"]
            if record["tool_calls"] >= self.limits.max_tool_calls_per_agent:
                raise TeamError(
                    "tool_budget",
                    f"Tool-call budget ({self.limits.max_tool_calls_per_agent}) exhausted this turn. "
                    "Stop calling tools and end your turn with a summary of what you have.",
                )
            record["tool_calls"] += 1

    def render_activation(self, activation: Activation) -> str:
        with self.lock:
            return activation_text(self, activation.agent, activation.items)

    def context_messages(self, name: str) -> list[dict]:
        return self.store.load_context(name)

    def save_context(self, name: str, messages: list[dict]) -> None:
        self.store.save_context(name, messages)

    # Actions (the team tool) --------------------------------------------------------

    def act_list_members(self, actor, activation, *, _context):
        with self.lock:
            self._require_open()
            return {
                "members": [
                    {
                        "name": spec.name,
                        "description": spec.description,
                        "enabled": spec.enabled,
                        "status": self.state.agents.get(spec.name, {}).get("status"),
                        "tools": sorted(self.member_tool_names(spec)),
                        "skills": self.skill_names(spec),
                        "dynamic": spec.dynamic,
                    }
                    for spec in self.member_specs()
                ]
                + [{"name": MAIN, "description": "Talks to the user; integrates final results."}]
            }

    def act_get_state(self, actor, activation, *, _context):
        with self.lock:
            self._require_open()
            return self.state.view(actor)

    def act_send_message(
        self, actor, activation, *, _context, content, to=None, kind="message", reply_to=None
    ):
        with self.lock:
            self._require_open()
            origin = self._origin(actor, activation)
            original = None
            if reply_to is not None:
                original = self.state.visible_message(reply_to, actor)
                if original is None:
                    raise TeamError("not_found", f"No message {reply_to!r} visible to {actor}")
            if not to:
                if original is None:
                    raise TeamError("invalid_arguments", "send_message needs `to` unless replying")
                to = [original["sender"]]
            recipients, group = self._recipients(actor, to)
            isolated = self.protocols.isolated(actor, recipients)
            if isolated is not None:
                raise TeamError(
                    "isolated",
                    f"{isolated} is in a blind round: work independently and contribute before "
                    "talking to other participants.",
                )
            self._message_budget()
            hop = origin.hop + 1
            # Replies close a chain, so only new messages count toward the hop limit.
            if original is None and hop > self.limits.max_message_hops:
                raise TeamError(
                    "hop_limit",
                    f"This conversation chain reached {self.limits.max_message_hops} hops. "
                    "Stop forwarding; answer with what you have in your final message.",
                )
            message_kind = "reply" if original is not None else kind
            busy = [name for name in recipients if name in self.activations]
            message = self._post(
                actor,
                recipients,
                content,
                kind=message_kind,
                reply_to=reply_to,
                group=group,
                hop=hop,
                depth=origin.depth,
            )
            if original is not None:
                origin.replied.add(original["id"])
            self._persist()
            return {
                "message_id": message["id"],
                "delivered_to": recipients,
                "recipients_busy": busy,
                "note": "Busy recipients read it when their current turn ends."
                if busy
                else "Delivered.",
            }

    def act_wait(self, actor, activation, *, _context, timeout=120):
        deadline = min(time.monotonic() + timeout, _context.deadline)
        with self.changed:
            self._require_open()
            while True:
                mailbox = self.state.mailbox(MAIN)
                if mailbox:
                    items, mailbox[:] = list(mailbox), []
                    self._persist()
                    return {
                        "reason": "arrived",
                        "items": [self._main_item(item) for item in items],
                        "team": self._summary(),
                    }
                if self._quiescent():
                    return {
                        "reason": "idle",
                        "items": [],
                        "team": self._summary(),
                        "note": "No member is working and nothing is queued for you.",
                    }
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return {
                        "reason": "timeout",
                        "items": [],
                        "team": self._summary(),
                        "note": "Members are still working; wait again or continue meanwhile.",
                    }
                _context.remaining()
                self.changed.wait(min(0.25, remaining))

    def _quiescent(self) -> bool:
        if self.activations:
            return False
        return not any(
            self._wakeable(name)
            and self.state.agents[name]["status"] not in {"disabled", "removed", "exhausted"}
            for name in self.state.agents
            if name != MAIN
        )

    def _summary(self) -> dict:
        return {
            "working": sorted(self.activations),
            "queued": sorted(
                name
                for name, record in self.state.agents.items()
                if name not in self.activations
                and self._wakeable(name)
                and record["status"] not in {"disabled", "removed", "exhausted"}
            ),
            "open_tasks": [
                task["id"]
                for task in self.state.data["tasks"].values()
                if task["status"] not in TERMINAL_TASK
            ],
            "running_protocols": [
                record["id"]
                for record in self.state.data["protocols"].values()
                if record["status"] in {"running", "concluding"}
            ],
        }

    def _main_item(self, item: dict) -> dict:
        data = self.state.data
        kind = item["kind"]
        if kind in {"message", "request", "reply"}:
            message = data["messages"].get(item["ref"], {})
            return {
                "type": kind,
                **{
                    key: message.get(key)
                    for key in ("id", "sender", "content", "reply_to", "group", "auto")
                },
            }
        if kind in {"task", "task_update", "review"}:
            task = data["tasks"].get(item["ref"], {})
            return {
                "type": kind,
                **{
                    key: task.get(key)
                    for key in (
                        "id",
                        "title",
                        "status",
                        "assignee",
                        "creator",
                        "result",
                        "notes",
                        "auto_completed",
                    )
                },
            }
        if kind == "protocol":
            return {
                "type": "protocol",
                "step": item.get("step"),
                "text": self.protocols.item_text(MAIN, item),
            }
        return {"type": kind, **{key: value for key, value in item.items() if key != "kind"}}

    def act_create_group(self, actor, activation, *, _context, name, members, topic=""):
        with self.lock:
            self._require_open()
            if len(self.state.data["groups"]) >= GROUP_LIMIT:
                raise TeamError("group_limit", f"At most {GROUP_LIMIT} groups per session")
            for member in members:
                self.require_member(member)
            group = {
                "id": self.state.next_id("g"),
                "name": name,
                "members": list(dict.fromkeys([actor, *members])),
                "creator": actor,
                "topic": topic,
                "created_at": now(),
            }
            self.state.data["groups"][group["id"]] = group
            self._event("group_created", actor, group_id=group["id"], members=group["members"])
            self._persist()
            return {**group, "address": f"group:{group['id']}"}

    def act_create_task(
        self, actor, activation, *, _context, title, description="", assignee=None, depends_on=None
    ):
        with self.lock:
            self._require_open()
            return self._create_task(
                actor,
                self._origin(actor, activation),
                title,
                description,
                assignee,
                depends_on or [],
            )

    def _create_task(
        self, actor, origin, title, description, assignee, depends_on, *, kind="task", subject=None
    ):
        depth = origin.depth + 1
        if depth > self.limits.max_task_depth:
            raise TeamError(
                "task_depth",
                f"Task nesting is limited to {self.limits.max_task_depth} levels. Do the work "
                "yourself or report back to the task's creator.",
            )
        if assignee is not None:
            self.require_member(assignee)
        tasks = self.state.data["tasks"]
        unknown = [task_id for task_id in depends_on if task_id not in tasks]
        if unknown:
            raise TeamError("not_found", f"Unknown dependencies: {', '.join(unknown)}")
        task = {
            "id": self.state.next_id("t"),
            "title": title,
            "description": description,
            "kind": kind,
            "subject": subject,
            "creator": actor,
            "assignee": assignee,
            "status": "waiting",
            "depends_on": list(dict.fromkeys(depends_on)),
            "result": None,
            "notes": [],
            "depth": depth,
            "hop": origin.hop + 1,
            "auto_completed": False,
            "created_at": now(),
            "updated_at": now(),
        }
        tasks[task["id"]] = task
        self._event(
            "task_created",
            actor,
            task_id=task["id"],
            assignee=assignee,
            depends_on=task["depends_on"],
            kind=kind,
        )
        states = [tasks[dependency]["status"] for dependency in task["depends_on"]]
        if any(state in {"failed", "cancelled"} for state in states):
            task["status"] = "blocked"
        elif all(state == "completed" for state in states):
            self._assign(task)
        self._persist()
        return {
            key: task[key] for key in ("id", "title", "assignee", "status", "depends_on", "depth")
        }

    def act_update_task(self, actor, activation, *, _context, task_id, status, result=None):
        with self.lock:
            self._require_open()
            task = self.state.data["tasks"].get(task_id)
            if task is None:
                raise TeamError("not_found", f"Unknown task {task_id!r}")
            if task["status"] in TERMINAL_TASK:
                raise TeamError("task_closed", f"{task_id} is already {task['status']}")
            if task["assignee"] is None and status == "in_progress":
                task["assignee"] = actor
            elif actor != task["assignee"] and not (
                actor == task["creator"] and status == "cancelled"
            ):
                raise TeamError(
                    "not_assignee",
                    f"{task_id} is assigned to {task['assignee']}; only the assignee updates it",
                )
            self._task_status(task, actor, status, result)
            if activation is not None:
                activation.updated.add(task_id)
            self._persist()
            return {key: task[key] for key in ("id", "title", "status", "assignee", "result")}

    def act_get_task(self, actor, activation, *, _context, task_id):
        with self.lock:
            self._require_open()
            task = self.state.data["tasks"].get(task_id)
            if task is None:
                raise TeamError("not_found", f"Unknown task {task_id!r}")
            return dict(task)

    def act_share_artifact(
        self, actor, activation, *, _context, title, content, visibility="team", group=None
    ):
        with self.lock:
            self._require_open()
            if visibility == "group":
                record = self.state.group(group or "")
                if record is None or actor not in record["members"]:
                    raise TeamError("not_in_group", "Group artifacts need a group you belong to")
            artifact = {
                "id": self.state.next_id("a"),
                "title": title,
                "content": content,
                "author": actor,
                "visibility": visibility,
                "group": group if visibility == "group" else None,
                "kind": "shared",
                "created_at": now(),
            }
            self.state.data["artifacts"][artifact["id"]] = artifact
            self._event(
                "artifact_shared",
                actor,
                artifact_id=artifact["id"],
                visibility=visibility,
                title=title,
            )
            self._persist()
            return {key: artifact[key] for key in ("id", "title", "visibility", "group")}

    def act_get_artifact(self, actor, activation, *, _context, artifact_id):
        with self.lock:
            self._require_open()
            artifact = self.state.data["artifacts"].get(artifact_id)
            if artifact is None or not self.state.artifact_visible(artifact, actor):
                raise TeamError("not_found", f"No artifact {artifact_id!r} visible to {actor}")
            return dict(artifact)

    def act_request_review(
        self, actor, activation, *, _context, subject, reviewers, instructions=""
    ):
        with self.lock:
            self._require_open()
            data = self.state.data
            if subject in data["tasks"]:
                target = data["tasks"][subject]
                reference = f'task {subject} "{target["title"]}": {target.get("result") or target.get("description") or ""}'
            elif subject in data["artifacts"] and self.state.artifact_visible(
                data["artifacts"][subject], actor
            ):
                artifact = data["artifacts"][subject]
                if artifact["visibility"] != "team":
                    raise TeamError(
                        "not_shared", "Share the artifact with the team before requesting a review"
                    )
                reference = f'artifact {subject} "{artifact["title"]}" (read it with get_artifact)'
            else:
                raise TeamError("not_found", f"No task or visible artifact {subject!r}")
            origin = self._origin(actor, activation)
            created = [
                self._create_task(
                    actor,
                    origin,
                    f"Review {subject}",
                    f"Review {reference}\n\n{instructions}".strip(),
                    reviewer,
                    [],
                    kind="review",
                    subject=subject,
                )
                for reviewer in dict.fromkeys(reviewers)
            ]
            return {"reviews": created}

    def act_record_decision(self, actor, activation, *, _context, title, decision, rationale=""):
        with self.lock:
            self._require_open()
            record = {
                "id": self.state.next_id("d"),
                "title": title,
                "decision": decision,
                "rationale": rationale,
                "by": actor,
                "participants": [actor],
                "source": None,
                "created_at": now(),
            }
            self.state.data["decisions"].append(record)
            self._event("decision_recorded", actor, decision_id=record["id"], title=title)
            self._persist()
            return record

    def act_start_protocol(
        self,
        actor,
        activation,
        *,
        _context,
        protocol,
        topic,
        participants,
        rounds=None,
        goal="",
        stances=None,
        facilitator=None,
    ):
        with self.lock:
            self._require_open()
            self._message_budget()
            result = self.protocols.start(
                actor,
                self._origin(actor, activation),
                protocol=protocol,
                topic=topic,
                participants=participants,
                rounds=rounds,
                goal=goal,
                stances=stances,
                facilitator=facilitator,
            )
            self._persist()
            return result

    def act_contribute(self, actor, activation, *, _context, protocol_id, content, ready=False):
        with self.lock:
            self._require_open()
            record = self.protocols.get(protocol_id)
            current_round = record["round"]
            result = self.protocols.contribute(actor, protocol_id, content, ready)
            if activation is not None:
                activation.contributed.add(f"{protocol_id}:{current_round}:contribute")
            self._persist()
            return result

    def act_conclude(self, actor, activation, *, _context, protocol_id, summary, decisions=None):
        with self.lock:
            self._require_open()
            result = self.protocols.conclude(actor, protocol_id, summary, decisions)
            if activation is not None:
                record = self.protocols.get(protocol_id)
                activation.contributed.add(f"{protocol_id}:{record['round']}:conclude")
            self._persist()
            return result

    def act_load_skill(self, actor, activation, *, _context, name):
        with self.lock:
            spec = self.member_spec(actor) if actor != MAIN else None
            allowed = (
                self.skill_names(spec)
                if spec is not None
                else [skill.name for skill in (self.skills.skills if self.skills else [])]
            )
        if name not in allowed:
            raise TeamError(
                "skill_not_granted",
                f"Skill {name!r} is not granted to {actor}; granted: {', '.join(allowed) or 'none'}",
            )
        skill = self.skills.find(name)
        return {"skill": name, "instructions": skill_block(skill)}

    def act_spawn_agent(
        self, actor, activation, *, _context, name, description, instructions="", tools=None
    ):
        with self.lock:
            self._require_open()
            if not self.team.collaboration.dynamic_agents:
                raise TeamError(
                    "dynamic_agents_disabled", "This environment does not allow spawning agents"
                )
            if len(self.member_specs()) >= self.limits.max_agents:
                raise TeamError(
                    "agent_limit", f"The team already has {self.limits.max_agents} members"
                )
            if self.member_spec(name) is not None or name in self.state.agents:
                raise TeamError("name_taken", f"Member {name!r} already exists")
            # A spawned agent never exceeds its creator's permissions.
            ceiling = (
                self.member_tool_names(self.member_spec(actor))
                if actor != MAIN
                else frozenset(self.base_tools.registry) - {"ask_user", "monitor", "update_plan"}
            )
            requested = set(tools) if tools is not None else {"read", "grep", "find", "ls"}
            granted = sorted((requested & ceiling) - {"team", "mcp", "delegate"})
            definition = {
                "description": description,
                "instructions": instructions,
                "tools": granted,
            }
            try:
                parse_agent(name, definition, read_prompt=None, tool_names=frozenset(granted))
            except ValueError as exc:
                raise TeamError("invalid_arguments", str(exc)) from exc
            self.state.agents[name] = agent_record(dynamic={**definition, "spawned_by": actor})
            self.state.mailbox(name)
            self._event("agent_spawned", actor, agent=name, tools=granted)
            self._persist()
            return {"name": name, "tools": granted, "denied": sorted(requested - set(granted))}

    def act_cancel(self, actor, activation, *, _context, reason="cancelled by main"):
        with self.lock:
            self._require_open()
            running = list(self.activations.values())
            for item in running:
                item.cancelled.set()
            dropped = 0
            for name in self.state.agents:
                mailbox = self.state.mailbox(name)
                dropped += len(mailbox)
                mailbox.clear()
            for task in self.state.data["tasks"].values():
                if task["status"] not in TERMINAL_TASK:
                    task["status"] = "cancelled"
                    task["notes"].append(reason)
            self.protocols.cancel_all(actor, reason)
            self._event(
                "team_cancelled", actor, reason=reason, running=len(running), dropped=dropped
            )
            self._persist()
        self._stop_activations()
        with self.lock:
            # Cancelled activations requeue their items; a cancel discards them instead.
            for name in self.state.agents:
                self.state.mailbox(name).clear()
            self._persist()
            return {"cancelled_activations": len(running), "dropped_items": dropped}

    # Main-agent integration --------------------------------------------------------

    def final_answer_problem(self) -> str | None:
        with self.lock:
            if self.state is None or self.nudges >= 1:
                return None
            summary = self._summary()
            if not summary["working"] and not summary["queued"]:
                return None
            self.nudges += 1
            return (
                "Team members are still working ("
                + ", ".join(summary["working"] + summary["queued"])
                + "). Call team action=wait to collect their results, or team action=cancel to "
                "stop them, before giving your final answer."
            )

    def main_instructions(self) -> str:
        from team_tools import main_instructions

        with self.lock:
            return main_instructions(self)

    def turn_context(self) -> str:
        with self.lock:
            if self.state is None:
                return ""
            summary = self._summary()
            unread = len(self.state.mailbox(MAIN))
            members = ", ".join(spec.name for spec in self.member_specs() if spec.enabled) or "none"
            parts = [f"Team session: members {members}; unread for main: {unread}."]
            if summary["open_tasks"]:
                parts.append("Open tasks: " + ", ".join(summary["open_tasks"]) + ".")
            if summary["queued"]:
                parts.append("Queued work resumes now: " + ", ".join(summary["queued"]) + ".")
            if summary["running_protocols"]:
                parts.append("Running protocols: " + ", ".join(summary["running_protocols"]) + ".")
            if self.notices:
                parts.append("Recovered: " + "; ".join(self.notices) + ".")
                self.notices = []
            return " ".join(parts) + " Use team action=get_state or wait for details."

    def status_lines(self) -> list[str]:
        with self.lock:
            if self.state is None:
                return ["团队会话未打开"]
            data = self.state.data
            lines = [
                f"团队会话：第 {data['turn']['number']} 轮 · 本轮消息 {data['turn']['messages']}"
                f"/{self.limits.max_messages_per_turn} · 主代理未读 {len(self.state.mailbox(MAIN))}",
            ]
            for name, record in data["agents"].items():
                turn = record["turn"]
                lines.append(
                    f"  成员 {name} · {record['status']} · 待处理 {len(self.state.mailbox(name))}"
                    f" · 本轮激活 {turn['activations']} · 工具 {turn['tool_calls']}"
                    f" · 约 {turn['tokens']} tokens"
                    + (" · 动态成员" if "dynamic" in record else "")
                )
            for task in data["tasks"].values():
                lines.append(
                    f"  任务 {task['id']} · {task['status']} · {task['assignee'] or '未分配'}"
                    f" · {task['title']}"
                )
            for group in data["groups"].values():
                lines.append(
                    f"  群组 {group['id']} {group['name']} · {', '.join(group['members'])}"
                )
            for record in data["protocols"].values():
                lines.append(
                    f"  协议 {record['id']} {record['protocol']} · {record['status']}"
                    f" · 第 {record['round']}/{record['rounds']} 轮 · {record['topic']}"
                )
            if data["artifacts"]:
                lines.append(
                    f"  共享产物 {len(data['artifacts'])} 个；决策 {len(data['decisions'])} 条"
                )
            return lines

    def event_lines(self, limit: int) -> list[str]:
        with self.lock:
            if self.state is None:
                return []
            lines = []
            for event in self.state.data["events"][-limit:]:
                details = {
                    key: value
                    for key, value in event.items()
                    if key not in {"id", "at", "type", "actor"} and value not in (None, [], {})
                }
                stamp = time.strftime("%H:%M:%S", time.localtime(event["at"]))
                lines.append(f"{stamp} {event['actor']} {event['type']} {details}")
            return lines

    def execution_context(self, activation: Activation) -> ExecutionContext:
        return ExecutionContext(activation.deadline, activation.cancelled)
