"""Optional structured collaboration protocols.

One engine runs every protocol as rounds of contributions over the same mailboxes,
artifacts and decisions that informal collaboration uses. Presets differ only in round
count, which rounds are blind (participants cannot see or message each other until the
round completes) and whether a facilitator concludes.
"""

from typing import TYPE_CHECKING

from team_state import MAIN, TeamError, excerpt, now

if TYPE_CHECKING:
    from team_runtime import TeamRuntime

PRESETS = {
    "meeting": {
        "rounds": 2,
        "blind_rounds": 0,
        "conclude": True,
        "style": "Discuss the topic toward the goal. Build on others' points, state "
        "disagreements explicitly, and propose concrete next steps.",
    },
    "debate": {
        "rounds": 3,
        "blind_rounds": 0,
        "conclude": True,
        "style": "Debate. Take and defend a clear position with evidence. In later rounds, "
        "steelman the strongest opposing argument before rebutting it; concede what is right.",
    },
    "brainstorm": {
        "rounds": 2,
        "blind_rounds": 1,
        "conclude": True,
        "style": "Round 1: propose 3-7 distinct ideas independently, one line each on why they "
        "could work. Later rounds: combine, extend and critique ideas against the goal.",
    },
    "independent_research": {
        "rounds": 1,
        "blind_rounds": 1,
        "conclude": False,
        "style": "Research the question independently with your own tools. Report findings with "
        "evidence, sources and confidence. Do not coordinate with other participants.",
    },
    "review_decision": {
        "rounds": 1,
        "blind_rounds": 1,
        "conclude": True,
        "style": "Review the proposal independently. Give a verdict (approve, approve with "
        "changes, reject), the key reasons with evidence, and required changes.",
    },
    "coordinate_work": {
        "rounds": 1,
        "blind_rounds": 0,
        "conclude": True,
        "style": "Say which parts of the work you will take on, what you need from others and "
        "the dependencies. The facilitator turns the plan into tasks.",
    },
}
TRANSCRIPT_LIMIT = 24000


class ProtocolEngine:
    def __init__(self, runtime: "TeamRuntime"):
        self.runtime = runtime

    @property
    def state(self):
        return self.runtime.state

    # Start ---------------------------------------------------------------------------

    def start(
        self, actor, origin, *, protocol, topic, participants, rounds, goal, stances, facilitator
    ):
        runtime = self.runtime
        if protocol not in runtime.team.collaboration.protocols:
            raise TeamError(
                "protocol_disabled", f"Protocol {protocol!r} is disabled in this environment"
            )
        preset = PRESETS[protocol]
        names = list(dict.fromkeys(participants))
        for name in names:
            runtime.require_member(name, allow_main=False)
        if not names:
            raise TeamError("invalid_arguments", "A protocol needs at least one participant")
        if protocol in {"meeting", "debate"} and len(names) < 2:
            raise TeamError("invalid_arguments", f"A {protocol} needs at least two participants")
        facilitator = facilitator or actor
        if facilitator != MAIN:
            runtime.require_member(facilitator, allow_main=False)
        limit = runtime.limits.max_meeting_rounds
        count = min(rounds or preset["rounds"], limit)
        stances = dict(stances or {})
        unknown = set(stances) - set(names)
        if unknown:
            raise TeamError(
                "invalid_arguments", f"Stances for non-participants: {', '.join(sorted(unknown))}"
            )
        record = {
            "id": self.state.next_id("p"),
            "protocol": protocol,
            "topic": topic,
            "goal": goal or "",
            "initiator": actor,
            "facilitator": facilitator,
            "participants": names,
            "rounds": count,
            "blind_rounds": min(preset["blind_rounds"], count),
            "conclude": preset["conclude"],
            "round": 1,
            "status": "running",
            "contributions": {},
            "stances": stances,
            "outcome": None,
            "hop": origin.hop,
            "depth": origin.depth,
            "created_at": now(),
        }
        self.state.data["protocols"][record["id"]] = record
        self.state.event(
            "protocol_started",
            actor,
            protocol_id=record["id"],
            protocol=protocol,
            participants=names,
            rounds=count,
        )
        self._open_round(record)
        return self.summary(record)

    def _open_round(self, record):
        record["contributions"].setdefault(str(record["round"]), {})
        for name in record["participants"]:
            self.runtime.deliver(name, self._item(record, "contribute"))

    def _item(self, record, step):
        return {
            "kind": "protocol",
            "ref": record["id"],
            "round": record["round"],
            "step": step,
            "sender": record["initiator"],
            "hop": record["hop"],
            "depth": record["depth"],
            "created_at": now(),
        }

    # Contributions ---------------------------------------------------------------------

    def get(self, protocol_id):
        record = self.state.data["protocols"].get(protocol_id)
        if record is None:
            raise TeamError("not_found", f"Unknown protocol {protocol_id!r}")
        return record

    def contribute(self, actor, protocol_id, content, ready=False, *, auto=False, absent=None):
        record = self.get(protocol_id)
        if actor not in record["participants"]:
            raise TeamError("not_participant", f"{actor} is not a participant of {protocol_id}")
        if record["status"] != "running":
            raise TeamError(
                "protocol_closed", f"{protocol_id} is {record['status']}; contributions are closed"
            )
        current = record["contributions"].setdefault(str(record["round"]), {})
        if auto and actor in current:
            return self.summary(record)
        current[actor] = {
            "content": content,
            "ready": bool(ready),
            "auto": auto,
            **({"absent": absent} if absent else {}),
            "at": now(),
        }
        self.state.event(
            "protocol_contribution",
            actor,
            protocol_id=protocol_id,
            round=record["round"],
            auto=auto,
            absent=absent,
        )
        if all(name in current for name in record["participants"]):
            self._advance(record)
        return self.summary(record)

    def _advance(self, record):
        current = record["contributions"][str(record["round"])]
        consensus = all(entry.get("ready") for entry in current.values())
        blind = record["round"] <= record["blind_rounds"]
        if record["round"] >= record["rounds"] or (consensus and not blind):
            if record["conclude"]:
                record["status"] = "concluding"
                self.state.event(
                    "protocol_concluding", record["facilitator"], protocol_id=record["id"]
                )
                self.runtime.deliver(record["facilitator"], self._item(record, "conclude"))
            else:
                # No conclusion step: the runtime compiles the findings for the initiator.
                self._finish(record, actor="runtime", summary=None, decisions=[])
            return
        record["round"] += 1
        self.state.event(
            "protocol_round", record["initiator"], protocol_id=record["id"], round=record["round"]
        )
        self._open_round(record)

    def absent(self, actor, item, reason):
        """Record a non-contribution so a protocol never waits on an agent that cannot run."""
        record = self.state.data["protocols"].get(item["ref"])
        if record is None or record["status"] not in {"running", "concluding"}:
            return
        if item.get("step") == "contribute" and item.get("round") == record["round"]:
            self.contribute(actor, record["id"], None, auto=True, absent=reason)
        elif item.get("step") == "conclude" and record["status"] == "concluding":
            self._finish(
                record, actor=actor, summary=f"(not concluded: {reason})", decisions=[], auto=True
            )

    # Conclusion ------------------------------------------------------------------------

    def conclude(self, actor, protocol_id, summary, decisions=None, *, auto=False):
        record = self.get(protocol_id)
        if actor not in {record["facilitator"], record["initiator"]}:
            raise TeamError(
                "not_facilitator", f"Only the facilitator or initiator can conclude {protocol_id}"
            )
        if record["status"] not in {"running", "concluding"}:
            if auto:
                return self.summary(record)
            raise TeamError("protocol_closed", f"{protocol_id} is already {record['status']}")
        self._finish(
            record, actor=actor, summary=summary, decisions=list(decisions or []), auto=auto
        )
        return self.summary(record)

    def _finish(self, record, *, actor, summary, decisions, auto=False):
        state = self.state
        record["status"] = "concluded"
        transcript = self.transcript(record, viewer=None)
        artifact = {
            "id": state.next_id("a"),
            "title": f"{record['protocol']} {record['id']}: {record['topic']}",
            "content": (f"Summary:\n{summary}\n\n" if summary else "") + transcript,
            "author": actor,
            "visibility": "team",
            "group": None,
            "kind": "protocol_outcome",
            "created_at": now(),
        }
        state.data["artifacts"][artifact["id"]] = artifact
        recorded = []
        for text in decisions:
            decision = {
                "id": state.next_id("d"),
                "title": record["topic"],
                "decision": text,
                "rationale": excerpt(summary or "", 2000),
                "by": actor,
                "participants": record["participants"],
                "source": record["id"],
                "created_at": now(),
            }
            state.data["decisions"].append(decision)
            recorded.append(decision["id"])
        record["outcome"] = {
            "summary": summary,
            "decisions": recorded,
            "artifact": artifact["id"],
            "auto": auto,
            "concluded_by": actor,
        }
        state.event(
            "protocol_concluded",
            actor,
            protocol_id=record["id"],
            artifact=artifact["id"],
            decisions=recorded,
            auto=auto,
        )
        # The initiator asked for the outcome; participants only need to know it exists.
        if record["initiator"] != actor:
            self.runtime.deliver(record["initiator"], self._item(record, "outcome"))
        for name in record["participants"]:
            if name not in {record["initiator"], actor}:
                self.runtime.deliver(
                    name,
                    {
                        "kind": "notice",
                        "text": f'{record["protocol"]} {record["id"]} on "{record["topic"]}" '
                        f"concluded; outcome in artifact {artifact['id']}.",
                        "created_at": now(),
                    },
                )

    def cancel_all(self, actor, reason):
        for record in self.state.data["protocols"].values():
            if record["status"] in {"running", "concluding"}:
                record["status"] = "cancelled"
                record["outcome"] = {"summary": f"cancelled: {reason}", "decisions": []}
                self.state.event("protocol_cancelled", actor, protocol_id=record["id"])

    # Isolation and rendering ----------------------------------------------------------

    def isolated(self, actor, recipients) -> str | None:
        for record in self.state.data["protocols"].values():
            if (
                record["status"] == "running"
                and record["round"] <= record["blind_rounds"]
                and actor in record["participants"]
                and any(name in record["participants"] for name in recipients if name != actor)
            ):
                return record["id"]
        return None

    def transcript(self, record, viewer) -> str:
        lines = []
        blind = record["status"] == "running" and record["round"] <= record["blind_rounds"]
        for number in range(1, record["round"] + 1):
            contributions = record["contributions"].get(str(number), {})
            current = number == record["round"] and record["status"] == "running"
            if current and (blind or viewer is None):
                contributions = {}
            for name, entry in contributions.items():
                if entry.get("absent"):
                    lines.append(f"[round {number}] {name}: (absent: {entry['absent']})")
                    continue
                marker = " (ready to conclude)" if entry.get("ready") else ""
                lines.append(f"[round {number}] {name}{marker}:\n{entry.get('content') or ''}")
        text = "\n\n".join(lines)
        if len(text) > TRANSCRIPT_LIMIT:
            text = "… (earlier contributions truncated)\n" + text[-TRANSCRIPT_LIMIT:]
        return text

    def item_text(self, agent, item) -> str:
        record = self.state.data["protocols"].get(item["ref"])
        if record is None:
            return f"[{item['ref']}] protocol no longer exists"
        preset = PRESETS[record["protocol"]]
        header = (
            f'[{record["id"]}] {record["protocol"]} "{record["topic"]}" started by '
            f"{record['initiator']}; participants: {', '.join(record['participants'])}; "
            f"facilitator: {record['facilitator']}."
        )
        goal = f"\nGoal: {record['goal']}" if record["goal"] else ""
        step = item.get("step")
        if step == "outcome":
            outcome = record.get("outcome") or {}
            return (
                f"{header}\nConcluded. Outcome artifact: {outcome.get('artifact')}; decisions: "
                f"{', '.join(outcome.get('decisions') or []) or 'none'}.\n"
                f"{outcome.get('summary') or ''}\n\nTranscript:\n{self.transcript(record, viewer=agent)}"
            )
        if step == "conclude":
            return (
                f"{header}{goal}\nAll rounds are complete. As facilitator, conclude with team "
                f"action=conclude protocol_id={record['id']} summary=... decisions=[...] "
                "(or your final message becomes the summary).\n\nTranscript:\n"
                + self.transcript(record, viewer=agent)
            )
        if item.get("round") != record["round"] or record["status"] != "running":
            return f"{header}\nThis round is already closed (now {record['status']}, round {record['round']})."
        blind = record["round"] <= record["blind_rounds"]
        stance = record["stances"].get(agent)
        earlier = self.transcript(record, viewer=agent)
        return (
            f"{header}{goal}\nRound {record['round']} of {record['rounds']}"
            + (
                " (blind: work independently; messages to other participants are blocked until "
                "everyone has contributed)"
                if blind
                else ""
            )
            + f".\n{preset['style']}"
            + (f"\nYour assigned position: {stance}" if stance else "")
            + (f"\n\nContributions so far:\n{earlier}" if earlier else "")
            + f"\n\nContribute with team action=contribute protocol_id={record['id']} content=... "
            "(set ready=true if you think it can conclude), or your final message is your contribution."
        )

    @staticmethod
    def summary(record) -> dict:
        return {
            key: record.get(key)
            for key in (
                "id",
                "protocol",
                "topic",
                "status",
                "round",
                "rounds",
                "participants",
                "facilitator",
                "outcome",
            )
        }
