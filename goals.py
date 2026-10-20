"""Session-owned goals and pure parsing for the /goal command."""

import math
from dataclasses import asdict, dataclass, field
from uuid import uuid4

from tool_runtime import validate

GOAL_STATUSES = (
    "active",
    "paused",
    "waiting_user",
    "blocked",
    "completed",
    "budget_exhausted",
    "cancelled",
)
STATUS_LABELS = {
    "active": "进行中",
    "paused": "已暂停",
    "waiting_user": "等待用户",
    "blocked": "受阻",
    "completed": "已完成",
    "budget_exhausted": "预算耗尽",
    "cancelled": "已取消",
}
USAGE = (
    '/goal [start] "目标" [--criterion "验收条件"] [--max-rounds 20] [--max-seconds 3600]\n'
    "/goal status | pause | cancel\n"
    "/goal resume [补充说明] [--max-rounds 总轮数] [--max-seconds 总秒数]"
)


@dataclass
class GoalState:
    objective: str
    criteria: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    max_rounds: int = 20
    max_seconds: int = 3600
    goal_id: str = field(default_factory=lambda: uuid4().hex)
    status: str = "paused"
    rounds: int = 0
    elapsed_ms: int = 0
    phase: str = "work"
    review_fingerprint: str = ""
    summary: str = ""
    next_step: str = ""
    evidence: list[str] = field(default_factory=list)
    criteria_met: list[bool] = field(default_factory=list)
    last_fingerprint: str = ""
    repeated_rounds: int = 0
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict, *, restoring=False) -> "GoalState":
        text = {"type": "string", "maxLength": 16000}
        integer = {"type": "integer", "minimum": 0}
        strings = {"type": "array", "items": text, "maxItems": 32}
        properties = {
            "objective": {**text, "minLength": 1},
            "criteria": strings,
            "notes": strings,
            "max_rounds": {**integer, "minimum": 1, "maximum": 1000000},
            "max_seconds": {**integer, "minimum": 1, "maximum": 31536000},
            "goal_id": {**text, "minLength": 1, "maxLength": 64},
            "status": {"type": "string", "enum": list(GOAL_STATUSES)},
            "rounds": integer,
            "elapsed_ms": integer,
            "phase": {"type": "string", "enum": ["work", "review"]},
            "review_fingerprint": text,
            "summary": text,
            "next_step": text,
            "evidence": strings,
            "criteria_met": {"type": "array", "items": {"type": "boolean"}, "maxItems": 32},
            "last_fingerprint": text,
            "repeated_rounds": integer,
            "reason": text,
        }
        validate(
            data,
            {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
            "goal",
        )
        state = cls(**data)
        if not state.objective.strip() or any(not value.strip() for value in state.criteria):
            raise ValueError("目标和验收条件不能为空")
        if restoring and state.status == "active":
            state.status = "paused"
            state.reason = "上次执行已中断；用 /goal resume 继续，先核实未确定的操作结果。"
        return state

    def status_lines(self) -> list[str]:
        lines = [
            f"目标：{self.objective}",
            f"状态：{STATUS_LABELS[self.status]} · 轮次 {self.rounds}/{self.max_rounds}"
            f" · 用时 {math.ceil(self.elapsed_ms / 1000)}/{self.max_seconds} 秒",
        ]
        lines.extend(
            f"验收 {index + 1}：{criterion}" for index, criterion in enumerate(self.criteria)
        )
        lines.extend(f"用户补充：{note}" for note in self.notes)
        for label, value in (
            ("进展", self.summary),
            ("下一步", self.next_step),
            ("说明", self.reason),
        ):
            if value:
                lines.append(f"{label}：{value}")
        lines.extend(f"证据：{item}" for item in self.evidence)
        return lines


@dataclass(frozen=True)
class GoalCommand:
    action: str
    text: str = ""
    criteria: tuple[str, ...] = ()
    max_rounds: int | None = None
    max_seconds: int | None = None


def parse_goal(arguments: list[str]) -> GoalCommand:
    arguments = list(arguments)
    action = "start"
    if not arguments:
        return GoalCommand("status")
    if arguments[0] in {"start", "status", "pause", "resume", "cancel"}:
        action = arguments.pop(0)
    if action in {"status", "pause", "cancel"}:
        if arguments:
            raise ValueError(USAGE)
        return GoalCommand(action)
    text, criteria, options = [], [], {}
    literal = False
    while arguments:
        value = arguments.pop(0)
        if value == "--" and not literal:
            literal = True
        elif value.startswith("--") and not literal:
            if value not in {"--criterion", "--max-rounds", "--max-seconds"} or not arguments:
                raise ValueError(USAGE)
            option = arguments.pop(0)
            if value == "--criterion":
                if action != "start" or not option.strip():
                    raise ValueError("--criterion 仅用于创建目标，且不能为空。")
                criteria.append(option.strip())
            else:
                key = value[2:].replace("-", "_")
                if key in options or not option.isdecimal():
                    raise ValueError(f"{value} 需要一个正整数，且只能指定一次。")
                options[key] = int(option)
        else:
            text.append(value)
    description = " ".join(text).strip()
    if action == "start":
        GoalState.from_dict(GoalState(description, criteria, **options).to_dict())
    else:
        GoalState.from_dict(GoalState("resume", **options).to_dict())
        if len(description) > 16000:
            raise ValueError("补充说明过长。")
    return GoalCommand(action, description, tuple(criteria), **options)
