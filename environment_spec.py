"""Typed parts of an environment definition beyond the original capability fields.

Inheritance and override rules (see docs/environments.md):

* An environment only selects from or narrows the process's startup capabilities.
* ``skills`` (name -> enabled) is an allowlist over the environment's skill sources;
  without it every discovered skill stays active, as before.
* ``mcp`` (server -> settings) is an allowlist over the environment's MCP servers and
  adds per-tool patterns; without it every configured server stays active.
* Each agent's capabilities are the intersection of the environment's and its own
  declaration. Omitted agent tools mean read-only tools; omitted MCP means none.
"""

import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from config import parse_reasoning_effort
from mcp_scope import McpScope, ToolRule, parse_patterns

KINDS = ("single-agent", "multi-agent")
TEAM_TOOL = "team"
AGENT_NAME = re.compile(r"[a-z][a-z0-9-]{0,31}\Z")
RESERVED_AGENTS = frozenset({"main", "all", "user", "team", "group", "system"})
# Members never receive these: the main agent alone talks to the user, owns background
# processes and the visible task plan.
MEMBER_EXCLUDED = frozenset({"ask_user", "monitor", "update_plan"})
MEMBER_DEFAULT_TOOLS = frozenset({"read", "grep", "find", "ls", "websearch"})
PROTOCOLS = (
    "meeting",
    "debate",
    "brainstorm",
    "independent_research",
    "review_decision",
    "coordinate_work",
)
AGENT_FIELDS = {
    "description",
    "instructions",
    "prompt_file",
    "enabled",
    "skills",
    "tools",
    "mcp",
    "reasoning_effort",
    "write_paths",
}


@dataclass(frozen=True)
class AgentSpec:
    name: str
    description: str
    instructions: str
    enabled: bool = True
    skills: tuple[str, ...] | None = None
    tools: frozenset[str] | None = None
    mcp: McpScope | None = None
    reasoning_effort: str | None = None
    write_paths: tuple[str, ...] | None = None
    prompt_file: Path | None = None
    dynamic: bool = False

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "enabled": self.enabled,
            "skills": list(self.skills) if self.skills is not None else "environment",
            "tools": sorted(self.tools) if self.tools is not None else "read-only",
            "mcp": self.mcp.describe() if self.mcp is not None else "none",
            "write_paths": list(self.write_paths) if self.write_paths is not None else "workspace",
            **({"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}),
            **({"dynamic": True} if self.dynamic else {}),
        }


@dataclass(frozen=True)
class TeamLimits:
    max_concurrent_agents: int = 3
    max_agents: int = 8
    max_messages_per_turn: int = 200
    max_activations_per_agent: int = 8
    max_message_hops: int = 10
    max_meeting_rounds: int = 4
    max_task_depth: int = 3
    max_tool_calls_per_agent: int = 80
    token_budget_per_agent: int = 400_000
    activation_timeout_seconds: int = 600
    # Interactive sessions: how long members keep working after main answers (0 = never).
    background_seconds: int = 1800


LIMIT_RANGES = {
    "max_concurrent_agents": (1, 20),
    "max_agents": (1, 32),
    "max_messages_per_turn": (1, 5000),
    "max_activations_per_agent": (1, 200),
    "max_message_hops": (1, 50),
    "max_meeting_rounds": (1, 10),
    "max_task_depth": (1, 10),
    "max_tool_calls_per_agent": (1, 5000),
    "token_budget_per_agent": (1000, 50_000_000),
    "activation_timeout_seconds": (10, 7200),
    "background_seconds": (0, 86400),
}


@dataclass(frozen=True)
class CollaborationPolicy:
    protocols: frozenset[str] = frozenset(PROTOCOLS)
    dynamic_agents: bool = False
    broadcast: bool = True


@dataclass(frozen=True)
class McpSelection:
    enabled: bool = True
    tools: tuple[str, ...] | None = None
    deny: tuple[str, ...] = ()

    def rule(self) -> ToolRule:
        return ToolRule(self.tools, self.deny)

    def as_json(self) -> Any:
        if self.tools is None and not self.deny:
            return self.enabled
        return {
            "enabled": self.enabled,
            **({"tools": list(self.tools)} if self.tools is not None else {}),
            **({"deny": list(self.deny)} if self.deny else {}),
        }


@dataclass(frozen=True)
class TeamSpec:
    kind: str = "single-agent"
    agents: tuple[AgentSpec, ...] = ()
    limits: TeamLimits = field(default_factory=TeamLimits)
    collaboration: CollaborationPolicy = field(default_factory=CollaborationPolicy)

    @property
    def multi_agent(self) -> bool:
        return self.kind == "multi-agent"

    def agent(self, name: str) -> AgentSpec | None:
        return next((agent for agent in self.agents if agent.name == name), None)


def parse_kind(value: Any) -> str:
    if value not in KINDS:
        raise ValueError(f"type 必须是 {' / '.join(KINDS)}")
    return value


def parse_skill_selection(value: Any) -> dict[str, bool]:
    if not isinstance(value, dict):
        raise ValueError('skills 必须是 技能名→是否启用 的映射，例如 {"research": true}')
    for name, enabled in value.items():
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) or len(name) > 64:
            raise ValueError(f"非法技能名：{name!r}")
        if not isinstance(enabled, bool):
            raise ValueError(f"skills.{name} 必须是 boolean")
    return dict(value)


def parse_mcp_selection(value: Any) -> dict[str, McpSelection]:
    if not isinstance(value, dict):
        raise ValueError('mcp 必须是 服务名→设置 的映射，例如 {"github": true}')
    selection = {}
    for name, item in value.items():
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", name):
            raise ValueError(f"非法 MCP 服务名：{name!r}")
        if isinstance(item, bool):
            selection[name] = McpSelection(item)
            continue
        if not isinstance(item, dict) or set(item) - {"enabled", "tools", "deny"}:
            raise ValueError(f"mcp.{name} 只允许 enabled/tools/deny")
        enabled = item.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(f"mcp.{name}.enabled 必须是 boolean")
        tools = parse_patterns(item["tools"], f"mcp.{name}.tools") if "tools" in item else None
        deny = parse_patterns(item.get("deny", []), f"mcp.{name}.deny")
        selection[name] = McpSelection(enabled, tools, deny)
    return selection


def selection_scope(selection: dict[str, McpSelection]) -> McpScope:
    return McpScope({name: (item.rule(),) for name, item in selection.items() if item.enabled})


def parse_agent_mcp(value: Any, location: str) -> McpScope:
    if isinstance(value, list):
        if any(not isinstance(item, str) or not item for item in value):
            raise ValueError(f"{location} 必须是服务名数组或 服务名→工具模式 映射")
        return McpScope({name: (ToolRule(),) for name in value})
    if not isinstance(value, dict):
        raise ValueError(f"{location} 必须是服务名数组或 服务名→工具模式 映射")
    servers = {}
    for name, patterns in value.items():
        if patterns is True or patterns == "*":
            servers[name] = (ToolRule(),)
        else:
            servers[name] = (ToolRule(parse_patterns(patterns, f"{location}.{name}")),)
    return McpScope(servers)


def parse_write_paths(value: Any, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{location} 必须是工作区相对路径数组")
    paths = []
    for item in value:
        raw = Path(item)
        if not item.strip() or raw.is_absolute() or ".." in raw.parts or "\x00" in item:
            raise ValueError(f"{location} 只能包含不逃逸工作区的相对路径：{item!r}")
        paths.append(raw.as_posix())
    return tuple(paths)


def parse_agent(
    name: Any,
    value: Any,
    *,
    read_prompt,
    tool_names: frozenset[str],
) -> AgentSpec:
    if not isinstance(name, str) or AGENT_NAME.fullmatch(name) is None:
        raise ValueError(f"非法成员名：{name!r}（小写字母开头，可含数字和连字符）")
    if name in RESERVED_AGENTS or name.startswith("group"):
        raise ValueError(f"成员名 {name!r} 为保留名")
    location = f"agents.{name}"
    if not isinstance(value, dict) or set(value) - AGENT_FIELDS:
        unknown = sorted(set(value) - AGENT_FIELDS) if isinstance(value, dict) else []
        raise ValueError(f"{location} 包含不支持的字段：{', '.join(unknown) or '不是对象'}")
    description = value.get("description", "")
    if not isinstance(description, str) or len(description) > 1000:
        raise ValueError(f"{location}.description 必须是不超过 1000 字的字符串")
    if "instructions" in value and "prompt_file" in value:
        raise ValueError(f"{location} 只能使用 instructions 或 prompt_file 之一")
    prompt_file = None
    if "prompt_file" in value:
        prompt_file, instructions = read_prompt(value["prompt_file"])
    else:
        instructions = value.get("instructions", "")
        if not isinstance(instructions, str) or len(instructions) > 65536:
            raise ValueError(f"{location}.instructions 必须是不超过 64 KiB 的字符串")
    instructions = instructions.strip()
    if not instructions and not description.strip():
        raise ValueError(f"{location} 至少需要 description 或 instructions")
    enabled = value.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"{location}.enabled 必须是 boolean")
    skills = None
    if "skills" in value:
        raw = value["skills"]
        if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
            raise ValueError(f"{location}.skills 必须是技能名数组")
        skills = tuple(dict.fromkeys(raw))
    tools = None
    if "tools" in value:
        raw = value["tools"]
        if not isinstance(raw, list) or any(not isinstance(item, str) for item in raw):
            raise ValueError(f"{location}.tools 必须是工具名数组")
        unknown = set(raw) - tool_names
        if unknown:
            raise ValueError(f"{location}.tools 未知工具：{', '.join(sorted(unknown))}")
        excluded = set(raw) & MEMBER_EXCLUDED
        if excluded:
            raise ValueError(
                f"{location}.tools 不能包含仅主代理可用的工具：{', '.join(sorted(excluded))}"
            )
        tools = frozenset(raw)
    mcp = parse_agent_mcp(value["mcp"], f"{location}.mcp") if "mcp" in value else None
    effort = None
    if "reasoning_effort" in value:
        if not isinstance(value["reasoning_effort"], str):
            raise ValueError(f"{location}.reasoning_effort 必须是合法档位")
        effort = parse_reasoning_effort(value["reasoning_effort"])
    write_paths = (
        parse_write_paths(value["write_paths"], f"{location}.write_paths")
        if "write_paths" in value
        else None
    )
    return AgentSpec(
        name,
        description.strip(),
        instructions,
        enabled,
        skills,
        tools,
        mcp,
        effort,
        write_paths,
        prompt_file,
    )


def parse_limits(value: Any, startup_max_agents: int) -> TeamLimits:
    if not isinstance(value, dict):
        raise ValueError("limits 必须是对象")
    unknown = set(value) - set(LIMIT_RANGES)
    if unknown:
        raise ValueError(f"不支持的 limits 字段：{', '.join(sorted(unknown))}")
    values = {}
    for name, item in value.items():
        low, high = LIMIT_RANGES[name]
        if name == "max_concurrent_agents":
            high = min(high, startup_max_agents)
        if type(item) is not int or not low <= item <= high:
            raise ValueError(f"limits.{name} 必须是 {low} 到 {high} 之间的整数")
        values[name] = item
    limits = TeamLimits(**values)
    if "max_concurrent_agents" not in values:
        limits = replace(
            limits,
            max_concurrent_agents=min(limits.max_concurrent_agents, startup_max_agents),
        )
    return limits


def parse_collaboration(value: Any) -> CollaborationPolicy:
    if not isinstance(value, dict) or set(value) - {"protocols", "dynamic_agents", "broadcast"}:
        raise ValueError("collaboration 只允许 protocols/dynamic_agents/broadcast")
    protocols = value.get("protocols", list(PROTOCOLS))
    if not isinstance(protocols, list) or any(item not in PROTOCOLS for item in protocols):
        raise ValueError(f"collaboration.protocols 只能包含：{', '.join(PROTOCOLS)}")
    for name in ("dynamic_agents", "broadcast"):
        if name in value and not isinstance(value[name], bool):
            raise ValueError(f"collaboration.{name} 必须是 boolean")
    return CollaborationPolicy(
        frozenset(protocols),
        value.get("dynamic_agents", False),
        value.get("broadcast", True),
    )


def parse_team(data: dict, *, read_prompt, tool_names, startup_max_agents) -> TeamSpec:
    kind = parse_kind(data.get("type", "single-agent"))
    raw_agents = data.get("agents", {})
    if not isinstance(raw_agents, dict):
        raise ValueError("agents 必须是 成员名→定义 的映射")
    if raw_agents and kind != "multi-agent":
        raise ValueError("agents 仅用于 type=multi-agent 的环境")
    limits = parse_limits(data.get("limits", {}), startup_max_agents)
    agents = tuple(
        parse_agent(name, value, read_prompt=read_prompt, tool_names=tool_names)
        for name, value in raw_agents.items()
    )
    if len(agents) > limits.max_agents:
        raise ValueError(f"成员数超过 limits.max_agents（{limits.max_agents}）")
    return TeamSpec(kind, agents, limits, parse_collaboration(data.get("collaboration", {})))


def capability_notes(team: TeamSpec, skills: dict[str, bool] | None, mcp: McpScope | None):
    """Non-fatal mismatches: an agent asks for a capability the environment does not grant."""
    notes = []
    for agent in team.agents:
        if agent.skills is not None and skills is not None:
            missing = [name for name in agent.skills if not skills.get(name)]
            if missing:
                notes.append(f"成员 {agent.name} 的技能未在环境中启用：{', '.join(missing)}")
        if agent.mcp is not None and agent.mcp.servers is not None:
            granted = mcp.servers if mcp is not None and mcp.servers is not None else None
            missing = [
                name for name in agent.mcp.servers if granted is not None and name not in granted
            ]
            if missing:
                notes.append(f"成员 {agent.name} 的 MCP 服务未在环境中启用：{', '.join(missing)}")
    return tuple(notes)
