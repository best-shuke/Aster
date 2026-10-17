"""Least-privilege views of a shared MCP manager.

An environment selects servers and tool patterns; an agent declares its own ceiling.
The effective permission is their intersection, so neither side can widen the other.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any

from mcp_client import McpExecutionUnknownError, McpManager
from tool_runtime import ExecutionContext, ToolResult

PATTERN_LIMIT = 64


@dataclass(frozen=True)
class ToolRule:
    """Tool-name patterns (fnmatch) for one server; ``allow=None`` admits every tool."""

    allow: tuple[str, ...] | None = None
    deny: tuple[str, ...] = ()

    def permits(self, tool: str) -> bool:
        if any(fnmatchcase(tool, pattern) for pattern in self.deny):
            return False
        return self.allow is None or any(fnmatchcase(tool, pattern) for pattern in self.allow)

    def describe(self) -> dict[str, Any]:
        return {
            "allow": list(self.allow) if self.allow is not None else "*",
            "deny": list(self.deny),
        }


@dataclass(frozen=True)
class McpScope:
    """Servers an actor may use, each with a conjunction of tool rules.

    ``servers=None`` is unrestricted. Intersections keep both sides' rules, so a tool
    must satisfy the environment's and the agent's patterns.
    """

    servers: Mapping[str, tuple[ToolRule, ...]] | None = None

    @property
    def unrestricted(self) -> bool:
        return self.servers is None

    def permits_server(self, server: str) -> bool:
        return self.servers is None or server in self.servers

    def permits(self, server: str, tool: str) -> bool:
        if self.servers is None:
            return True
        rules = self.servers.get(server)
        return rules is not None and all(rule.permits(tool) for rule in rules)

    def intersect(self, other: "McpScope | None") -> "McpScope":
        if other is None or other.servers is None:
            return self
        if self.servers is None:
            return other
        return McpScope(
            {
                name: (*rules, *other.servers[name])
                for name, rules in self.servers.items()
                if name in other.servers
            }
        )

    def describe(self) -> dict[str, Any] | str:
        if self.servers is None:
            return "*"
        return {name: [rule.describe() for rule in rules] for name, rules in self.servers.items()}


def parse_patterns(value: Any, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() or len(item) > 128 or "\x00" in item
        for item in value
    ):
        raise ValueError(f"{location} 必须是工具名模式数组（如 search_* ）")
    if len(value) > PATTERN_LIMIT:
        raise ValueError(f"{location} 最多 {PATTERN_LIMIT} 个模式")
    return tuple(item.strip() for item in value)


class ScopedMcp:
    """The ``mcp`` tool handler for one actor; every action is filtered by its scope."""

    def __init__(self, manager: McpManager, scope: McpScope):
        self.manager = manager
        self.scope = scope

    @property
    def servers(self):
        return {
            name: server
            for name, server in self.manager.servers.items()
            if self.scope.permits_server(name)
        }

    @property
    def server_count(self) -> int:
        return len([server for server in self.servers.values() if not server.disabled])

    def status(self) -> ToolResult:
        result = self.manager.status()
        servers = [
            {**entry, "scope": self._scope_of(entry["name"])}
            for entry in result.data["servers"]
            if self.scope.permits_server(entry["name"])
        ]
        return ToolResult(data={**result.data, "servers": servers})

    def _scope_of(self, server: str):
        if self.scope.servers is None:
            return "*"
        return [rule.describe() for rule in self.scope.servers.get(server, ())]

    def execute(
        self,
        action: str,
        server: str | None = None,
        query: str | None = None,
        tool: str | None = None,
        arguments: dict[str, Any] | None = None,
        limit: int = 12,
        *,
        _context: ExecutionContext,
    ) -> ToolResult:
        try:
            if server is not None and not self.scope.permits_server(server):
                raise PermissionError(f"MCP server {server!r} is not permitted for this agent")
            if action == "status":
                return self.status()
            if action == "connect":
                if not server:
                    raise ValueError("server is required for MCP connect")
                return self.manager.connect(server, _context)
            if action == "search":
                return self.manager.search(query or "", server, limit, _context, permit=self.scope)
            if action == "describe":
                if not tool:
                    raise ValueError("tool is required for MCP describe")
                return self.manager.describe(tool, server, _context, permit=self.scope)
            if action == "call":
                if not tool:
                    raise ValueError("tool is required for MCP call")
                try:
                    return self.manager.call(
                        tool, arguments or {}, server, _context, permit=self.scope
                    )
                except McpExecutionUnknownError as exc:
                    return ToolResult.failure("execution_unknown", str(exc), status="unknown")
        except PermissionError as exc:
            return ToolResult.failure(
                "permission_denied",
                str(exc),
                data={
                    "guidance": "This MCP server or tool is outside your authorized scope. "
                    "Do not retry; ask a teammate who holds the permission, or report the gap.",
                    "scope": self.scope.describe(),
                },
            )
        raise ValueError(f"Unknown MCP action {action!r}")
