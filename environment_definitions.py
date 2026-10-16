"""Create and edit environment definitions without hand-editing JSON.

Every change is validated with the same resolver that loads environments, written
atomically, and audited. Trust is carried over only when the environment was trusted
before the edit, so an edit never blesses resources that changed elsewhere.
"""

import json
import re
import shlex
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from config import BUILTIN_TEMPLATES_DIR
from environment_spec import LIMIT_RANGES, MEMBER_DEFAULT_TOOLS, parse_kind
from environments import (
    NAME,
    RESERVED,
    EnvironmentEntry,
    EnvironmentStore,
    ResolvedEnvironment,
    _read_json,
)
from file_state import atomic_write_text

# Words the /environment command uses; an environment cannot be created with these names.
SUBCOMMANDS = frozenset(
    {
        "create",
        "use",
        "exit",
        "status",
        "list",
        "skills",
        "mcp",
        "agents",
        "limits",
        "log",
        "reload",
        "type",
        "help",
    }
)
SENSITIVE = re.compile(
    r"token|password|passwd|secret|api[-_]?key|authorization|credential|auth", re.I
)
REFERENCE = re.compile(r"\$\{[^}]+\}")


@dataclass(frozen=True)
class EditResult:
    entry: EnvironmentEntry
    environment: ResolvedEnvironment
    approved: bool


class EnvironmentEditor:
    def __init__(self, store: EnvironmentStore, templates_dir: Path = BUILTIN_TEMPLATES_DIR):
        self.store = store
        self.templates_dir = templates_dir

    # Templates and creation ---------------------------------------------------------

    def templates(self) -> dict[str, dict]:
        found = {}
        for path in sorted(self.templates_dir.glob("*.json")):
            found[path.stem] = _read_json(path)
        return found

    def create(
        self,
        name: str,
        *,
        template: str = "single-agent",
        description: str | None = None,
        scope: str = "user",
    ) -> EditResult:
        if not NAME.fullmatch(name) or name in RESERVED or name in SUBCOMMANDS:
            raise ValueError(
                f"环境名 {name!r} 无效或为保留字；使用字母、数字、点、下划线或连字符，"
                f"且不能是 {', '.join(sorted(RESERVED | SUBCOMMANDS))}"
            )
        templates = self.templates()
        if template not in templates:
            raise ValueError(f"未知模板 {template!r}；可选：{', '.join(templates) or '无'}")
        if scope not in self.store.roots:
            raise ValueError(f"未知位置 {scope!r}；可选：{', '.join(self.store.roots)}")
        if any(entry.name == name for entry in self.store.list()):
            raise ValueError(f"环境 {name!r} 已存在；用 /env show {name} 查看")
        data = deepcopy(templates[template])
        data["name"] = name
        if description is not None:
            data["description"] = description
        root = self.store.roots[scope]
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{name}.json"
        # Exclusive create: never overwrite a definition that appeared concurrently.
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        try:
            entry = self.store.find(f"{scope}:{name}")
            environment = self.store.resolve(entry.environment_id)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        # The user authored this definition through the CLI, so record it as trusted.
        self.store.approve(environment)
        self.store.audit("created", entry.environment_id, template=template)
        return EditResult(entry, self.store.resolve(entry.environment_id), True)

    # Generic edit -------------------------------------------------------------------

    def load(self, selector: str) -> tuple[EnvironmentEntry, dict]:
        entry = self.store.find(selector)
        return entry, _read_json(entry.path)

    def update(self, selector: str, mutate: Callable[[dict], str | None]) -> EditResult:
        entry, data = self.load(selector)
        try:
            was_approved = self.store.approved(self.store.resolve(entry.environment_id))
        except (OSError, ValueError):
            was_approved = False
        candidate = deepcopy(data)
        detail = mutate(candidate)
        if candidate == data:
            environment = self.store.resolve(entry.environment_id)
            return EditResult(entry, environment, self.store.approved(environment))
        self.store.resolve_data(entry, candidate)
        original = entry.path.read_text(encoding="utf-8")
        atomic_write_text(entry.path, json.dumps(candidate, ensure_ascii=False, indent=2) + "\n")
        try:
            environment = self.store.resolve(entry.environment_id)
        except Exception:
            atomic_write_text(entry.path, original)
            raise
        if was_approved:
            self.store.approve(environment)
        self.store.audit("edited", entry.environment_id, change=detail, reapproved=was_approved)
        return EditResult(entry, self.store.resolve(entry.environment_id), was_approved)

    # Skills -------------------------------------------------------------------------

    def available_skills(self, selector: str) -> list:
        environment = self.store.resolve(selector)
        return self.store.skill_catalog(environment, selected=False).available

    def edit_skills(self, selector: str, action: str, names: list[str]) -> EditResult:
        if action not in {"add", "remove", "enable", "disable"} or not names:
            raise ValueError("用法：/env skills add|remove|enable|disable 名称...")
        environment = self.store.resolve(selector)
        available = {skill.name for skill in self.available_skills(selector)}
        current = self.store.skill_catalog(environment)

        def mutate(data):
            selection = data.get("skills")
            if selection is None:
                # A legacy environment used every discovered skill; keep them all when the
                # first explicit selection is made, so the edit changes only what was asked.
                selection = {skill.name: True for skill in current.skills}
            for name in names:
                if action == "add":
                    if name not in available:
                        raise ValueError(
                            f"技能 {name!r} 不在此环境的任何技能来源中；/env skills list 查看可用技能"
                        )
                    selection[name] = True
                elif name not in selection:
                    raise ValueError(f"技能 {name!r} 未加入此环境")
                elif action == "remove":
                    del selection[name]
                else:
                    selection[name] = action == "enable"
            data["skills"] = selection
            return f"skills {action} {' '.join(names)}"

        return self.update(selector, mutate)

    # MCP ----------------------------------------------------------------------------

    def available_mcp(self, selector: str) -> dict:
        entry, data = self.load(selector)
        unselected = {key: value for key, value in data.items() if key != "mcp"}
        return self.store.resolve_data(entry, unselected).servers

    def edit_mcp(
        self,
        selector: str,
        action: str,
        name: str,
        *,
        tools: list[str] | None = None,
        deny: list[str] | None = None,
    ) -> EditResult:
        if action not in {"add", "remove", "enable", "disable", "tools"}:
            raise ValueError("用法：/env mcp add|remove|enable|disable|tools 服务名")
        environment = self.store.resolve(selector)
        available = self.available_mcp(selector)

        def mutate(data):
            selection = data.get("mcp")
            if selection is None:
                selection = {
                    server: not config.disabled for server, config in environment.servers.items()
                }
            if action == "add":
                if name not in available:
                    raise ValueError(
                        f"MCP 服务 {name!r} 未在全局 MCP 配置或环境的 MCP 文件中定义；"
                        "用 /env mcp add 名称 --command 命令 或 --url 地址 定义新服务"
                    )
                selection[name] = _mcp_entry(True, tools, deny)
            elif name not in selection:
                raise ValueError(f"MCP 服务 {name!r} 未加入此环境")
            elif action == "remove":
                del selection[name]
            elif action == "tools":
                previous = selection[name]
                enabled = previous if isinstance(previous, bool) else previous.get("enabled", True)
                selection[name] = _mcp_entry(enabled, tools or None, deny)
            else:
                previous = selection[name]
                enabled = action == "enable"
                selection[name] = (
                    enabled if isinstance(previous, bool) else {**previous, "enabled": enabled}
                )
            data["mcp"] = selection
            return f"mcp {action} {name}"

        return self.update(selector, mutate)

    def define_mcp_server(self, selector: str, name: str, server: dict) -> EditResult:
        """Add a server to the environment's own MCP file; secrets must be ${VAR} references."""
        check_credentials(name, server)
        entry, data = self.load(selector)
        resources = entry.path.parent / entry.name
        relative = f"{entry.name}/mcp.json"
        configured = data.get("mcp_config_file")
        if configured is not None and configured != relative:
            raise ValueError(f"环境已使用 MCP 文件 {configured}；请直接编辑该文件")
        path = resources / "mcp.json"
        payload = _read_json(path) if path.exists() else {"mcpServers": {}}
        servers = payload.setdefault("mcpServers", {})
        if name in servers:
            raise ValueError(f"环境 MCP 文件已定义服务 {name!r}")
        original = path.read_text(encoding="utf-8") if path.exists() else None
        servers[name] = server
        atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        path.chmod(0o600)

        def mutate(definition):
            inherit = definition.get("inherit_mcp", "mcp_config_file" not in definition)
            definition["mcp_config_file"] = relative
            definition["inherit_mcp"] = inherit
            selection = definition.get("mcp")
            if selection is not None:
                selection[name] = True
            return f"mcp define {name}"

        try:
            return self.update(selector, mutate)
        except Exception:
            if original is None:
                path.unlink(missing_ok=True)
            else:
                atomic_write_text(path, original)
            raise

    # Agents -------------------------------------------------------------------------

    def edit_agent(self, selector: str, action: str, name: str, values: dict) -> EditResult:
        if action not in {"add", "remove", "enable", "disable", "set"}:
            raise ValueError("用法：/env agents add|remove|enable|disable|set 名称 [选项]")

        def mutate(data):
            if data.get("type", "single-agent") != "multi-agent":
                raise ValueError("该环境类型为 single-agent；先执行 /env type multi-agent")
            agents = data.setdefault("agents", {})
            if action == "add":
                if name in agents:
                    raise ValueError(f"成员 {name!r} 已存在；用 /env agents set 修改")
                agents[name] = {
                    "description": values.get("description", ""),
                    **{key: value for key, value in values.items() if key != "description"},
                }
                if "tools" not in agents[name]:
                    agents[name]["tools"] = sorted(MEMBER_DEFAULT_TOOLS)
            elif name not in agents:
                raise ValueError(f"成员 {name!r} 不存在")
            elif action == "remove":
                del agents[name]
            elif action == "set":
                if not values:
                    raise ValueError("用法：/env agents set 名称 --description ... 等选项")
                agents[name].update(values)
                if "prompt_file" in values:
                    agents[name].pop("instructions", None)
                if "instructions" in values:
                    agents[name].pop("prompt_file", None)
            else:
                agents[name]["enabled"] = action == "enable"
            return f"agents {action} {name}"

        return self.update(selector, mutate)

    # Type and limits ----------------------------------------------------------------

    def set_kind(self, selector: str, kind: str) -> EditResult:
        parse_kind(kind)

        def mutate(data):
            if kind == "single-agent" and data.get("agents"):
                raise ValueError("环境仍有成员；先用 /env agents remove 移除成员")
            data["type"] = kind
            return f"type {kind}"

        return self.update(selector, mutate)

    def set_limits(self, selector: str, assignments: list[str]) -> EditResult:
        values = {}
        for item in assignments:
            key, separator, raw = item.partition("=")
            if not separator or key not in LIMIT_RANGES:
                raise ValueError(f"用法：/env limits 键=整数；可设置：{', '.join(LIMIT_RANGES)}")
            if raw == "default":
                values[key] = None
                continue
            try:
                values[key] = int(raw)
            except ValueError as exc:
                raise ValueError(f"limits.{key} 必须是整数") from exc

        def mutate(data):
            limits = data.setdefault("limits", {})
            for key, value in values.items():
                if value is None:
                    limits.pop(key, None)
                else:
                    limits[key] = value
            if not limits:
                data.pop("limits")
            return "limits " + " ".join(assignments)

        return self.update(selector, mutate)


def _mcp_entry(enabled: bool, tools: list[str] | None, deny: list[str] | None):
    if not tools and not deny:
        return enabled
    return {
        "enabled": enabled,
        **({"tools": list(tools)} if tools else {}),
        **({"deny": list(deny)} if deny else {}),
    }


def check_credentials(name: str, server: dict) -> None:
    """Reject literal secrets; credentials stay in the process environment as ${VAR}."""
    for field_name in ("env", "headers"):
        for key, value in (server.get(field_name) or {}).items():
            if not isinstance(value, str):
                raise ValueError(f"{name}.{field_name}.{key} 必须是字符串")
            sensitive = field_name == "headers" or SENSITIVE.search(key)
            if sensitive and not REFERENCE.search(value):
                raise ValueError(
                    f"{name}.{field_name}.{key} 看起来是凭据，必须写成 ${{环境变量}} 引用，"
                    "不能把明文写入环境配置"
                )
    arguments = list(server.get("args") or [])
    for index, argument in enumerate(arguments):
        flagged = SENSITIVE.search(argument)
        if flagged and "=" in argument and not REFERENCE.search(argument.split("=", 1)[1]):
            raise ValueError(f"{name}.args 第 {index + 1} 项包含明文凭据；请使用 ${{环境变量}}")
        follows = arguments[index + 1] if index + 1 < len(arguments) else None
        if flagged and "=" not in argument and follows and not REFERENCE.search(follows):
            raise ValueError(f"{name}.args 第 {index + 2} 项可能是明文凭据；请使用 ${{环境变量}}")
    url = server.get("url") or ""
    if re.match(r"[a-z][a-z0-9+.-]*://[^/@]+@", url, re.I) and not REFERENCE.search(url):
        raise ValueError(f"{name}.url 包含明文认证信息；请使用 ${{环境变量}}")


def parse_server_options(arguments: list[str]) -> dict:
    """Parse --command/--arg/--url/--env/--header/--transport/--timeout/--required."""
    server: dict = {}
    index = 0
    while index < len(arguments):
        option = arguments[index]
        if option == "--required":
            server["required"] = True
            index += 1
            continue
        if index + 1 >= len(arguments):
            raise ValueError(f"{option} 缺少取值")
        value = arguments[index + 1]
        index += 2
        if option == "--command":
            parts = shlex.split(value)
            if not parts:
                raise ValueError("--command 不能为空")
            server["command"], extra = parts[0], parts[1:]
            server.setdefault("args", []).extend(extra)
        elif option == "--arg":
            server.setdefault("args", []).append(value)
        elif option == "--url":
            server["url"] = value
        elif option in {"--env", "--header"}:
            key, separator, item = value.partition("=")
            if not separator or not key:
                raise ValueError(f"{option} 需要 键=值，值中的凭据写成 ${{环境变量}}")
            server.setdefault("env" if option == "--env" else "headers", {})[key] = item
        elif option == "--transport":
            server["transport"] = value
        elif option == "--timeout":
            try:
                server["timeout"] = float(value)
            except ValueError as exc:
                raise ValueError("--timeout 必须是数字") from exc
        else:
            raise ValueError(f"未知选项 {option}")
    if ("command" in server) == ("url" in server):
        raise ValueError("新 MCP 服务需要且只能指定 --command 或 --url 之一")
    return server
