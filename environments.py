from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

from config import APP_NAME, CLI_NAME, Settings, parse_reasoning_effort
from environment_spec import (
    McpSelection,
    TeamSpec,
    capability_notes,
    format_limit,
    parse_mcp_selection,
    parse_skill_selection,
    parse_team,
    selection_scope,
)
from file_state import (
    GENERATED_DIRECTORIES,
    atomic_write_text,
    cached_file_digest,
    local_code_files,
)
from mcp_client import McpServerConfig, load_mcp_config
from mcp_scope import McpScope
from skills import SkillCatalog, skill_files, skill_roots
from tools import TOOL_NAMES, WORKER_EXCLUDED, ToolPolicy

NAME = re.compile(r"[\w][\w.-]{0,63}\Z")
RESERVED = {"off", "show", "trust", "deny", "save", "keep"}
LOADED_ENVIRONMENT_MTIME = Path(__file__).stat().st_mtime_ns
FIELDS = {
    "version",
    "name",
    "description",
    "prompt_file",
    "skills_dirs",
    "mcp_config_file",
    "inherit_mcp",
    "inherit_skills",
    "reasoning_effort",
    "websearch",
    "tools",
    "worker_tools",
    "role_prompts",
    "role_source",
    "status_file",
    "max_subagents",
    "opening_template",
    # Environment runtime fields (all optional; definitions without them behave as before).
    "type",
    "skills",
    "mcp",
    "agents",
    "limits",
    "collaboration",
}

# 环境里 role_prompts 的角色名规范。
ROLE_NAME = re.compile(r"[a-z][a-z0-9-]{0,63}\Z")


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"重复 JSON 键：{key}")
        result[key] = value
    return result


def _read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object)
    if not isinstance(data, dict):
        raise ValueError(f"JSON 必须是对象：{path}")
    return data


@dataclass(frozen=True)
class EnvironmentEntry:
    name: str
    source: str
    path: Path

    @property
    def environment_id(self) -> str:
        return f"{self.source}:{self.name}"


@dataclass(frozen=True)
class ResolvedEnvironment:
    entry: EnvironmentEntry
    description: str
    fingerprint: str
    settings: Settings
    prompt_text: str
    skills_dirs: tuple[Path, ...]
    policy: ToolPolicy
    servers: dict[str, McpServerConfig]
    worker_tools: frozenset[str]
    role_prompts: dict[str, str]
    template_data: dict | None
    template_id: str | None
    template_path: Path | None
    status_file: Path | None = None
    role_source: dict | None = None
    inherit_mcp: bool = False
    inherit_skills: bool = False
    # None keeps legacy semantics: every discovered skill / configured server is active.
    skill_selection: dict[str, bool] | None = None
    mcp_selection: dict[str, McpSelection] | None = None
    mcp_scope: McpScope | None = None
    missing_mcp: tuple[str, ...] = ()
    team: TeamSpec = field(default_factory=TeamSpec)
    notes: tuple[str, ...] = ()

    @property
    def environment_id(self) -> str:
        return self.entry.environment_id

    @property
    def multi_agent(self) -> bool:
        return self.team.multi_agent

    @property
    def active_skills(self) -> frozenset[str] | None:
        if self.skill_selection is None:
            return None
        return frozenset(name for name, enabled in self.skill_selection.items() if enabled)

    @property
    def reference(self) -> dict:
        return {"path": str(self.entry.path), "fingerprint": self.fingerprint}


class EnvironmentStore:
    def __init__(
        self,
        settings: Settings,
        templates=None,
        *,
        user_dir: Path | None = None,
        baseline_skills: SkillCatalog | None = None,
        baseline_servers: dict[str, McpServerConfig] | None = None,
    ):
        self.baseline = settings
        self.baseline_skills = baseline_skills
        self.baseline_servers = baseline_servers
        self.templates = templates
        self.roots = {
            "program": settings.environments_dir.expanduser().resolve(),
            "user": (user_dir or Path.home() / ".agents" / "environments").resolve(),
            "workspace": settings.workspace.resolve() / ".agent" / "environments",
        }
        self.state_dir = settings.environment_state_dir.expanduser().resolve()
        self.trust_file = self.state_dir / "trust.json"
        self.audit_file = self.state_dir / "audit.jsonl"
        self.diagnostics: list[str] = []
        self._digest_cache: dict = {}

    def list(self) -> list[EnvironmentEntry]:
        self.diagnostics.clear()
        entries = []
        seen = set()
        for source, root in self.roots.items():
            for path in sorted(root.glob("*.json")):
                if not NAME.fullmatch(path.stem) or path.stem in RESERVED:
                    self.diagnostics.append(f"无效环境文件名：{path}")
                    continue
                resolved = path.resolve()
                if not resolved.is_relative_to(root) or path.is_symlink():
                    self.diagnostics.append(f"环境定义不能使用符号链接：{path}")
                    continue
                if resolved not in seen:
                    entries.append(EnvironmentEntry(path.stem, source, resolved))
                    seen.add(resolved)
        return entries

    def find(self, selector: str) -> EnvironmentEntry:
        entries = self.list()
        matches = [entry for entry in entries if selector in {entry.name, entry.environment_id}]
        if not matches:
            raise ValueError(f"环境 {selector!r} 未找到")
        if len(matches) != 1:
            choices = "、".join(entry.environment_id for entry in matches)
            raise ValueError(f"环境重名，请指定来源：{choices}")
        return matches[0]

    @staticmethod
    def _path(base: Path, value, *, directory: bool = False) -> Path:
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ValueError("资源路径必须是非空字符串")
        raw = Path(value).expanduser()
        resolved = (raw if raw.is_absolute() else base / raw).resolve()
        if not raw.is_absolute() and (".." in raw.parts or not resolved.is_relative_to(base)):
            raise ValueError(f"相对路径不能逃逸环境目录：{value}")
        if not (resolved.is_dir() if directory else resolved.is_file()):
            raise ValueError(f"环境资源不存在或类型错误：{resolved}")
        return resolved

    @staticmethod
    def _files(roots: tuple[Path, ...]) -> list[Path]:
        files = []
        for root in roots:
            for directory, directories, filenames in os.walk(root):
                parent = Path(directory)
                directories[:] = [
                    name
                    for name in directories
                    if name not in GENERATED_DIRECTORIES and not name.startswith(".")
                ]
                filenames = [
                    name
                    for name in filenames
                    if not name.startswith(".")
                    and Path(name).suffix not in {".pyc", ".pyo", ".log", ".tmp"}
                ]
                for name in [*directories, *filenames]:
                    if (parent / name).is_symlink():
                        raise ValueError(f"技能包不能包含符号链接：{parent / name}")
                files.extend(parent / name for name in filenames)
                if len(files) > 4096:
                    raise ValueError("环境能力包超过限制（4096 个文件 / 32 MiB）")
        return sorted(set(files))

    @staticmethod
    def _policy(raw) -> ToolPolicy:
        if not isinstance(raw, dict) or set(raw) - {"allow", "deny"}:
            raise ValueError("tools 只允许 allow/deny，不允许注入 handler")
        parsed = {}
        for key, names in raw.items():
            if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
                raise ValueError(f"tools.{key} 必须是工具名数组")
            unknown = set(names) - TOOL_NAMES
            if unknown:
                raise ValueError(f"未知工具：{', '.join(sorted(unknown))}")
            parsed[key] = frozenset(names)
        return ToolPolicy(parsed.get("allow"), parsed.get("deny", frozenset()))

    def resolve(self, selector: str) -> ResolvedEnvironment:
        if Path(__file__).stat().st_mtime_ns != LOADED_ENVIRONMENT_MTIME:
            raise ValueError(
                f"环境加载代码已更新，当前 {APP_NAME} 进程仍使用旧版本；"
                f"请退出并重新启动 {CLI_NAME}，再选择环境。/new 或重选环境不会重载程序代码。"
            )
        entry = self.find(selector)
        return self.resolve_data(entry, _read_json(entry.path))

    def resolve_data(self, entry: EnvironmentEntry, data: dict) -> ResolvedEnvironment:
        """Validate a definition's content; editors use it before writing a change."""
        if set(data) - FIELDS:
            raise ValueError(f"不支持的环境字段：{', '.join(sorted(set(data) - FIELDS))}")
        if type(data.get("version")) is not int or data["version"] != 1:
            raise ValueError("环境 version 必须是 1")
        if data.get("name") != entry.name:
            raise ValueError("环境 name 必须与 JSON 文件名一致")
        description = data.get("description", "")
        if not isinstance(description, str):
            raise ValueError("description 必须是字符串")
        base = entry.path.parent
        status_file = self._status_path(base, data.get("status_file"))
        role_source = self._role_source(data.get("role_source"))
        baseline = self.baseline
        startup_skills_enabled = (
            self.baseline_skills.enabled
            if self.baseline_skills is not None
            else baseline.skills_enabled
        )
        inherit_mcp = data.get("inherit_mcp", "mcp_config_file" not in data)
        inherit_skills = data.get("inherit_skills", False)
        for name, value in (("inherit_mcp", inherit_mcp), ("inherit_skills", inherit_skills)):
            if not isinstance(value, bool):
                raise ValueError(f"{name} 必须是 boolean")
        prompt = (
            self._path(base, data["prompt_file"]) if "prompt_file" in data else baseline.prompt_file
        )
        prompt_text = prompt.read_text(encoding="utf-8").strip()
        if not prompt_text:
            raise ValueError(f"提示词为空：{prompt}")
        roots = data.get("skills_dirs", [])
        if not isinstance(roots, list):
            raise ValueError("skills_dirs 必须是目录数组")
        skills = tuple(self._path(base, item, directory=True) for item in roots)
        websearch = data.get("websearch", baseline.websearch_enabled)
        if not isinstance(websearch, bool):
            raise ValueError("websearch 必须是 boolean")
        if websearch and not baseline.websearch_enabled:
            raise ValueError("环境不能启用启动时已关闭的 websearch")
        workers = data.get("max_subagents", baseline.max_subagents)
        if type(workers) is not int or not 1 <= workers <= baseline.max_subagents:
            raise ValueError(f"max_subagents 必须在 1 到启动上限 {baseline.max_subagents} 之间")
        effort = data.get("reasoning_effort", baseline.reasoning_effort)
        if not isinstance(effort, str) or not effort.strip():
            raise ValueError("reasoning_effort 必须是合法档位")
        policy = self._policy(data.get("tools", {}))
        worker_raw = data.get("worker_tools", [])
        if not isinstance(worker_raw, list) or any(
            not isinstance(name, str) for name in worker_raw
        ):
            raise ValueError("worker_tools 必须是工具名数组")
        unknown_workers = set(worker_raw) - TOOL_NAMES
        if unknown_workers:
            raise ValueError(f"worker_tools 未知工具：{', '.join(sorted(unknown_workers))}")
        blocked_workers = set(worker_raw) & WORKER_EXCLUDED
        if blocked_workers:
            raise ValueError(
                f"worker_tools 不能包含子代理禁用工具：{', '.join(sorted(blocked_workers))}"
            )
        worker_tools = frozenset(worker_raw)
        roles_raw = data.get("role_prompts", {})
        if not isinstance(roles_raw, dict):
            raise ValueError("role_prompts 必须是 角色名→提示词文件 的映射")
        role_prompts: dict[str, str] = {}
        role_files: list[Path] = []
        for role, value in roles_raw.items():
            if not isinstance(role, str) or ROLE_NAME.fullmatch(role) is None:
                raise ValueError(f"非法角色名：{role!r}")
            role_path = self._path(base, value)
            role_text = role_path.read_text(encoding="utf-8").strip()
            if not role_text:
                raise ValueError(f"角色稿为空：{role_path}")
            role_prompts[role] = role_text
            role_files.append(role_path)
        skill_selection = parse_skill_selection(data["skills"]) if "skills" in data else None
        mcp_selection = parse_mcp_selection(data["mcp"]) if "mcp" in data else None
        agent_files: list[Path] = []

        def agent_prompt(value):
            path = self._path(base, value)
            agent_files.append(path)
            return path, path.read_text(encoding="utf-8")

        team = parse_team(
            data,
            read_prompt=agent_prompt,
            tool_names=TOOL_NAMES,
            startup_max_agents=baseline.max_subagents,
        )
        mcp_enabled = baseline.mcp_enabled and inherit_mcp
        mcp_path = baseline.mcp_config_file
        if "mcp_config_file" in data:
            mcp_enabled = data["mcp_config_file"] is not None
            if mcp_enabled:
                mcp_path = self._path(base, data["mcp_config_file"])
        startup_servers = self.baseline_servers
        if (
            mcp_enabled
            and mcp_path.is_file()
            and ("mcp_config_file" in data or startup_servers is None)
        ):
            _read_json(mcp_path)
        servers = (
            dict(startup_servers)
            if mcp_enabled and "mcp_config_file" not in data and startup_servers is not None
            else load_mcp_config(
                mcp_path,
                baseline.workspace,
                base_dir=base if "mcp_config_file" in data else None,
            )
            if mcp_enabled
            else {}
        )
        inherited_mcp = (
            inherit_mcp and "mcp_config_file" in data and mcp_enabled and baseline.mcp_enabled
        )
        if inherited_mcp:
            inherited = (
                dict(startup_servers)
                if startup_servers is not None
                else load_mcp_config(baseline.mcp_config_file, baseline.workspace)
            )
            inherited = {
                name: replace(server, cwd=server.cwd or baseline.workspace)
                for name, server in inherited.items()
            }
            servers = {**inherited, **servers}
        missing_mcp: tuple[str, ...] = ()
        mcp_scope = None
        if mcp_selection is not None:
            # The selection only chooses among configured servers; credentials stay in the
            # MCP config's ${VAR} references and never enter the environment definition.
            missing_mcp = tuple(sorted(set(mcp_selection) - set(servers)))
            servers = {
                name: replace(server, disabled=server.disabled or not mcp_selection[name].enabled)
                for name, server in servers.items()
                if name in mcp_selection
            }
            mcp_scope = selection_scope(mcp_selection)
        if role_source is not None:
            server = servers.get(role_source["server"])
            if server is None or server.disabled or not policy.permits("mcp"):
                raise ValueError("role_source 必须引用已启用且允许调用的 MCP 服务")
            if not role_prompts:
                servers[server.name] = replace(server, required=True)
        if "mcp_config_file" not in data:
            servers = {
                name: replace(server, cwd=server.cwd or baseline.workspace)
                for name, server in servers.items()
            }
        executables = []
        for name, server in servers.items():
            if server.command is None or server.disabled:
                continue
            command = server.command
            if "/" not in command:
                command = (
                    shutil.which(command, path=server.env.get("PATH", os.environ.get("PATH")))
                    or command
                )
            executable = Path(command).expanduser()
            if not executable.is_absolute():
                executable = (server.cwd or base) / executable
            if executable.is_file():
                servers[name] = replace(server, command=str(executable.absolute()))
                executables.append(executable.resolve())
        template_data, template_id, template_path = None, None, None
        if "opening_template" in data:
            selector = data["opening_template"]
            if not isinstance(selector, str) or not selector or self.templates is None:
                raise ValueError("opening_template 必须引用可用模板")
            matches = [
                template
                for template in self.templates.list()
                if selector in {template.name, template.template_id}
            ]
            if len(matches) != 1:
                raise ValueError(f"opening_template 没有唯一匹配：{selector}")
            template = matches[0]
            template_data = self.templates.read(template)
            template_id, template_path = template.template_id, template.path
        files = [entry.path, prompt, *self._files(skills), *role_files, *agent_files]
        builtin_skills = baseline.builtin_skills_dir.expanduser().resolve()
        uses_builtin_skills = (
            skill_selection is not None and startup_skills_enabled and builtin_skills.is_dir()
        )
        if uses_builtin_skills:
            files.extend(self._files((builtin_skills,)))
        inherited_skill_roots = []
        inherited_explicit_roots = set()
        if inherit_skills and startup_skills_enabled:
            catalog = self.baseline_skills
            if catalog is None:
                inherited_skill_roots = skill_roots(baseline.workspace, baseline.skills_dir)
            elif catalog.enabled:
                inherited_skill_roots = catalog.roots
                inherited_explicit_roots = {
                    (baseline.workspace / root.expanduser()).resolve()
                    for root in catalog.extra_dirs
                }
        for root in inherited_skill_roots:
            files.extend(
                skill_files(
                    root,
                    include_root_files=root in inherited_explicit_roots
                    or root.parts[-2:] != (".agents", "skills"),
                )
            )
        if mcp_enabled and mcp_path.is_file():
            files.append(mcp_path)
        if inherited_mcp and baseline.mcp_config_file.is_file():
            files.append(baseline.mcp_config_file)
        if template_path is not None:
            files.append(template_path)
        if status_file in files:
            raise ValueError("status_file 不能覆盖环境能力包资源")
        for server in servers.values():
            for argument in server.args:
                candidate = Path(argument).expanduser()
                if not candidate.is_absolute():
                    candidate = (server.cwd or base) / candidate
                if candidate.is_file():
                    files.append(candidate)
                    if candidate.suffix in {".py", ".js", ".mjs", ".cjs", ".sh"}:
                        files.extend(local_code_files(candidate.parent))
            command = Path(server.command or "")
            if command.is_file() and command.suffix in {".py", ".js", ".mjs", ".cjs", ".sh"}:
                files.extend(local_code_files(command.parent))
        files = sorted(set(files))
        if len(files) > 4096 or sum(path.stat().st_size for path in files) > 32 * 1024 * 1024:
            raise ValueError("环境能力包超过限制（4096 个文件 / 32 MiB）")
        resources = [
            (str(path.resolve()), cached_file_digest(path, self._digest_cache))
            for path in sorted(set(files))
        ]
        if sum(path.stat().st_size for path in set(executables)) > 512 * 1024 * 1024:
            raise ValueError("MCP 可执行文件超过指纹校验限制（512 MiB）")
        executable_hashes = [
            (str(path), cached_file_digest(path, self._digest_cache))
            for path in sorted(set(executables))
        ]
        expanded = {
            name: {
                "command": server.command,
                "args": server.args,
                "env": server.env,
                "cwd": str(server.cwd or base),
                "url": server.url,
                "headers": server.headers,
                "transport": server.transport,
                "disabled": server.disabled,
                "timeout": server.timeout,
                "required": server.required,
                **({"mode": server.mode} if server.mode != "auto" else {}),
            }
            for name, server in servers.items()
        }
        fingerprint = hashlib.sha256(
            json.dumps(
                {
                    "resources": resources,
                    "executables": executable_hashes,
                    "mcp": expanded,
                    "definition": data,
                    "prompt": prompt_text,
                    "skills": list(map(str, skills)),
                    "template": template_data,
                    "effective": {
                        "effort": parse_reasoning_effort(effort),
                        "websearch": websearch,
                        "max_subagents": workers,
                        "mcp_enabled": mcp_enabled,
                        "skills_enabled": startup_skills_enabled
                        and (inherit_skills or bool(skills) or skill_selection is not None),
                        "inherit_mcp": inherit_mcp,
                        "inherited_skill_roots": list(map(str, inherited_skill_roots)),
                        **({"status_file": str(status_file)} if status_file is not None else {}),
                        **({"role_source": role_source} if role_source is not None else {}),
                    },
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        settings = replace(
            baseline,
            prompt_file=prompt,
            reasoning_effort=parse_reasoning_effort(effort),
            skills_enabled=startup_skills_enabled
            and (inherit_skills or bool(skills) or skill_selection is not None),
            websearch_enabled=websearch,
            max_subagents=workers,
            mcp_enabled=mcp_enabled,
            mcp_config_file=mcp_path,
        )
        return ResolvedEnvironment(
            entry,
            description,
            fingerprint,
            settings,
            prompt_text,
            skills,
            policy,
            servers,
            worker_tools,
            role_prompts,
            template_data,
            template_id,
            template_path,
            status_file,
            role_source,
            inherit_mcp=inherit_mcp,
            inherit_skills=inherit_skills,
            skill_selection=skill_selection,
            mcp_selection=mcp_selection,
            mcp_scope=mcp_scope,
            missing_mcp=missing_mcp,
            team=team,
            notes=(
                *(f"MCP 服务未在任何来源中配置：{name}" for name in missing_mcp),
                *capability_notes(team, skill_selection, mcp_scope),
            ),
        )

    def startup_skills_enabled(self) -> bool:
        if self.baseline_skills is not None:
            return self.baseline_skills.enabled
        return self.baseline.skills_enabled

    def skill_catalog(
        self,
        environment: ResolvedEnvironment,
        *,
        selected: bool = True,
        baseline: SkillCatalog | None = None,
    ) -> SkillCatalog:
        """The environment's skills; selected=False lists every skill it could select."""
        settings = environment.settings
        baseline = baseline if baseline is not None else self.baseline_skills
        inherited = baseline if environment.inherit_skills else None
        return SkillCatalog(
            settings.workspace,
            settings.skills_dir,
            enabled=settings.skills_enabled if selected else self.startup_skills_enabled(),
            extra_dirs=environment.skills_dirs
            + (inherited.extra_dirs if inherited is not None and inherited.enabled else ()),
            include_defaults=environment.inherit_skills
            and (inherited is None or (inherited.enabled and inherited.include_defaults)),
            # Built-in collaboration skills are a lowest-priority source that only an
            # explicit skill selection can activate.
            fallback_dirs=(settings.builtin_skills_dir,)
            if environment.skill_selection is not None or not selected
            else (),
            only=environment.active_skills if selected else None,
        )

    @staticmethod
    def _status_path(base: Path, value) -> Path | None:
        if value is None:
            return None
        if not isinstance(value, str) or not value.strip() or "\x00" in value:
            raise ValueError("status_file 必须是非空 JSON 文件路径")
        base = base.resolve()
        raw = Path(value).expanduser()
        path = (raw if raw.is_absolute() else base / raw).resolve()
        if not raw.is_absolute() and (".." in raw.parts or not path.is_relative_to(base)):
            raise ValueError("status_file 相对路径不能逃逸环境目录")
        if path.suffix != ".json" or path.is_dir():
            raise ValueError("status_file 必须指向 JSON 文件，不能是目录")
        return path

    @staticmethod
    def _role_source(value) -> dict | None:
        if value is None:
            return None
        defaults = {
            "field": "prompt",
            "cache_seconds": 60,
            "max_bytes": 65536,
            "timeout_seconds": 10,
        }
        if not isinstance(value, dict) or set(value) - {"server", "tool", *defaults}:
            raise ValueError("role_source 包含不支持的字段或不是对象")
        source = {**defaults, **value}
        for name in ("server", "tool", "field"):
            item = source.get(name)
            if not isinstance(item, str) or not item.strip() or len(item) > 128 or "\x00" in item:
                raise ValueError(f"role_source.{name} 必须是非空字符串")
        for name, minimum, maximum in (
            ("cache_seconds", 0, 3600),
            ("max_bytes", 1, 1048576),
            ("timeout_seconds", 1, 10),
        ):
            item = source[name]
            if type(item) is not int or not minimum <= item <= maximum:
                raise ValueError(f"role_source.{name} 必须在 {minimum} 到 {maximum} 之间")
        return source

    @contextmanager
    def _locked(self):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with (self.state_dir / "trust.lock").open("a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)

    def _records(self) -> dict:
        if not self.trust_file.exists():
            return {}
        data = _read_json(self.trust_file)
        if data.get("version") != 1 or not isinstance(data.get("records"), dict):
            raise ValueError("环境信任记录损坏，未自动重置")
        return data["records"]

    def approved(self, environment: ResolvedEnvironment) -> bool:
        record = self._records().get(str(environment.entry.path), {})
        return (
            isinstance(record, dict)
            and not record.get("denied")
            and record.get("fingerprint") == environment.fingerprint
            and record.get("id") == environment.environment_id
        )

    def require_approved(self, environment: ResolvedEnvironment) -> None:
        if not self.approved(environment):
            action = "trust" if environment.entry.source == "workspace" else "确认"
            raise ValueError(f"环境未{action}或资源已改变：{environment.environment_id}")

    def approve(self, environment: ResolvedEnvironment) -> None:
        self.verify(environment)
        with self._locked():
            records = self._records()
            records[str(environment.entry.path)] = {
                "id": environment.environment_id,
                "fingerprint": environment.fingerprint,
                "denied": False,
                "confirmed_at": datetime.now(timezone.utc).isoformat(),
            }
            self.audit("trust", environment.environment_id, fingerprint=environment.fingerprint)
            atomic_write_text(self.trust_file, json.dumps({"version": 1, "records": records}))
            self.trust_file.chmod(0o600)

    def deny(self, selector: str) -> None:
        entry = self.find(selector)
        with self._locked():
            records = self._records()
            records[str(entry.path)] = {"id": entry.environment_id, "denied": True}
            self.audit("deny", entry.environment_id)
            atomic_write_text(self.trust_file, json.dumps({"version": 1, "records": records}))
            self.trust_file.chmod(0o600)

    def denied(self, environment: ResolvedEnvironment) -> bool:
        record = self._records().get(str(environment.entry.path), {})
        return not isinstance(record, dict) or bool(record.get("denied"))

    def verify(self, environment: ResolvedEnvironment) -> ResolvedEnvironment:
        current = self.resolve(environment.environment_id)
        if current.reference != environment.reference:
            raise ValueError("环境资源在确认或使用后改变，请重新 trust/切换")
        return current

    def audit(self, event: str, environment_id: str | None, **details) -> str:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        event_id = uuid4().hex
        record = {
            "event_id": event_id,
            "at": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "environment_id": environment_id,
            "workspace": str(self.baseline.workspace),
            **details,
        }
        descriptor = os.open(self.audit_file, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return event_id

    @staticmethod
    def summary(environment: ResolvedEnvironment) -> str:
        settings = environment.settings
        policy = environment.policy
        lines = [
            f"环境：{environment.environment_id} · {environment.description}",
            f"定义：{environment.entry.path}",
            f"能力包 SHA256：{environment.fingerprint}",
            f"prompt：{settings.prompt_file}",
            "环境 skills 目录：" + (", ".join(map(str, environment.skills_dirs)) or "无"),
            f"skills 加载：{'开启' if settings.skills_enabled else '关闭'}",
            f"继承启动 skills：{'是' if environment.inherit_skills else '否'}",
            f"继承启动 MCP：{'是' if environment.inherit_mcp else '否'}；"
            f"启用 {sum(not server.disabled for server in environment.servers.values())} 个服务",
            f"effort：{settings.reasoning_effort}；子代理上限：{settings.max_subagents}",
            f"websearch：{'开启（需启动凭据）' if settings.websearch_enabled else '关闭'}",
            "tools.allow："
            + (", ".join(sorted(policy.allow)) if policy.allow is not None else "启动工具面"),
            "tools.deny：" + (", ".join(sorted(policy.deny)) or "无"),
            "子代理额外工具："
            + (", ".join(sorted(environment.worker_tools)) or "无（仅只读工具）"),
            "子代理角色稿：" + (", ".join(sorted(environment.role_prompts)) or "无"),
            "角色稿来源："
            + (
                f"从资料库活取（{environment.role_source['server']}/{environment.role_source['tool']}；"
                f"缓存 {environment.role_source['cache_seconds']} 秒；静态文件兜底；运行时内容不纳入指纹）"
                if environment.role_source
                else "静态文件"
            ),
            f"子代理状态文件：{environment.status_file or '未启用'}",
            f"MCP 配置：{settings.mcp_config_file if settings.mcp_enabled else '关闭'}",
            f"类型：{environment.team.kind}",
        ]
        if environment.skill_selection is not None:
            lines.append(
                "技能选择："
                + (
                    ", ".join(
                        f"{name}{'' if enabled else '（停用）'}"
                        for name, enabled in environment.skill_selection.items()
                    )
                    or "无"
                )
            )
        if environment.mcp_selection is not None:
            lines.append(
                "MCP 选择："
                + (
                    ", ".join(
                        name
                        + ("" if item.enabled else "（停用）")
                        + (f" tools={','.join(item.tools)}" if item.tools is not None else "")
                        + (f" deny={','.join(item.deny)}" if item.deny else "")
                        for name, item in environment.mcp_selection.items()
                    )
                    or "无"
                )
            )
        team = environment.team
        if team.multi_agent:
            limits = team.limits
            lines.append(
                f"成员：{len(team.agents)} 个；并发上限 {limits.max_concurrent_agents}；"
                f"每轮消息上限 {format_limit(limits.max_messages_per_turn)}；消息链上限 {limits.max_message_hops}；"
                f"会议轮数上限 {limits.max_meeting_rounds}；任务嵌套上限 {limits.max_task_depth}"
            )
            for agent in team.agents:
                summary = agent.summary()
                lines.append(
                    f"  成员 {agent.name}{'' if agent.enabled else '（停用）'}：{agent.description}；"
                    f"tools={summary['tools']}；skills={summary['skills']}；mcp={summary['mcp']}"
                )
            lines.append("协作协议：" + ", ".join(sorted(team.collaboration.protocols)))
        lines.extend(f"注意：{note}" for note in environment.notes)
        for name, server in environment.servers.items():
            if server.command:
                arguments = []
                secret = False
                for item in server.args:
                    sensitive = bool(
                        re.search(r"token|password|secret|api[-_]?key|authorization", item, re.I)
                    )
                    arguments.append("<redacted>" if secret or sensitive else item)
                    secret = sensitive and "=" not in item
                target = shlex.join([server.command, *arguments])
            else:
                parsed = urlsplit(server.url or "")
                target = urlunsplit(
                    (parsed.scheme, parsed.netloc.rsplit("@", 1)[-1], parsed.path, "", "")
                )
            lines.append(
                f"  MCP {name} [{'disabled' if server.disabled else server.transport}]：{target}；"
                f"cwd={server.cwd or environment.entry.path.parent}；"
                f"env keys={','.join(sorted(server.env)) or '无'}；"
                f"header keys={','.join(sorted(server.headers)) or '无'}"
            )
        if environment.template_data is not None:
            lines.append(
                f"开场模板：{environment.template_data.get('name') or environment.template_id or '导入模板'}"
                f" · {environment.template_path}"
            )
        return "\n".join(
            "".join(
                character
                if ord(character) >= 32 and ord(character) != 127
                else f"\\x{ord(character):02x}"
                for character in line
            )
            for line in lines
        )
