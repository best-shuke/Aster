"""Prepare, validate, and commit capability changes as one reversible transition."""

import json
import time
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING
from uuid import uuid4

from client_runtime import ClientState, configure_client, fork_client
from config import Settings, system_prompt
from contracts import Message, ToolRun
from environments import ResolvedEnvironment
from mcp_client import McpManager
from reasoning import RolePromptCache
from skills import SkillCatalog
from tool_runtime import ExecutionContext
from tools import CodingTools
from usage import CacheStats

if TYPE_CHECKING:
    from conversation import Conversation


@dataclass(frozen=True)
class ConversationState:
    settings: Settings
    skills: SkillCatalog | None
    tools: CodingTools | None
    system_instructions: str | None
    system_message: Message
    environment_id: str | None
    environment_ref: dict | None
    active_environment: ResolvedEnvironment | None
    mcp_manager: McpManager | None
    last_environment_id: str | None
    last_environment_ref: dict | None
    messages: list[Message]
    template_id: str | None
    session_id: str
    tool_runs: list[ToolRun]
    last_outcome: dict | None
    recovery_notice: str | None

    @classmethod
    def capture(cls, conversation: "Conversation") -> "ConversationState":
        return cls(**{item.name: getattr(conversation, item.name) for item in fields(cls)})

    def restore(self, conversation: "Conversation") -> None:
        for item in fields(self):
            setattr(conversation, item.name, getattr(self, item.name))


@dataclass(frozen=True)
class PreparedEnvironment:
    environment: ResolvedEnvironment | None
    settings: Settings
    skills: SkillCatalog | None
    tools: CodingTools | None
    instructions: str | None
    prompt: str
    cache_stats: CacheStats | None


class McpRoleLoader:
    def __init__(self, manager: McpManager | None, source: dict):
        self.manager = manager
        self.source = dict(source)

    def __call__(self, role, execution):
        if self.manager is None:
            raise RuntimeError("Role prompt MCP is unavailable")
        source = self.source
        response = self.manager.call(source["tool"], {"role": role}, source["server"], execution)
        if response.status != "ok":
            raise RuntimeError("Role prompt MCP call failed")
        result = response.data["result"]
        payload = result.get("structuredContent")
        if not isinstance(payload, dict):
            texts = [
                item["text"] for item in result.get("content", []) if item.get("type") == "text"
            ]
            text = "\n".join(texts)
            if len(text.encode("utf-8")) > source["max_bytes"] * 4 + 8192:
                raise ValueError("Role prompt MCP response exceeds the size limit")
            payload = json.loads(text)
        if not isinstance(payload, dict) or payload.get("ok") is False:
            raise ValueError("Role prompt MCP returned no usable role")
        return payload.get(source["field"])


class EnvironmentTransition:
    def __init__(self, conversation: "Conversation", *, keep=False, restore=None, on_connect=None):
        self.conversation = conversation
        self.keep = keep
        self.restore_data = restore
        self.on_connect = on_connect
        self.manager: McpManager | None = None
        self.warnings: list[str] = []

    def run(self, environment: ResolvedEnvironment | None) -> list[str]:
        conversation = self.conversation
        if self.restore_data is None:
            current = conversation.active_environment
            if (
                (
                    (
                        current is None
                        and environment is None
                        and conversation.last_environment_id is None
                    )
                    or (current is not None and environment is not None and current == environment)
                )
                and conversation.environment_health() != "blocked"
                and conversation.settings
                == (environment.settings if environment else conversation._baseline_settings)
            ):
                self.validate(environment)
                conversation.environments.audit("unchanged", conversation.environment_id)
                return []
        environment_id = environment.environment_id if environment is not None else None
        attempt = conversation.environments.audit(
            "attempt",
            environment_id,
            previous_environment=conversation.environment_id,
            keep=self.keep,
            resume=self.restore_data is not None,
        )
        previous = ConversationState.capture(conversation)
        previous_data = conversation._session_data()
        client_state = ClientState.capture(conversation.client)
        try:
            prepared = self.prepare(environment, client_state)
            self.commit(prepared, previous_data)
        except BaseException as exc:
            self.rollback(previous, client_state, environment_id, attempt, exc)
            raise
        self.finish(prepared, previous, attempt)
        return self.warnings

    def validate(self, environment: ResolvedEnvironment | None) -> ResolvedEnvironment | None:
        if environment is None:
            return None
        store = self.conversation.environments
        store.require_approved(environment)
        return store.verify(environment)

    def prepare(self, environment, client_state: ClientState) -> PreparedEnvironment:
        conversation = self.conversation
        environment = self.validate(environment)
        settings = environment.settings if environment else conversation._baseline_settings
        skills = self.prepare_skills(environment, settings)
        tools = self.prepare_tools(environment, settings, skills)
        instructions = None if environment else conversation._baseline_instructions
        prompt = conversation._compose_prompt(
            settings,
            tools,
            skills,
            instructions,
            environment.prompt_text if environment else None,
        )
        old_stats = client_state.cache_stats
        stats = CacheStats() if isinstance(old_stats, CacheStats) else old_stats
        if self.keep and isinstance(stats, CacheStats):
            saved = old_stats.snapshot()
            stats.restore(saved)
            stats.latest = saved["latest"]
        if tools is not None:
            self.prepare_workers(tools, environment, settings, skills, stats)
        return PreparedEnvironment(
            environment, settings, skills, tools, instructions, prompt, stats
        )

    def prepare_skills(self, environment, settings) -> SkillCatalog | None:
        if environment is not None:
            baseline = self.conversation._baseline_skills
            inherited = baseline if environment.inherit_skills else None
            skills = SkillCatalog(
                settings.workspace,
                settings.skills_dir,
                enabled=settings.skills_enabled,
                extra_dirs=environment.skills_dirs
                + (inherited.extra_dirs if inherited is not None and inherited.enabled else ()),
                include_defaults=environment.inherit_skills
                and (inherited is None or (inherited.enabled and inherited.include_defaults)),
            )
        elif self.conversation._baseline_skills is not None:
            skills = self.conversation._baseline_skills.fork()
        else:
            skills = None
        if skills is not None:
            self.warnings.extend(
                f"Skill {item.path}: {item.message}" for item in skills.diagnostics
            )
        return skills

    def prepare_tools(self, environment, settings, skills) -> CodingTools | None:
        conversation = self.conversation
        tools = (
            conversation.tools.fork(registry=conversation._baseline_registry)
            if conversation.tools is not None
            else None
        )
        if tools is not None:
            tools.policy = (
                conversation._baseline_policy.narrowed(environment.policy)
                if environment
                else conversation._baseline_policy
            )
            tools.worker_allow = environment.worker_tools if environment else frozenset()
            tools.worker_roles = dict(environment.role_prompts) if environment else {}
            tools.role_loader = None
            tools.cache_scope = environment.environment_id if environment else "default"
            if not settings.websearch_enabled:
                tools.detach("websearch")
            tools.detach("mcp")
        servers = environment.servers if environment else deepcopy(conversation._baseline_servers)
        if settings.mcp_enabled and servers:
            self.manager = McpManager(
                settings.mcp_config_file,
                settings.workspace,
                servers=servers,
                skill_registry=skills.mcp_registry() if skills is not None else None,
                runtime_env={
                    "DS_AGENT_STATUS_FILE": str(
                        tools.worker_status.status_path(environment.status_file)
                    )
                    if tools is not None
                    and environment is not None
                    and environment.status_file is not None
                    else "",
                    "DS_AGENT_STATUS_OWNER": tools.worker_status.owner if tools is not None else "",
                },
                previous=conversation.mcp_manager,
            )
            if tools is not None and self.manager.server_count:
                tools.attach_mcp(self.manager)
                if tools.policy.permits("mcp"):
                    result = self.manager.connect_all(on_connect=self.on_connect)
                    self.warnings.extend(
                        f"MCP {failure['server']} 连接失败：{failure['error']}；使用时会自动重试。"
                        for failure in result.data["failures"]
                    )
                    required = [
                        server["name"]
                        for server in result.data.get("servers", [])
                        if server.get("required")
                        and server.get("status") != "disabled"
                        and not server.get("available")
                    ]
                    if required:
                        raise ValueError(
                            "必需 MCP 服务不可用，环境切换已取消：" + ", ".join(required)
                        )
        if tools is not None and environment is not None and environment.role_source is not None:
            source = environment.role_source
            try:
                self.manager.describe(
                    source["tool"],
                    source["server"],
                    ExecutionContext(time.monotonic() + source["timeout_seconds"]),
                )
            except Exception as exc:
                if not environment.role_prompts:
                    raise ValueError("必需动态角色工具不可用") from exc
                self.warnings.append("动态角色工具不可用，将使用环境的静态角色。")
            tools.role_loader = RolePromptCache(
                McpRoleLoader(self.manager, source),
                cache_seconds=source["cache_seconds"],
                max_bytes=source["max_bytes"],
                timeout_seconds=source["timeout_seconds"],
            )
        return tools

    def prepare_workers(self, tools, environment, settings, skills, stats) -> None:
        names = tuple(item["function"]["name"] for item in tools.worker_definitions)
        tools.worker_instructions = (
            system_prompt(
                settings.workspace,
                settings.prompt_file,
                names,
                instructions=environment.prompt_text,
            )
            if environment
            else ""
        )
        tools.worker_skills = skills.prompt_section() if skills else ""
        client = fork_client(self.conversation.client, settings, stats)
        tools.attach_reasoning(client, max_workers=settings.max_subagents)

    def commit(self, prepared: PreparedEnvironment, previous_data: dict) -> None:
        conversation = self.conversation
        environment = prepared.environment
        environment_id = environment.environment_id if environment else None
        conversation.settings = prepared.settings
        conversation.skills = prepared.skills
        conversation.tools = prepared.tools
        conversation.system_instructions = prepared.instructions
        conversation.system_message = {"role": "system", "content": prepared.prompt}
        configure_client(conversation.client, prepared.settings, prepared.cache_stats)
        conversation.environment_id = environment_id
        conversation.environment_ref = environment.reference if environment else None
        conversation.active_environment = environment
        conversation.mcp_manager = self.manager
        conversation.last_environment_id = environment_id
        conversation.last_environment_ref = conversation.environment_ref
        self.commit_messages(prepared)
        self.validate(environment)
        if self.manager is not None:
            self.manager.stage_skill_registry()
            if self.keep and previous_data.get("messages"):
                changed = sorted(
                    name
                    for name in self.manager.servers
                    if name not in self.manager.reused_servers
                    and not self.manager.servers[name].disabled
                )
                if changed:
                    notice = (
                        "MCP 工具连接已新建或重启："
                        + ", ".join(changed)
                        + "。历史工具会话状态不保证保留，请先核实实际状态。"
                    )
                    self.warnings.append(notice)
                    conversation.messages.append({"role": "user", "content": notice})
        conversation.sessions.transition(previous_data, conversation._session_data())

    def commit_messages(self, prepared: PreparedEnvironment) -> None:
        conversation = self.conversation
        environment = prepared.environment
        if self.restore_data is not None:
            conversation.messages = conversation._restore(deepcopy(self.restore_data))
            conversation.last_environment_id = conversation.environment_id
            conversation.last_environment_ref = conversation.environment_ref
            conversation._repair_incomplete_tools()
        elif not self.keep:
            conversation.messages = (
                deepcopy(environment.template_data["messages"])
                if environment and environment.template_data
                else []
            )
            conversation.template_id = None
            conversation.session_id = uuid4().hex
            conversation.tool_runs = []
            conversation.last_outcome = None
            conversation.recovery_notice = None
            if prepared.tools is not None:
                prepared.tools.reset_task()
            conversation._repair_incomplete_tools()

    def rollback(self, previous, client_state, environment_id, attempt, error) -> None:
        previous.restore(self.conversation)
        client_state.restore(self.conversation.client)
        cleanup_error = None
        if self.manager is not None:
            try:
                self.manager.close()
            except Exception as exc:
                cleanup_error = exc
        with suppress(OSError):
            self.conversation.environments.audit(
                "failed",
                environment_id,
                attempt=attempt,
                error_type=type(error).__name__,
                cleanup_failed=cleanup_error is not None,
            )
        if cleanup_error is not None:
            raise RuntimeError(f"环境切换已回滚，但候选 MCP 清理失败：{cleanup_error}") from error

    def finish(self, prepared: PreparedEnvironment, previous: ConversationState, attempt) -> None:
        conversation = self.conversation
        environment = prepared.environment
        if self.manager is not None:
            self.manager.commit_reuse()
        if previous.tools is not None and prepared.tools is not None:
            try:
                previous.tools.apply_configuration(prepared.tools)
                conversation.tools = previous.tools
            except Exception as exc:
                self.warnings.append(f"环境已提交，旧工具对象同步失败，继续使用新工具对象：{exc}")
        if conversation.tools is not None:
            try:
                stopped = conversation.tools.processes.stop_all()
                if stopped:
                    self.warnings.append(
                        f"环境已切换，已停止 {stopped} 个后台命令；保留的日志仍可查看。"
                    )
            except Exception as exc:
                self.warnings.append(f"环境已提交，但后台命令清理失败：{exc}")
            try:
                conversation.tools.worker_status.configure(
                    environment.status_file if environment else None,
                    min(prepared.settings.max_subagents, conversation.tools._global_worker_limit),
                )
                if conversation.tools.worker_status.last_error:
                    self.warnings.append(conversation.tools.worker_status.last_error)
            except Exception as exc:
                self.warnings.append(f"环境已提交，但子代理状态同步失败：{exc}")
        if previous.mcp_manager is not None:
            try:
                previous.mcp_manager.close()
            except Exception as exc:
                self.warnings.append(f"环境已提交，但旧 MCP 清理失败：{exc}")
        try:
            conversation.environments.audit(
                "applied",
                conversation.environment_id,
                attempt=attempt,
                session_id=conversation.session_id,
                fingerprint=environment.fingerprint if environment else None,
            )
        except OSError as exc:
            self.warnings.append(f"环境已提交，但完成审计写入失败：{exc}")
