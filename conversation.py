import json
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timezone
from uuid import uuid4

from config import Settings, environment_context, parse_reasoning_effort, system_prompt
from context_budget import WireCache, context_snapshot
from context_management import compact_messages, summarize
from contracts import (
    CompletionClient,
    ContentCallback,
    Message,
    ProgressCallback,
    ToolCall,
    ToolDefinition,
    ToolResultPayload,
    ToolRun,
)
from environment_transition import EnvironmentTransition
from environments import EnvironmentStore, ResolvedEnvironment
from execution_trace import (
    EventSink,
    TraceContext,
    display_callback,
    emit_event,
    trace_scope,
    traced_progress,
)
from mcp_client import McpManager
from sessions import SessionStore
from skills import SkillCatalog
from task_acceptance import TaskAcceptance, TaskContract
from task_runtime import RUNTIME_INSTRUCTIONS, TaskRuntime
from team_runtime import TeamRuntime
from templates import SavedTemplate, TemplateStore
from tool_runtime import ToolResult, validate
from tools import CodingTools, ToolPolicy
from usage import CacheStats
from workflows import WorkflowStore


class Conversation:
    def __init__(
        self,
        settings: Settings,
        client: CompletionClient,
        tools: CodingTools | None,
        *,
        system_instructions: str | None = None,
        skills: SkillCatalog | None = None,
        mcp_manager: McpManager | None = None,
        event_sink: EventSink | None = None,
    ) -> None:
        self.settings, self.client, self.tools = settings, client, tools
        self.event_sink = event_sink
        self.last_task_trace: TraceContext | None = None
        self.system_instructions = system_instructions
        self.skills = skills
        if tools is not None:
            tools.on_execution = self._observe_execution
            tools.attach_reasoning(client, max_workers=settings.max_subagents)
        self._operation = threading.Lock()
        self.environment_id: str | None = None
        self.environment_ref: dict | None = None
        self.last_environment_id: str | None = None
        self.last_environment_ref: dict | None = None
        self.active_environment: ResolvedEnvironment | None = None
        # Multi-agent environments attach a team runtime; None keeps single-agent behavior.
        self.team: TeamRuntime | None = None
        # Interactive sessions let members keep working between the user's messages.
        self.team_background = False
        if mcp_manager is None and tools is not None and "mcp" in tools.registry:
            mcp_manager = tools.registry["mcp"].handler.__self__
        self.mcp_manager = mcp_manager
        self._baseline_settings = settings
        self._baseline_instructions = system_instructions
        self._baseline_skills = skills
        self._baseline_registry = dict(tools.registry) if tools is not None else {}
        self._baseline_policy = tools.policy if tools is not None else ToolPolicy()
        configured = getattr(mcp_manager, "configured_servers", None)
        self._baseline_servers = (
            deepcopy(configured if isinstance(configured, dict) else mcp_manager.servers)
            if mcp_manager is not None
            else {}
        )
        self.system_message: Message = {"role": "system", "content": ""}
        self.reload_prompt()
        self.tool_runs: list[ToolRun] = []
        self._tool_runs_lock = threading.RLock()
        self._wire_cache = WireCache()
        self._execution_title = ""
        self._execution_closed = False
        self.recovery_notice = None
        self.last_outcome = None
        self.sessions = SessionStore(settings.session_file)
        self.templates = TemplateStore(
            self.sessions, settings.workspace, directory=settings.templates_dir
        )
        self.workflows = WorkflowStore(settings.workflows_dir)
        self.environments = EnvironmentStore(
            settings,
            self.templates,
            baseline_skills=skills,
            baseline_servers=self._baseline_servers if mcp_manager is not None else None,
        )
        self.session_id = uuid4().hex
        self.template_id: str | None = None
        self.messages: list[Message] = self._load()
        repaired = self._repair_incomplete_tools()
        if repaired or (
            self.messages and not self.sessions.snapshot_path(self.session_id).exists()
        ):
            self.save()

    @property
    def definitions(self) -> list[ToolDefinition]:
        return self.tools.definitions if self.tools is not None else []

    @contextmanager
    def _exclusive_operation(self, message: str):
        if not self._operation.acquire(blocking=False):
            raise RuntimeError(message)
        try:
            yield
        finally:
            self._operation.release()

    def _runtime_system_message(self) -> Message:
        return {
            **self.system_message,
            "content": self.system_message["content"] + "\n\n" + RUNTIME_INSTRUCTIONS,
        }

    def context_status(self) -> dict:
        return context_snapshot(
            [self._runtime_system_message(), *self.messages],
            self.definitions,
            self.settings,
            self._wire_cache,
        )

    def _load(self) -> list[Message]:
        if not self.settings.session_file.exists():
            return []
        data = self.sessions.read()
        if data.get("template_id") is not None:
            if data["messages"]:
                data.setdefault("session_id", uuid4().hex)
                data.setdefault("updated_at", datetime.now(timezone.utc).isoformat())
                self.sessions.save(data)
            self.sessions.remove_active()
            return self._restore({"messages": []})
        return self._restore(data)

    def _validate_task_state(self, data: dict) -> None:
        try:
            if self.tools is not None:
                state = data.get("task_state") or {}
                plan = state.get("plan")
                if plan is not None:
                    validate(plan, self.tools.registry["update_plan"].schema)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Cannot load session file {self.settings.session_file}: {exc}"
            ) from exc

    def _restore(self, data: dict) -> list[Message]:
        self._validate_task_state(data)
        self.session_id = data.get("session_id") or self.session_id
        self.template_id = data.get("template_id")
        self.last_environment_id = data.get("environment_id")
        self.last_environment_ref = data.get("environment_ref")
        self.tool_runs = data.get("tool_runs", [])
        self.last_outcome = data.get("last_outcome")
        self.recovery_notice = None
        if self.tools is not None:
            self.tools.reset_task()
            state = data.get("task_state") or {}
            self.tools.task_title = str(state.get("title") or "")
            self.tools.task_plan = state.get("plan")
        stats = getattr(self.client, "cache_stats", None)
        if isinstance(stats, CacheStats):
            stats.reset()
            stats.restore(data.get("cache_stats"))
        return data["messages"]

    def resume(self, session_id: str, *, messages_only: bool = False) -> None:
        with self._exclusive_operation("任务或环境切换正在运行，不能恢复会话"):
            self._resume(session_id, messages_only=messages_only)

    def _resume(self, session_id: str, *, messages_only: bool = False) -> None:
        if session_id == self.session_id and messages_only:
            data = self._session_data()
        elif session_id == self.session_id:
            can_keep = self.last_environment_id is None
            if self.active_environment is not None:
                try:
                    self.environments.require_approved(self.active_environment)
                    self.environments.verify(self.active_environment)
                    can_keep = True
                except (OSError, ValueError):
                    can_keep = False
            if can_keep:
                if self.messages:
                    self.save()
                return
            data = self._session_data()
        else:
            data = self.sessions.read(self.sessions.snapshot_path(session_id))
        if data.get("session_id") != session_id:
            raise ValueError("session ID does not match file name")
        self._validate_task_state(data)
        if messages_only:
            data = deepcopy(data)
            data["session_id"] = uuid4().hex
            for field in ("cache_stats", "template_id", "tool_runs", "task_state", "last_outcome"):
                data.pop(field, None)
            self._apply_environment(
                self.active_environment, keep=False, restore=data, on_connect=None
            )
            self.recovery_notice = (
                "仅复制历史消息到新会话，保持当前能力；历史工具会话状态没有恢复。"
            )
            return
        if (
            not data.get("environment_id")
            and self.environment_id is None
            and self.last_environment_id is None
        ):
            if self.messages:
                self.save()
            self.messages = self._restore(data)
            self._repair_incomplete_tools()
            self.save()
            return
        environment = None
        selector = data.get("environment_id")
        if selector:
            try:
                environment = self.environments.resolve(selector)
                self.environments.require_approved(environment)
                if data.get("environment_ref") and environment.reference != data["environment_ref"]:
                    raise ValueError("环境来源或资源指纹与会话快照不匹配")
            except (OSError, RuntimeError, ValueError) as exc:
                raise ValueError(
                    f"环境 {selector} 不可安全恢复（{exc}）；当前能力与会话保持不变。"
                    "重新确认匹配的环境后恢复，或显式使用 /resume ID --messages-only "
                    "仅复制消息并保持当前能力。"
                ) from exc
        warnings = self._apply_environment(environment, keep=False, restore=data, on_connect=None)
        if warnings:
            self.recovery_notice = "\n".join(filter(None, (self.recovery_notice, *warnings)))

    def _repair_incomplete_tools(self) -> bool:
        repaired, missing = [], 0
        for run in self.tool_runs:
            if (run.get("trace") or {}).get("agent_id") and run.get("state") in {
                "pending",
                "running",
            }:
                run.update(
                    state="unknown",
                    result=self._tool_error(
                        "execution_unknown",
                        "The worker stopped before its result was saved.",
                        "unknown",
                    ),
                )
                missing += 1
        index = 0
        while index < len(self.messages):
            message = self.messages[index]
            repaired.append(message)
            assistant_index = index
            index += 1
            calls = message.get("tool_calls") if message.get("role") == "assistant" else None
            if not calls:
                continue
            results = []
            while index < len(self.messages) and self.messages[index]["role"] == "tool":
                results.append(self.messages[index])
                index += 1
            repaired.extend(results)
            answered = {m.get("tool_call_id") for m in results}
            for call in calls:
                call_id = call.get("id", "")
                if call_id in answered:
                    continue
                run = next(
                    (
                        r
                        for r in reversed(self.tool_runs)
                        if r.get("assistant_index") == assistant_index
                        and r.get("tool_call_id") == call_id
                    ),
                    None,
                )
                pending = run is not None and run.get("state") == "pending"
                result = self._tool_error(
                    "not_executed" if pending else "execution_unknown",
                    "The process stopped before this tool started. It was not executed."
                    if pending
                    else "The process stopped before the result was saved. The operation may already have taken effect. "
                    "Inspect the actual files or external state before deciding what to do. Do not blindly repeat this operation.",
                    "unknown" if not pending else "error",
                )
                repaired.append({"role": "tool", "tool_call_id": call_id, "content": result})
                if run is not None:
                    run.update(state="not_executed" if pending else "unknown", result=result)
                missing += 1
        if missing:
            self.messages = repaired
            self.last_outcome = {**(self.last_outcome or {}), "status": "incomplete"}
            self.recovery_notice = (
                f"已恢复 {missing} 个中断的工具调用；执行结果未知的操作需要先核实实际状态。"
            )
        return bool(missing)

    def _session_data(self) -> dict:
        data = {
            "version": 1,
            "session_id": self.session_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "messages": list(self.messages),
            "tool_runs": self.execution_snapshot(),
            "last_outcome": self.last_outcome,
        }
        if self.template_id is not None:
            data["template_id"] = self.template_id
        environment_id = self.environment_id or self.last_environment_id
        reference = self.environment_ref or self.last_environment_ref
        if environment_id is not None:
            data["environment_id"] = environment_id
            if reference is not None:
                data["environment_ref"] = reference
        if self.tools is not None:
            data["task_state"] = {"title": self.tools.task_title, "plan": self.tools.task_plan}
        stats = getattr(self.client, "cache_stats", None)
        if isinstance(stats, CacheStats):
            data["cache_stats"] = stats.snapshot()
        return data

    def save(self) -> None:
        with self._tool_runs_lock:
            self.sessions.save(self._session_data())

    def clear(self) -> None:
        template = self.templates.default()
        data = self.templates.read(template) if template is not None else None
        self._start_new(data, template.template_id if template is not None else None)

    def clear_template(self) -> None:
        self._start_new(None, None)
        self.templates.set_default(None)

    def use_template(self, template: SavedTemplate) -> SavedTemplate:
        active_file = self.settings.session_file.resolve()
        if template.path.resolve() == active_file or (
            template.source is not None and template.source.resolve() == active_file
        ):
            raise ValueError(
                "模板文件不能同时作为当前会话文件；请使用默认会话文件启动后再选择模板。"
            )
        if template.template_id is None:
            template = self.templates.add(template.path, template.name)
        data = self.templates.read(template)
        self._start_new(data, template.template_id)
        self.templates.set_default(template.template_id)
        return template

    def _start_new(self, data: dict | None, template_id: str | None) -> None:
        messages = deepcopy(data["messages"]) if data is not None else []
        if self.messages:
            self.save()
        if not messages:
            self.sessions.remove_active()
        self.messages.clear()
        self.tool_runs.clear()
        self.session_id = uuid4().hex
        self.template_id = template_id
        self.recovery_notice = None
        self.last_outcome = None
        self.last_environment_id = self.environment_id
        self.last_environment_ref = self.environment_ref
        if self.tools is not None:
            self.tools.reset_task()
        if self.team is not None:
            # A new conversation session is a new, isolated environment session.
            try:
                self.recovery_notice = "\n".join(self.team.open_session(self.session_id)) or None
            except (OSError, RuntimeError, ValueError) as exc:
                self.recovery_notice = f"团队会话无法打开，成员暂不可用：{exc}"
        stats = getattr(self.client, "cache_stats", None)
        if isinstance(stats, CacheStats):
            stats.reset()
            self.client.cache_stats = CacheStats()
        if self.tools is not None:
            self.tools.attach_reasoning(self.client, max_workers=self.settings.max_subagents)
        self.messages = messages
        if self.messages:
            self._repair_incomplete_tools()
            self.save()
        elif self.environment_id is not None:
            self.save()

    def set_reasoning_effort(self, value: str) -> None:
        settings = replace(self.settings, reasoning_effort=parse_reasoning_effort(value))
        self.client.settings = settings
        self.settings = settings
        if self.tools is not None:
            self.tools.attach_reasoning(self.client, max_workers=settings.max_subagents)

    def apply_environment(
        self,
        environment: ResolvedEnvironment | None,
        *,
        keep: bool = False,
        restore: dict | None = None,
        on_connect=None,
    ) -> list[str]:
        with self._exclusive_operation("任务或环境切换正在运行，不能切换环境"):
            return self._apply_environment(
                environment, keep=keep, restore=restore, on_connect=on_connect
            )

    def _apply_environment(self, environment, *, keep, restore, on_connect) -> list[str]:
        return EnvironmentTransition(self, keep=keep, restore=restore, on_connect=on_connect).run(
            environment
        )

    def close(self) -> None:
        with self._tool_runs_lock:
            self._execution_closed = True
        if self.team is not None:
            self.team.close()
        if self.tools is not None:
            self.tools.close()
        if self.mcp_manager is not None:
            self.mcp_manager.close()

    @staticmethod
    def _message_size(messages: list[Message]) -> int:
        return len(json.dumps(messages, ensure_ascii=False))

    def _request_size(self, messages: list[Message]) -> int:
        return self._message_size(messages) + len(json.dumps(self.definitions, ensure_ascii=False))

    def _summarize(self, messages: list[Message], deadline: float | None = None) -> str:
        return summarize(self.client, self.settings, messages, deadline)

    def compact(self, deadline: float | None = None, on_progress=None) -> None:
        compact_messages(
            self.messages,
            self._runtime_system_message(),
            self.definitions,
            self.client,
            self.settings,
            deadline,
            on_progress=on_progress,
            cache=self._wire_cache,
        )
        self.save()

    @staticmethod
    def _compose_prompt(settings, tools, skills, instructions, prompt_text=None, team=None) -> str:
        names = (
            tuple(item["function"]["name"] for item in tools.definitions)
            if tools is not None
            else ()
        )
        content = instructions or system_prompt(
            settings.workspace,
            settings.prompt_file,
            names,
            instructions=prompt_text,
        )
        if tools is not None and tools.definitions:
            content += "\n\n" + tools.environment_instructions()
        else:
            content += "\nNo tools are available. Do not simulate tool calls; respond using the supplied information and your knowledge."
        if skills is not None:
            skill_section = skills.prompt_section()
            if skill_section:
                content += "\n\n" + skill_section
        if team is not None:
            content += "\n\n" + team.main_instructions()
        return content

    def reload_prompt(self) -> None:
        self.refresh_skills()
        self.system_message["content"] = self._compose_prompt(
            self.settings,
            self.tools,
            self.skills,
            self.system_instructions,
            self.active_environment.prompt_text if self.active_environment else None,
            self.team,
        )

    def refresh_skills(self) -> None:
        if self.active_environment is not None:
            self.environments.require_approved(self.active_environment)
            self.environments.verify(self.active_environment)
        if self.skills is None:
            return
        self.skills.reload()
        registry = self.skills.mcp_registry()
        if self.mcp_manager is not None:
            self.mcp_manager.update_skill_registry(registry)
        worker_skills = self.skills.prompt_section()
        if self.tools is not None and self.tools.worker_skills != worker_skills:
            self.tools.worker_skills = worker_skills
            self.tools.attach_reasoning(self.client, max_workers=self.settings.max_subagents)

    def environment_health(self) -> str:
        if self.mcp_manager is None or self.tools is None or not self.tools.policy.permits("mcp"):
            return "ready"
        servers = self.mcp_manager.status().data["servers"]
        missing = [
            server
            for server in servers
            if server.get("status") != "disabled"
            and not server.get("available", server.get("status") == "connected")
        ]
        if any(server.get("required") for server in missing):
            return "blocked"
        return "degraded" if missing else "ready"

    @staticmethod
    def _tool_error(code: str, message: str, status: str = "error") -> str:
        return json.dumps(ToolResult.failure(code, message, status).as_dict(), ensure_ascii=False)

    def ask(
        self,
        prompt: str,
        on_content: ContentCallback | None = None,
        on_response_start: Callable[[], None] | None = None,
        on_response_end: Callable[[], None] | None = None,
        final_validator: Callable[[str], str | None] | None = None,
        on_progress: ProgressCallback | None = None,
        task_contract: TaskContract | None = None,
        on_reasoning: ContentCallback | None = None,
        on_tool_result: Callable[[str, ToolResultPayload], None] | None = None,
    ) -> str:
        with self._exclusive_operation("任务或环境切换正在运行"):
            self._execution_title = " ".join(prompt.split())[:240]
            self.last_task_trace = TraceContext.task(self.session_id)
            stats = getattr(self.client, "cache_stats", None)
            scope = (
                stats.request_scope(f"{self.environment_id or 'default'}/main")
                if isinstance(stats, CacheStats)
                else nullcontext()
            )
            with scope, trace_scope(self.last_task_trace, self.event_sink):
                emit_event("task_started")
                try:
                    answer = self._ask(
                        prompt,
                        on_content=on_content,
                        on_response_start=on_response_start,
                        on_response_end=on_response_end,
                        final_validator=final_validator,
                        on_progress=traced_progress(on_progress),
                        task_contract=task_contract,
                        on_reasoning=on_reasoning,
                        on_tool_result=display_callback(on_tool_result, name="tool_result"),
                    )
                except BaseException as exc:
                    emit_event("task_failed", error_type=type(exc).__name__)
                    raise
                emit_event("task_completed")
                return answer

    def _save_outcome(self, acceptance: TaskAcceptance) -> None:
        self.last_outcome = acceptance.outcome()
        self.save()

    def _record_tool_calls(self, tool_calls: list[ToolCall], runtime: TaskRuntime) -> list[ToolRun]:
        runs: list[ToolRun] = []
        for call in tool_calls:
            execution_id = uuid4().hex
            runs.append(
                {
                    "execution_id": execution_id,
                    "assistant_index": len(self.messages) - 1,
                    "tool_call_id": call.get("id", ""),
                    "tool": call.get("function", {}).get("name"),
                    "arguments": call.get("function", {}).get("arguments", "{}"),
                    "task_title": self._execution_title,
                    "state": "pending",
                    "trace": runtime.last_request_trace.tool(call["id"], execution_id).as_dict(),
                }
            )
        with self._tool_runs_lock:
            self.tool_runs.extend(runs)
        self.save()
        return runs

    def execution_snapshot(self) -> list[ToolRun]:
        with self._tool_runs_lock:
            return [dict(run) for run in self.tool_runs]

    def _observe_execution(self, event: dict) -> None:
        trace = event["trace"]
        call = event["call"]
        with self._tool_runs_lock:
            if self._execution_closed or trace.get("session_id") != self.session_id:
                return
            run = next(
                (
                    item
                    for item in self.tool_runs
                    if item.get("execution_id") == trace.get("execution_id")
                ),
                None,
            )
            if run is None:
                parent = next(
                    (
                        item
                        for item in self.tool_runs
                        if item.get("execution_id") == trace.get("parent_execution_id")
                    ),
                    None,
                )
                run = {
                    "execution_id": trace["execution_id"],
                    "assistant_index": -1,
                    "tool_call_id": call.get("id", ""),
                    "tool": call.get("function", {}).get("name", ""),
                    "arguments": call.get("function", {}).get("arguments", "{}"),
                    "task_title": parent.get("task_title", self._execution_title)
                    if parent
                    else self._execution_title,
                    "trace": trace,
                    "state": "running",
                }
                self.tool_runs.append(run)
            if event["state"] == "started":
                run.update(state="running", started_at=event["timestamp"])
            else:
                run.update(
                    state="unknown" if event["state"] == "interrupted" else "completed",
                    finished_at=event["timestamp"],
                    duration_seconds=event["duration_seconds"],
                )
                if "result" in event:
                    run["result"] = event["result"]
            if trace.get("agent_id"):
                self.save()

    def _run_tool_call(
        self,
        tool_call: ToolCall,
        run: ToolRun,
        runtime: TaskRuntime,
        acceptance: TaskAcceptance,
        on_progress: ProgressCallback | None = None,
    ) -> tuple[str, bool]:
        try:
            run["state"] = "running"
            self.save()
            acceptance.before_tool(tool_call)
            tool_trace = runtime.last_request_trace.tool(tool_call["id"], run["execution_id"])
            run["trace"] = tool_trace.as_dict()
            with trace_scope(tool_trace):
                options = {"deadline": runtime.tool_deadline()}
                if on_progress is not None:
                    options["on_progress"] = traced_progress(on_progress)
                return self.tools.execute(tool_call, **options), False
        except KeyboardInterrupt:
            runtime.cancelled.set()
            return self._tool_error(
                "execution_unknown",
                "Tool interrupted by user. It may have taken effect. "
                "Inspect actual state before repeating this operation.",
                "unknown",
            ), True
        except Exception as exc:
            return self._tool_error(type(exc).__name__, str(exc)), False

    def _record_tool_result(
        self,
        tool_call: ToolCall,
        run: ToolRun,
        result: str,
        runtime: TaskRuntime,
        acceptance: TaskAcceptance,
        on_progress: ProgressCallback | None,
        on_tool_result: Callable[[str, ToolResultPayload], None] | None,
    ) -> None:
        parsed_result = json.loads(result)
        run.update(
            state="unknown" if parsed_result.get("status") == "unknown" else "completed",
            result=result,
        )
        self.messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.get("id", ""),
                "content": result,
            }
        )
        acceptance.observe(tool_call, parsed_result)
        self._save_outcome(acceptance)
        with trace_scope(TraceContext(**run["trace"])):
            notice = runtime.progress.observe(tool_call, parsed_result)
            if notice and on_progress:
                on_progress(notice)
            if on_tool_result:
                on_tool_result(tool_call.get("function", {}).get("name") or "", parsed_result)
        if on_progress:
            on_progress({"phase": "context", **self.context_status()})

    def _ask(
        self,
        prompt: str,
        on_content: ContentCallback | None = None,
        on_response_start: Callable[[], None] | None = None,
        on_response_end: Callable[[], None] | None = None,
        final_validator: Callable[[str], str | None] | None = None,
        on_progress: ProgressCallback | None = None,
        task_contract: TaskContract | None = None,
        on_reasoning: ContentCallback | None = None,
        on_tool_result: Callable[[str, ToolResultPayload], None] | None = None,
    ) -> str:
        deadline = (
            time.monotonic() + self.settings.max_task_seconds
            if self.settings.max_task_seconds is not None
            else float("inf")
        )
        self.reload_prompt()
        self.last_environment_id = self.environment_id
        self.last_environment_ref = self.environment_ref
        if self.skills is None and prompt.lstrip().startswith("/skill:"):
            raise ValueError("Skills 已关闭")
        expanded_prompt = self.skills.expand_command(prompt) if self.skills is not None else prompt
        if self.tools is not None:
            self.tools.reset_task(" ".join(prompt.split())[:120])
        acceptance = TaskAcceptance(
            prompt,
            task_contract or TaskContract(kind="answer" if final_validator else "auto"),
            has_tools=bool(self.definitions),
            workspace=self.settings.workspace,
            ignored_paths=(self.settings.session_file, self.sessions.directory),
            deadline=deadline,
            others=self.team.member_names() if self.team is not None else (),
        )
        note = acceptance.instructions()
        self.last_outcome = acceptance.outcome()

        def validate_final(content):
            if final_validator:
                problem = final_validator(content)
                if problem:
                    return problem
            background = False
            if self.team is not None:
                problem = self.team.final_answer_problem(handoff=acceptance.handoff_allowed)
                if problem:
                    return problem
                # Members' unfinished edits are verified when their work is done, not now.
                background = self.team.continues_in_background()
            return acceptance.validate(content, background=background)

        # Members work during the turn; afterwards they pause or, interactively, continue
        # in the background. Pending work resumes next turn. The team's turn starts before
        # the turn note so that it takes over background work first: nothing finishes in
        # the background after the note reports it.
        with (
            self.team.turn(deadline, background=self.team_background)
            if self.team is not None
            else nullcontext()
        ):
            team_note = self.team.turn_context() if self.team is not None else ""
            self.messages.append({"role": "user", "content": expanded_prompt})
            # Task-specific context belongs to this turn, never to the shared prefix.
            self.messages.append(
                {
                    "role": "system",
                    "content": environment_context(self.settings.workspace)
                    + ("\n\n" + note if note else "")
                    + ("\n\n" + team_note if team_note else ""),
                }
            )
            try:
                self.compact(deadline, on_progress)
            except BaseException:
                acceptance.status = "incomplete"
                self._save_outcome(acceptance)
                raise
            self.save()
            runtime = TaskRuntime(
                self.client, self.settings, deadline, validate_final, wire_cache=self._wire_cache
            )
            answer = self._run_turn(
                runtime,
                acceptance,
                deadline,
                on_content,
                on_response_start,
                on_response_end,
                on_progress,
                on_reasoning,
                on_tool_result,
            )
        if (
            self.team is not None
            and acceptance.status == "background"
            and not self.team.went_background
        ):
            # The members finished before the turn ended: nothing continues, nothing verified.
            acceptance.status = "incomplete"
            self._save_outcome(acceptance)
        return answer

    def _run_turn(
        self,
        runtime,
        acceptance,
        deadline,
        on_content,
        on_response_start,
        on_response_end,
        on_progress,
        on_reasoning,
        on_tool_result,
    ) -> str:
        while True:
            try:
                message = runtime.generate(
                    self.messages,
                    self.system_message,
                    self.definitions,
                    on_content,
                    on_response_start,
                    on_response_end,
                    on_progress,
                    on_reasoning,
                )
            except BaseException:
                acceptance.status = "incomplete"
                raise
            finally:
                self._save_outcome(acceptance)
            self.messages.append(message)
            tool_calls = message.get("tool_calls") or []
            runs = self._record_tool_calls(tool_calls, runtime)
            if on_progress:
                on_progress({"phase": "context", **self.context_status()})
            if not tool_calls:
                return message.get("content") or ""

            interrupted = False
            for tool_call, run in zip(tool_calls, runs, strict=True):
                remaining = deadline - time.monotonic()
                if interrupted:
                    result = self._tool_error("interrupted", "Tool skipped after user interruption")
                elif remaining <= 0:
                    result = self._tool_error(
                        "deadline_exceeded", "Task time limit reached", "timeout"
                    )
                else:
                    result, interrupted = self._run_tool_call(
                        tool_call, run, runtime, acceptance, on_progress
                    )
                self._record_tool_result(
                    tool_call, run, result, runtime, acceptance, on_progress, on_tool_result
                )
            if interrupted:
                acceptance.status = "incomplete"
                self._save_outcome(acceptance)
                raise KeyboardInterrupt
