"""One team member: least-privilege capabilities and a persistent activation loop.

Each member has its own model client, private context and tool set. Its tools are the
environment's tools intersected with its own declaration; its MCP scope is the
environment's scope intersected with its own; its skills are the environment's active
skills it declares. Activations reuse the main agent's TaskRuntime for generation,
compaction, recovery, deadlines and cancellation.
"""

import json
from contextlib import nullcontext, suppress
from dataclasses import replace
from typing import TYPE_CHECKING

from client_runtime import fork_client
from context_budget import LOCAL_METADATA, estimate_tokens
from deepseek_client import CompletionError
from environment_spec import MEMBER_DEFAULT_TOOLS, MEMBER_EXCLUDED, TEAM_TOOL, AgentSpec
from execution_trace import TraceContext, trace_scope
from mcp_scope import McpScope, ScopedMcp
from skills import skills_prompt_section
from task_runtime import TaskRuntime
from team_state import TeamError
from team_tools import TeamTool, member_instructions
from tool_runtime import ExecutionContext, ToolResult
from tools import ToolPolicy
from usage import CacheStats

if TYPE_CHECKING:
    from team_runtime import Activation, ActivationResult, TeamRuntime

# Tools that are registered on demand rather than inherited from the main registry.
ATTACHED = frozenset({"mcp", "delegate", TEAM_TOOL})


def allowed_tools(runtime: "TeamRuntime", spec: AgentSpec) -> frozenset[str]:
    declared = set(spec.tools if spec.tools is not None else MEMBER_DEFAULT_TOOLS)
    if spec.mcp is not None:
        declared.add("mcp")
    declared.add(TEAM_TOOL)
    declared -= MEMBER_EXCLUDED
    base = runtime.base_tools
    available = set(base.registry) | {TEAM_TOOL}
    manager = runtime.mcp_manager
    if manager is None or not manager.server_count:
        available.discard("mcp")
    else:
        available.add("mcp")
    return frozenset(name for name in declared if name in available and base.policy.permits(name))


def mcp_scope(runtime: "TeamRuntime", spec: AgentSpec) -> McpScope:
    environment = runtime.environment.mcp_scope or McpScope()
    return environment.intersect(spec.mcp)


class BudgetedClient:
    """Charges every model request (including summaries) to the member's token budget."""

    def __init__(self, inner, runtime: "TeamRuntime", agent: str):
        self._inner = inner
        self._runtime = runtime
        self._agent = agent

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def complete(self, messages, tools=None, on_content=None, timeout=180, **kwargs):
        try:
            self._runtime.charge_tokens(self._agent, 0, before=True)
        except TeamError as exc:
            raise CompletionError("budget_exhausted", str(exc)) from exc
        response = self._inner.complete(messages, tools, on_content, timeout, **kwargs)
        usage = (response.get(LOCAL_METADATA) or {}).get("usage") or {}
        tokens = usage.get("total_tokens")
        if not isinstance(tokens, int):
            tokens = (
                estimate_tokens(messages)
                + estimate_tokens(tools or [])
                + estimate_tokens(
                    {key: value for key, value in response.items() if key != LOCAL_METADATA}
                )
            )
        self._runtime.charge_tokens(self._agent, tokens)
        return response


def repair_incomplete(messages: list[dict]) -> int:
    """Answer tool calls whose results were never recorded (crash or cancellation)."""
    answered = {
        message.get("tool_call_id") for message in messages if message.get("role") == "tool"
    }
    repaired = 0
    for index in range(len(messages) - 1, -1, -1):
        message = messages[index]
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            if call.get("id") not in answered:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id", ""),
                        "content": json.dumps(
                            ToolResult.failure(
                                "execution_unknown",
                                "The activation stopped before this result was recorded. "
                                "The operation may have taken effect; check actual state first.",
                                "unknown",
                            ).as_dict()
                        ),
                    }
                )
                repaired += 1
        break
    return repaired


class MemberAgent:
    def __init__(self, spec: AgentSpec, runtime: "TeamRuntime"):
        from team_runtime import ActivationResult

        self.Result = ActivationResult
        self.spec = spec
        self.runtime = runtime
        settings = runtime.settings
        if spec.reasoning_effort:
            settings = replace(settings, reasoning_effort=spec.reasoning_effort)
        self.settings = settings
        stats = getattr(runtime.client, "cache_stats", None)
        self.client = BudgetedClient(
            fork_client(runtime.client, settings, stats), runtime, spec.name
        )
        self.team_tool = TeamTool(runtime, spec.name)
        self.tools = self._tools()
        skills = runtime.skills
        granted = set(runtime.skill_names(spec))
        section = skills_prompt_section(
            [skill for skill in (skills.skills if skills else []) if skill.name in granted],
            "Skills granted to you. Load one with team action=load_skill name=<skill> "
            "before acting on it.",
        )
        prompt = member_instructions(runtime, spec, section)
        if self.tools.definitions:
            prompt += "\n\n" + self.tools.environment_instructions()
        self.system = {"role": "system", "content": prompt}

    def _tools(self):
        runtime, spec = self.runtime, self.spec
        tools = runtime.base_tools.fork()
        tools.on_tool = None
        tools.on_task_update = None
        tools.on_execution = None
        names = allowed_tools(runtime, spec)
        tools.registry = {
            name: item
            for name, item in tools.registry.items()
            if name in names and name not in ATTACHED
        }
        tools.policy = ToolPolicy(names, runtime.base_tools.policy.deny)
        if "mcp" in names:
            tools.attach_mcp(ScopedMcp(runtime.mcp_manager, mcp_scope(runtime, spec)))
        if "delegate" in names:
            tools.worker_allow = frozenset()
            tools.attach_reasoning(self.client, max_workers=runtime.limits.max_concurrent_agents)
        tools.registry[TEAM_TOOL] = self.team_tool.spec()
        tools.protected_paths = runtime.protected_paths
        if spec.write_paths is not None:
            workspace = runtime.settings.workspace.resolve()
            tools.write_roots = tuple((workspace / item).resolve() for item in spec.write_paths)
        return tools

    def run(self, activation: "Activation") -> "ActivationResult":
        runtime, name = self.runtime, self.spec.name
        trace = TraceContext(session_id=runtime.session_id, task_id=activation.id, agent_id=name)
        stats = getattr(self.client, "cache_stats", None)
        scope = (
            stats.request_scope(f"{runtime.environment.environment_id}/{name}")
            if isinstance(stats, CacheStats)
            else nullcontext()
        )
        self.team_tool.activation = activation
        messages = runtime.context_messages(name)
        repair_incomplete(messages)
        messages.append({"role": "user", "content": runtime.render_activation(activation)})
        runtime.save_context(name, messages)
        execution = ExecutionContext(activation.deadline, activation.cancelled, trace)
        try:
            with scope, trace_scope(trace):
                return self._loop(activation, messages, execution)
        except (TimeoutError, CompletionError) as exc:
            if activation.cancelled.is_set():
                return self.Result("cancelled")
            code = getattr(exc, "code", "deadline_exceeded")
            status = {"budget_exhausted": "budget", "task_deadline": "timeout"}.get(code, "error")
            if isinstance(exc, TimeoutError):
                status = "timeout"
            return self.Result(status, error=str(exc) or code)
        except Exception as exc:
            if activation.cancelled.is_set():
                return self.Result("cancelled")
            return self.Result("error", error=f"{type(exc).__name__}: {exc}")
        finally:
            self.team_tool.activation = None
            # Best effort: every accepted step was already saved as it happened.
            with suppress(Exception):
                runtime.save_context(name, messages)

    def _loop(self, activation, messages, execution) -> "ActivationResult":
        runtime, name = self.runtime, self.spec.name
        task_runtime = TaskRuntime(
            self.client,
            self.settings,
            activation.deadline,
            check_active=execution.remaining,
            cancelled=activation.cancelled,
        )

        def model_progress(event):
            phase = event.get("phase")
            if phase in {None, "response_delta", "response_reset", "context"}:
                return
            runtime.progress(name, activity="model", model_phase=phase)

        while True:
            execution.remaining()
            runtime.progress(name, activity="model", model_phase="thinking")
            response = task_runtime.generate(
                messages, self.system, self.tools.definitions, on_progress=model_progress
            )
            execution.remaining()
            messages.append(response)
            runtime.save_context(name, messages)
            calls = response.get("tool_calls") or []
            if not calls:
                return self.Result("ok", response.get("content") or "")
            for call in calls:
                execution.remaining()
                function = call.get("function", {})
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except ValueError:
                    arguments = {}
                try:
                    runtime.charge_tool_call(name)
                except TeamError as exc:
                    output = json.dumps(ToolResult.failure(exc.code, str(exc)).as_dict())
                else:
                    runtime.progress(
                        name,
                        activity="tool",
                        tool=function.get("name"),
                        arguments=arguments if isinstance(arguments, dict) else {},
                    )
                    context = replace(
                        execution.child_until(task_runtime.tool_deadline()),
                        trace=task_runtime.last_request_trace.tool(call["id"]),
                    )
                    output = self.tools.execute_in(call, context)
                messages.append(
                    {"role": "tool", "tool_call_id": call.get("id", ""), "content": output}
                )
                runtime.save_context(name, messages)
                decoded = json.loads(output)
                with trace_scope(execution.trace):
                    task_runtime.progress.observe(call, decoded)
                runtime.progress(
                    name,
                    activity="tool_result",
                    tool=function.get("name"),
                    arguments=arguments if isinstance(arguments, dict) else {},
                    tool_status=decoded.get("status"),
                )
