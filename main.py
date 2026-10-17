import argparse
import shlex
from contextlib import nullcontext
from pathlib import Path

from command_ui import (
    conversation_count,
    present_command,
    resume_conversation,
    show_history,
    show_skill_diagnostics,
    show_skills,
    show_tool,
    show_trace,
    use_template,
)
from command_ui import (
    sync_environment_ui as sync_environment_ui,
)
from command_ui import (
    update_effort as update_effort,
)
from command_ui import (
    use_environment as use_environment,
)
from command_ui import (
    use_multiagents as use_multiagents,
)
from command_ui import (
    use_workflow as present_workflow,
)
from commands import (
    COMMAND_NAMES,
    QUOTED_ARGUMENTS,
    CommandName,
    CommandResult,
    execute_command,
    parse_command,
)
from config import APP_NAME, CLI_NAME, Settings
from conversation import Conversation
from deepseek_client import DeepSeekClient
from execution_history import execution_entries
from mcp_client import McpManager
from skills import SkillCatalog
from tools import CodingTools
from ui import Console
from usage import CacheStats
from websearch import WebSearchClient

HELP = """
  常用命令

  /status             查看上下文、TPS、MCP 连接及运行配置
  /history [数量|all] 选择范围，或直接查看指定数量 / 全部对话
  /trace [筛选词|errors] 浏览执行轨迹、命令参数和工具结果（输入框 Ctrl-T）
  /tool 编号|last      直接打开某次调用；详情中 o 打开保留文件
  /resume [编号|ID|latest] [--messages-only] 恢复会话，或仅复制历史到当前环境
  /template [编号|名称] 选择空白或模板，模板仅作为本次 /new 的起点
  /template add 文件 [名称] 导入转换后的对话为模板
  /template save 名称 保存当前对话为模板
  /template off       保存当前会话并开启空白（无模板）新会话
  /environment, /env  选择能力环境；非交互模式列出环境与当前配置
  /environment 名称 [keep] [--yes] 切换环境，默认开启新会话
  /environment off    恢复进程启动默认能力并开启新会话
  /environment show 名称 查看解析摘要，不应用
  /environment trust|deny 名称 确认或撤销指定环境的资源包信任
  /env create 名称 [--template multi-agent] 创建环境；/env use 名称 进入；/env exit 退出
  /env status|log     查看当前环境、团队成员、任务与事件
  /env skills|mcp|agents|limits 管理当前环境的技能、MCP、成员与资源限制（/env help）
  /multiagents        进入默认多代理团队环境（不存在时自动创建）
  /workflow [编号|名称] 选择并运行 workflow，或在菜单中新建
  /workflow new [名称] 新建可复用的任务步骤
  /workflow add 文件 [名称] 导入 Markdown workflow
  /workflow show 编号|名称 查看步骤，不执行任务
  /effort [档位]      选择或直接修改推理强度
  /cache              查看缓存命中率
  /skills [名称|all]  选择技能查看说明，或直接列出全部
  /skill:名称 [任务]  加载技能完整说明并执行任务
  /tasks, /todo       查看任务标题、Todo 和子代理状态

  /new, /clear        全新会话（清屏并重置所有对话状态）
  /session            查看会话文件
  /prompt             查看系统提示词
  /exit, /quit        退出
  /help               查看帮助

  推理档位：default / off / low / high / max
  输入 / 自动提示命令；Tab 补全，↑ / ↓ 选择，Esc 关闭提示。
"""


def use_workflow(conversation: Conversation, console: Console, arguments: list[str]) -> None:
    present_workflow(conversation, console, arguments, ask_task=ask)


def handle_command(conversation: Conversation, console: Console, prompt: str) -> CommandResult:
    execution = execute_command(parse_command(prompt), conversation)
    present_command(
        execution,
        conversation,
        console,
        help_text=HELP,
        interactions={
            CommandName.HISTORY: show_history,
            CommandName.TRACE: show_trace,
            CommandName.TOOL: show_tool,
            CommandName.RESUME: resume_conversation,
            CommandName.TEMPLATE: use_template,
            CommandName.ENVIRONMENT: use_environment,
            CommandName.MULTIAGENTS: use_multiagents,
            CommandName.WORKFLOW: use_workflow,
            CommandName.SKILLS: show_skills,
            CommandName.EFFORT: update_effort,
        },
    )
    return execution.result


def run_interactive(conversation: Conversation, console: Console) -> None:
    # Another message may follow, so team members can keep working between turns.
    conversation.team_background = True
    console.execution_provider = lambda: execution_entries(
        conversation.execution_snapshot(), conversation.messages
    )
    sync_environment_ui(conversation, console)
    console.update_context(conversation.context_status())
    console.banner(conversation.settings.session_file, conversation_count(conversation))
    console.configuration_notice()
    if conversation.last_environment_id and conversation.environment_id is None:
        console.info(
            f"上次环境：{conversation.last_environment_id}；本次为默认能力，"
            f"/environment {conversation.last_environment_id} 可显式恢复。"
        )
    show_skill_diagnostics(conversation, console)
    if conversation.recovery_notice:
        console.info(conversation.recovery_notice)
    if conversation.tools is not None:
        console.restore_task(
            conversation.tools.task_title,
            conversation.tools.task_plan,
            completed=(conversation.last_outcome or {}).get("status") == "completed",
            status=(conversation.last_outcome or {}).get("status"),
        )

    while True:
        try:
            sync_environment_ui(conversation, console)
            if conversation.skills is not None:
                conversation.skills.reload()
            console.update_context(conversation.context_status())
        except KeyboardInterrupt:
            print()
            continue
        except Exception as exc:
            console.error(exc)
        try:
            prompt = console.prompt()
        except EOFError:
            print()
            return
        except KeyboardInterrupt:
            print()
            continue

        if not prompt.strip():
            continue
        try:
            command = handle_command(conversation, console, prompt)
            if command is CommandResult.EXIT:
                return
            if command is CommandResult.HANDLED:
                continue
            ask(conversation, console, prompt)
        except KeyboardInterrupt:
            console.info("当前任务已暂停，已保存可恢复的进展。")
        except Exception as exc:
            console.error(exc)


def ask(conversation: Conversation, console: Console, prompt: str) -> str:
    console.execution_provider = lambda: execution_entries(
        conversation.execution_snapshot(), conversation.messages
    )
    console.update_context(conversation.context_status())
    console.task_start(prompt)
    completed = False
    failed = False
    status = "incomplete"
    try:
        interrupt = (
            console.interrupt_on_escape()
            if getattr(console, "interactive", False) is True
            else nullcontext()
        )
        with interrupt:
            answer = conversation.ask(
                prompt,
                on_content=console.stream,
                on_response_start=console.assistant_start,
                on_response_end=console.assistant_end,
                on_progress=console.progress,
                on_reasoning=console.reasoning,
                on_tool_result=console.tool_result,
            )
        status = (conversation.last_outcome or {}).get("status", "incomplete")
        completed = status == "completed"
        return answer
    except KeyboardInterrupt:
        status = "paused"
        raise
    except Exception:
        failed = True
        status = "error"
        raise
    finally:
        try:
            console.update_context(conversation.context_status())
        except Exception as exc:
            console.error(exc)
        console.task_end(completed, failed=failed, status=status)
        stats = getattr(conversation.client, "cache_stats", None)
        if isinstance(stats, CacheStats):
            console.cache(stats.snapshot(), compact=True)


def main() -> int:
    parser = argparse.ArgumentParser(prog=CLI_NAME, description=f"{APP_NAME} — 终端任务助手")
    parser.add_argument("--environment", metavar="NAME", help="显式选择能力环境")
    parser.add_argument("--yes", action="store_true", help="确认指定环境当前能力包")
    parser.add_argument(
        "--skill", action="append", default=[], metavar="PATH", help="加载技能文件或目录，可重复"
    )
    parser.add_argument(
        "--no-skills", action="store_true", help="关闭默认技能发现；仍加载 --skill 指定的路径"
    )
    parser.add_argument("prompt", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    prompt = " ".join(arguments.prompt).strip()
    if len(arguments.prompt) > 1 and COMMAND_NAMES.get(arguments.prompt[0]) in QUOTED_ARGUMENTS:
        prompt = shlex.join(arguments.prompt)
    command = parse_command(prompt)
    environment_command = command is not None and command.name is CommandName.ENVIRONMENT
    settings = Settings.from_env()
    skills = SkillCatalog(
        settings.workspace,
        settings.skills_dir,
        enabled=settings.skills_enabled or bool(arguments.skill),
        extra_dirs=tuple(Path(path) for path in arguments.skill),
        include_defaults=settings.skills_enabled and not arguments.no_skills,
    )
    if not settings.websearch_enabled:
        websearch_status = "关闭"
    elif settings.resolved_websearch_api_key:
        websearch_status = "可用"
    else:
        websearch_status = "未配置"
    mcp_manager = None
    mcp_error = None
    if settings.mcp_enabled:
        try:
            mcp_manager = McpManager(
                settings.mcp_config_file, settings.workspace, skill_registry=skills.mcp_registry()
            )
            mcp_status = (
                f"{mcp_manager.server_count} 个服务" if mcp_manager.server_count else "未配置"
            )
        except ValueError as exc:
            mcp_status = "配置错误"
            mcp_error = str(exc)
    else:
        mcp_status = "关闭"

    console = Console(
        settings.model,
        workspace=settings.workspace,
        reasoning_effort=settings.reasoning_effort,
        websearch_status=websearch_status,
        mcp_status=mcp_status,
    )
    tools = CodingTools(
        settings.workspace, on_tool=console.tool, on_task_update=console.task_update
    )
    if console.interactive is True:
        tools.attach_user_input(console.ask_user)
    if settings.websearch_enabled and settings.resolved_websearch_api_key:
        tools.attach_websearch(WebSearchClient(settings))
    if mcp_manager is not None and mcp_manager.server_count:
        tools.attach_mcp(mcp_manager)
    conversation = Conversation(
        settings,
        DeepSeekClient(settings),
        tools,
        skills=skills,
        mcp_manager=mcp_manager,
    )

    screen = (
        console.session_screen() if console.interactive is True and not prompt else nullcontext()
    )
    with screen:
        try:
            if mcp_error:
                console.error(ValueError(f"MCP: {mcp_error}"))
            if arguments.environment:
                options = [arguments.environment, *(["--yes"] if arguments.yes else [])]
                if not use_environment(conversation, console, options):
                    return 2
            elif mcp_manager is not None and mcp_manager.server_count and not environment_command:
                console.mcp_provider = lambda: conversation.mcp_manager.status().data["servers"]
                startup = mcp_manager.connect_all(on_connect=console.mcp_connecting)
                console.update_mcp(startup.data["servers"])
                for failure in startup.data["failures"]:
                    console.warning(
                        f"MCP {failure['server']} 连接失败：{failure['error']}；使用时会自动重试。"
                    )
            if prompt:
                if environment_command:
                    if command.error is not None:
                        console.error(command.error)
                        return 2
                    return (
                        0 if use_environment(conversation, console, list(command.arguments)) else 2
                    )
                if handle_command(conversation, console, prompt) is not CommandResult.NOT_COMMAND:
                    return 0
                console.configuration_notice()
                if conversation.last_environment_id and conversation.environment_id is None:
                    console.warning(
                        f"上次环境 {conversation.last_environment_id} 未自动应用；本次使用默认能力。"
                    )
                try:
                    ask(conversation, console, prompt)
                except KeyboardInterrupt:
                    console.info("当前任务已暂停，已保存可恢复的进展。")
                    return 130
                except ValueError as exc:
                    console.error(exc)
                    return 2
                except Exception as exc:
                    console.error(exc)
                    return 1
                return 0 if (conversation.last_outcome or {}).get("status") == "completed" else 1
            run_interactive(conversation, console)
            return 0
        finally:
            conversation.close()


if __name__ == "__main__":
    raise SystemExit(main())
