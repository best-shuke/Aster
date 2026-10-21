import argparse
import os
import shlex
import sys
from contextlib import nullcontext
from pathlib import Path

from auth import AuthError, AuthStore
from chatgpt_client import ChatGPTClient
from command_ui import (
    conversation_count,
    present_command,
    resume_conversation,
    show_history,
    show_skill_diagnostics,
    show_skills,
    show_tool,
    show_trace,
    use_login,
    use_logout,
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
from command_ui import use_goal as present_goal
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
from config import APP_NAME, CLI_NAME, LoginRequired, Settings, auth_file, load_env
from conversation import Conversation
from deepseek_client import DeepSeekClient
from execution_history import execution_entries
from login_ui import describe, run_login
from mcp_client import McpManager
from skills import SkillCatalog
from tools import CodingTools
from ui import Console
from usage import CacheStats
from websearch import websearch_client

HELP = (
    (
        "对话与会话",
        (
            ("/new, /clear", "保存当前会话并开启全新对话（清屏）"),
            ("/resume [编号|ID|latest]", "恢复历史会话；加 --messages-only 仅复制历史到当前环境"),
            ("/history [数量|all]", "查看对话记录；不带参数时选择范围"),
            ("/session", "查看会话文件位置"),
            ("/exit, /quit", "退出（也可按 Ctrl-D）"),
        ),
    ),
    (
        "持续目标",
        (
            ('/goal [start] "目标"', "创建持续目标，自动执行、核对验收并继续下一轮"),
            ("/goal status|pause|cancel", "查看、暂停或取消当前会话的目标"),
            ("/goal resume [补充说明]", "恢复目标；--max-rounds / --max-seconds 调整总预算"),
            ('--criterion "条件"', "创建目标时添加验收条件，可重复；默认 20 轮、3600 秒"),
        ),
    ),
    (
        "查看与诊断",
        (
            ("/status", "模型、能力、上下文、用量与会话总览"),
            ("/tasks, /todo", "任务标题、Todo 和子代理状态"),
            ("/trace [筛选词|errors]", "浏览执行轨迹、命令参数和工具结果（Ctrl-T）"),
            ("/tool 编号|last", "打开某次调用；详情中按 o 打开保留的完整输出"),
            ("/cache", "查看提示缓存命中率"),
            ("/prompt", "查看系统提示词"),
        ),
    ),
    (
        "环境与团队",
        (
            ("/env, /environment", "选择能力环境；切换默认开启新会话"),
            ("/env 名称 [keep] [--yes]", "切换环境；keep 保留当前上下文"),
            ("/env off", "恢复进程启动时的默认能力"),
            ("/env show|trust|deny 名称", "查看解析摘要 / 确认 / 撤销资源包信任"),
            ("/env create 名称 [--template 模板]", "创建环境，如 --template multi-agent"),
            ("/env use 名称, /env exit", "进入或退出环境"),
            ("/env status|log|results", "当前环境、团队成员、任务、事件与后台结果"),
            ("/env skills|mcp|agents|limits", "管理技能、MCP、成员与资源限制（/env help）"),
            ("/multiagents", "进入默认多代理团队环境（不存在时自动创建）"),
        ),
    ),
    (
        "模板、Workflow 与技能",
        (
            ("/template [编号|名称]", "选择空白或模板，作为本次 /new 的起点"),
            ("/template add 文件 [名称]", "导入转换后的对话为模板"),
            ("/template save 名称", "保存当前对话为模板"),
            ("/template off", "保存当前会话并开启空白（无模板）新会话"),
            ("/workflow [编号|名称]", "选择并运行 workflow，或在菜单中新建"),
            ("/workflow new [名称]", "新建可复用的任务步骤"),
            ("/workflow add 文件 [名称]", "导入 Markdown workflow"),
            ("/workflow show 编号|名称", "查看步骤，不执行任务"),
            ("/skills [名称|all]", "选择技能查看说明，或列出全部"),
            ("/skill:名称 [任务]", "加载技能完整说明并执行任务"),
        ),
    ),
    (
        "设置",
        (
            ("/login", "登录方式：ChatGPT 账户（Plus / Pro）或第三方 API，并选择模型"),
            ("/logout", "退出登录并删除保存的凭据"),
            ("/effort [档位]", "推理强度：default / off / low / high / max"),
        ),
    ),
    (
        "快捷键",
        (
            ("Enter", "发送"),
            ("\\ + Enter", "行尾输入 \\ 再回车即换行；也可以直接粘贴多行内容"),
            ("/ 与 Tab", "命令提示与补全；↑ ↓ 选择，Esc 关闭提示"),
            ("Esc, Ctrl-C", "任务运行中：暂停并保存可恢复的进展"),
            ("Ctrl-T", "打开执行轨迹"),
            ("Ctrl-D", "退出"),
        ),
    ),
)


def use_workflow(conversation: Conversation, console: Console, arguments: list[str]) -> None:
    present_workflow(conversation, console, arguments, ask_task=ask)


def use_goal(conversation: Conversation, console: Console, arguments: list[str]) -> bool:
    return present_goal(conversation, console, arguments, ask_task=ask)


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
            CommandName.LOGIN: use_login,
            CommandName.LOGOUT: use_logout,
            CommandName.GOAL: use_goal,
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
    if getattr(conversation, "goal", None) is not None:
        console.info("当前会话有已保存的目标；/goal status 查看，/goal resume 继续。")
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


def cache_totals(conversation: Conversation) -> dict | None:
    stats = getattr(getattr(conversation, "client", None), "cache_stats", None)
    return stats.totals() if isinstance(stats, CacheStats) else None


def ask(conversation: Conversation, console: Console, prompt: str, **task_options) -> str:
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
                **task_options,
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


def load_settings(*, choose: bool, one_shot: bool) -> Settings | None:
    """Settings from the saved login; an interactive first start asks how to sign in."""
    store = AuthStore(auth_file())
    try:
        login = store.login()
    except AuthError as exc:
        print(f"警告：{exc}", file=sys.stderr)
        login = None
    if sys.stdin.isatty() and sys.stdout.isatty():
        load_env()
        # A one-shot prompt with model variables already configured runs without a menu.
        configured = one_shot and bool(os.environ.get("DEEPSEEK_API_KEY"))
        if choose or (login is None and not configured):
            console = Console()
            console.toolbar = False
            login = run_login(console, store, first_run=not choose) or login
    try:
        return Settings.from_env(login)
    except LoginRequired:
        print(
            f"尚未登录：在终端运行 {CLI_NAME} 选择 ChatGPT 账户或第三方 API，"
            "或设置 DEEPSEEK_API_KEY。",
            file=sys.stderr,
        )
        return None


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
    parser.add_argument(
        "--login", action="store_true", help="启动前重新选择登录方式（ChatGPT 账户或第三方 API）"
    )
    parser.add_argument("prompt", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    prompt = " ".join(arguments.prompt).strip()
    if len(arguments.prompt) > 1 and COMMAND_NAMES.get(arguments.prompt[0]) in QUOTED_ARGUMENTS:
        prompt = shlex.join(arguments.prompt)
    command = parse_command(prompt)
    environment_command = command is not None and command.name is CommandName.ENVIRONMENT
    settings = load_settings(choose=arguments.login, one_shot=bool(prompt))
    if settings is None:
        return 2
    skills = SkillCatalog(
        settings.workspace,
        settings.skills_dir,
        enabled=settings.skills_enabled or bool(arguments.skill),
        extra_dirs=tuple(Path(path) for path in arguments.skill),
        include_defaults=settings.skills_enabled and not arguments.no_skills,
    )
    if not settings.websearch_enabled:
        websearch_status = "关闭"
    elif settings.websearch_provider:
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
    console.account = describe(settings)
    tools = CodingTools(
        settings.workspace, on_tool=console.tool, on_task_update=console.task_update
    )
    if console.interactive is True:
        tools.attach_user_input(console.ask_user)
    search = websearch_client(settings)
    if search is not None:
        tools.attach_websearch(search)
    if mcp_manager is not None and mcp_manager.server_count:
        tools.attach_mcp(mcp_manager)
    conversation = Conversation(
        settings,
        ChatGPTClient(settings) if settings.provider == "chatgpt" else DeepSeekClient(settings),
        tools,
        skills=skills,
        mcp_manager=mcp_manager,
    )
    # The status bar, /status and task summaries show prompt cache hit rates.
    console.cache_provider = lambda: cache_totals(conversation)
    # A ChatGPT account reports its plan usage windows with each response.
    console.quota_provider = lambda: getattr(
        getattr(conversation.client, "account", None), "quota", None
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
                if command is not None and command.name is CommandName.GOAL:
                    try:
                        if command.error is not None:
                            raise command.error
                        return 0 if use_goal(conversation, console, list(command.arguments)) else 1
                    except KeyboardInterrupt:
                        console.info("目标已暂停，已保存可恢复的进展。")
                        return 130
                    except ValueError as exc:
                        console.error(exc)
                        return 2
                    except Exception as exc:
                        console.error(exc)
                        return 1
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
