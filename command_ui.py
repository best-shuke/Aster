"""Terminal presentation and interactive selection for parsed commands."""

from collections.abc import Callable, Sequence
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import cast

from auth import AuthStore
from commands import CommandActions, CommandExecution, CommandName, EffectKind, ParsedCommand
from config import REASONING_EFFORTS, LoginRequired, auth_file
from conversation import Conversation
from environment_commands import enter_multiagents, environment_subcommand
from environment_definitions import SUBCOMMANDS
from execution_history import execution_entries
from goal_runtime import GoalRuntime, prepare_goal
from goals import parse_goal
from login_ui import describe, logout, model_client, run_login
from ui import Console


def use_goal(conversation: Conversation, console: Console, arguments: list[str], *, ask_task) -> bool:
    command = parse_goal(arguments)
    if command.action == "status":
        if conversation.goal is None:
            console.info("当前会话没有目标；用 /goal 目标描述 创建。")
        else:
            for line in conversation.goal.status_lines():
                console.info(line)
        return True
    goal = prepare_goal(conversation, command)
    if command.action in {"start", "resume"}:
        interrupt = console.interrupt_on_escape() if console.interactive is True else nullcontext()
        with interrupt:
            GoalRuntime(
                conversation,
                ask_turn=lambda prompt, **options: ask_task(conversation, console, prompt, **options),
                on_progress=console.progress,
                on_round=lambda state: console.info(
                    f"目标第 {state.rounds}/{state.max_rounds} 轮：{state.objective}"
                ),
                on_review=lambda state: console.info("正在核对目标、验收条件与执行证据…"),
            ).run()
        console.update_context(conversation.context_status())
    for line in goal.status_lines():
        console.info(line)
    return command.action in {"pause", "cancel"} or goal.status == "completed"


def conversation_count(conversation: Conversation) -> int:
    return sum(message.get("role") in {"user", "assistant"} for message in conversation.messages)


def show_trace(conversation: Conversation, console: Console, arguments: list[str]) -> None:
    console.execution_provider = lambda: execution_entries(
        conversation.execution_snapshot(), conversation.messages
    )
    query = " ".join(arguments)
    console.trace(query="" if query in {"all", "errors"} else query, errors=query == "errors")


def show_tool(conversation: Conversation, console: Console, arguments: list[str]) -> None:
    if len(arguments) != 1:
        console.error(ValueError("用法：/tool 编号，或 /tool last；/trace 浏览全部调用"))
        return
    entries = execution_entries(conversation.execution_snapshot(), conversation.messages)
    if not entries:
        console.info("当前会话暂无工具执行记录。")
        return
    try:
        number = len(entries) if arguments[0] == "last" else int(arguments[0].lstrip("#"))
        if not 1 <= number <= len(entries):
            raise ValueError
    except ValueError:
        console.error(ValueError(f"调用编号应为 1 到 {len(entries)}，或使用 /tool last"))
        return
    console.execution_provider = lambda: execution_entries(
        conversation.execution_snapshot(), conversation.messages
    )
    console.trace(selected=number - 1)


def show_history(conversation: Conversation, console: Console, arguments: list[str]) -> None:
    if len(arguments) > 1:
        console.error(ValueError("用法：/history [数量|all]"))
        return
    if not arguments:
        limit = 10
        if getattr(console, "interactive", False) is True and conversation_count(conversation):
            limits = [10, 20, 50, None]
            selected = console.select_option(
                "选择要显示的对话范围，不会修改或清空当前会话。",
                [
                    {
                        "label": "最近 10 条（默认）",
                        "description": "保持输出紧凑，查看最近的对话。",
                    },
                    {"label": "最近 20 条", "description": "查看更多近期对话。"},
                    {"label": "最近 50 条", "description": "回顾较长的近期对话。"},
                    {
                        "label": "全部对话",
                        "description": "显示完整记录；较长会话可能产生很多输出。",
                    },
                    {"label": "自定义数量", "description": "自行输入 1 到 100 之间的条数。"},
                ],
                title="选择历史范围",
            )
            if selected is None:
                console.info("已取消查看历史。")
                return
            if selected < len(limits):
                limit = limits[selected]
            else:
                while True:
                    value = console.read_value("  对话条数 [1-100] › ")
                    if value is None:
                        console.info("已取消查看历史。")
                        return
                    try:
                        limit = int(value.strip())
                        if not 1 <= limit <= 100:
                            raise ValueError
                    except ValueError:
                        console.warning("数量应为 1 到 100。")
                        continue
                    break
    elif arguments[0].lower() == "all":
        limit = None
    else:
        try:
            limit = int(arguments[0])
            if not 1 <= limit <= 100:
                raise ValueError
        except ValueError:
            console.error(ValueError("数量应为 1 到 100，或使用 /history all"))
            return
    console.history(conversation.messages, limit)


def show_skills(conversation: Conversation, console: Console, arguments: list[str]) -> None:
    if len(arguments) > 1:
        console.error(ValueError("用法：/skills [名称|all]"))
        return
    if conversation.skills is None or not conversation.skills.enabled:
        console.info("Skills 已关闭")
        return
    conversation.refresh_skills()
    show_skill_diagnostics(conversation, console)
    skills = conversation.skills.skills
    if not skills:
        console.info("没有发现可用的 skill")
        return
    selected_skill = None
    if arguments and arguments[0].lower() != "all":
        selected_skill = next((skill for skill in skills if skill.name == arguments[0]), None)
        if selected_skill is None:
            console.error(ValueError(f"没有找到 skill：{arguments[0]}；用 /skills 查看可用技能。"))
            return
    elif not arguments and getattr(console, "interactive", False) is True:
        selected = console.select_option(
            "选择要查看的技能；这里只查看说明，不会自动执行任务。",
            [
                {"label": "查看全部技能", "description": f"列出当前发现的 {len(skills)} 个技能。"},
                *[{"label": skill.name, "description": skill.description} for skill in skills],
            ],
            title="选择技能",
        )
        if selected is None:
            console.info("已取消查看技能。")
            return
        if selected:
            selected_skill = skills[selected - 1]
    if selected_skill is not None:
        console.info(f"{selected_skill.name}: {selected_skill.description}")
        console.info(f"文件：{selected_skill.file_path}")
        console.info(f"用法：/skill:{selected_skill.name} <任务描述>")
        if selected_skill.disable_model_invocation:
            console.info("此技能仅在你显式调用时加载，不会由模型自动选择。")
        return
    for skill in skills:
        console.info(f"{skill.name}: {skill.description}")


def resume_conversation(conversation: Conversation, console: Console, arguments: list[str]) -> None:
    messages_only = "--messages-only" in arguments
    if arguments.count("--messages-only") > 1:
        console.error(ValueError("用法：/resume [编号|会话ID|latest] [--messages-only]"))
        return
    arguments = [argument for argument in arguments if argument != "--messages-only"]
    if len(arguments) > 1:
        console.error(ValueError("用法：/resume [编号|会话ID|latest] [--messages-only]"))
        return
    try:
        sessions = conversation.sessions.list()
    except OSError as exc:
        console.error(exc)
        return
    for diagnostic in conversation.sessions.diagnostics:
        console.warning(diagnostic)
    if not sessions:
        console.info("暂无可恢复的历史会话；对话会自动保存。")
        return
    options = [
        {
            "label": f"{index}. {session.title}"
            + ("（当前会话）" if session.session_id == conversation.session_id else ""),
            "description": f"{session.updated_at:%Y-%m-%d %H:%M} · {session.message_count} 条对话"
            f" · {session.session_id[:8]}",
        }
        for index, session in enumerate(sessions, 1)
    ]
    if not arguments:
        if getattr(console, "interactive", False) is not True:
            for option in options:
                console.info(f"{option['label']} · {option['description']}")
            console.info("用 /resume 编号 恢复会话，或 /resume latest 恢复最近的会话。")
            return
        selected = console.select_option(
            "选择要继续的历史会话；当前会话会先保存，不会丢失。",
            options,
            title="恢复会话",
        )
        if selected is None:
            console.info("已取消恢复会话。")
            return
        session = sessions[selected]
    else:
        selector = arguments[0].lower()
        if selector == "latest":
            session = sessions[0]
        else:
            if selector.isdecimal() and len(selector) < 32:
                index = int(selector) - 1
                session = sessions[index] if 0 <= index < len(sessions) else None
            else:
                matches = [
                    session for session in sessions if session.session_id.startswith(selector)
                ]
                session = matches[0] if len(matches) == 1 else None
            if session is None:
                console.error(ValueError("没有找到唯一匹配的会话；用 /resume 查看可恢复的记录。"))
                return
    try:
        CommandActions(conversation).resume(session.session_id, messages_only=messages_only)
    except (OSError, RuntimeError, ValueError) as exc:
        console.error(exc)
        return
    sync_environment_ui(conversation, console)
    console.reset_session(
        conversation.settings.session_file,
        context=conversation.context_status(),
        history_count=conversation_count(conversation),
    )
    console.success(f"已恢复会话：{session.title}；可以直接继续提问。")
    console.history(conversation.messages, 10)
    if conversation.recovery_notice:
        console.info(conversation.recovery_notice)
    if conversation.tools is not None:
        console.restore_task(
            conversation.tools.task_title,
            conversation.tools.task_plan,
            completed=(conversation.last_outcome or {}).get("status") == "completed",
            status=(conversation.last_outcome or {}).get("status"),
        )


def show_new_session(conversation: Conversation, console: Console) -> None:
    if conversation.template_id is None and not conversation.messages:
        console.reset_session(
            conversation.settings.session_file, context=conversation.context_status()
        )
        return
    console.reset_session(
        conversation.settings.session_file,
        context=conversation.context_status(),
        history_count=conversation_count(conversation),
        session_label="模板新会话" if conversation.template_id is not None else "环境新会话",
    )
    console.history(conversation.messages, 10)
    if conversation.recovery_notice:
        console.info(conversation.recovery_notice)


def show_blank_session(conversation: Conversation, console: Console) -> None:
    CommandActions(conversation).reset(blank_template=True)
    show_new_session(conversation, console)
    console.success("已保存当前对话并开启空白（无模板）新会话；旧对话可用 /resume 恢复。")


def use_template(conversation: Conversation, console: Console, arguments: list[str]) -> None:
    usage = (
        "用法：/template [编号|名称]、/template add 文件 [名称]、/template save 名称、/template off"
    )
    try:
        if arguments and arguments[0] in {"off", "0"} and len(arguments) == 1:
            show_blank_session(conversation, console)
            return
        if arguments and arguments[0] in {"add", "save"}:
            template = CommandActions(conversation).save_template(arguments)
            console.success(f"已保存模板：{template.name}；用 /template 选择后开启独立新会话。")
            console.info(
                f"模板文件：{template.path}；修改其中的 messages 后，新会话会使用更新内容。"
            )
            return
        if len(arguments) > 1:
            raise ValueError(usage)
        templates = conversation.templates.list()
        for diagnostic in conversation.templates.diagnostics:
            console.warning(diagnostic)
        selected_id = conversation.templates.selected_id
        options = [
            {
                "label": "0. 空白（无模板）" + ("（当前）" if selected_id is None else ""),
                "description": "保存当前对话，开启空白新会话；不删除模板或历史记录。",
            },
            *[
                {
                    "label": f"{index}. {template.name}"
                    + (
                        "（本次默认）"
                        if template.template_id == selected_id and selected_id
                        else ""
                    ),
                    "description": f"{template.message_count} 条对话 · "
                    + (
                        f"模板文件 {template.path.name}"
                        if template.template_id
                        else f"转换文件 {template.path.name}"
                    ),
                }
                for index, template in enumerate(templates, 1)
            ],
        ]
        if not arguments:
            if getattr(console, "interactive", False) is not True:
                for option in options:
                    console.info(f"{option['label']} · {option['description']}")
                console.info("用 /template 编号或名称 开启新会话；用 /template off 开启空白会话。")
                return
            selected = console.select_option(
                "选择空白或模板开启新会话；模板只在本次运行有效，退出重进不自动沿用。",
                options,
                title="选择对话模板",
                default_index=next(
                    (
                        index
                        for index, template in enumerate(templates, 1)
                        if template.template_id == selected_id and selected_id
                    ),
                    0,
                ),
            )
            if selected is None:
                console.info("已取消选择模板，当前会话不变。")
                return
            if selected == 0:
                show_blank_session(conversation, console)
                return
            template = templates[selected - 1]
        else:
            selector = arguments[0]
            if selector.isdecimal() and len(selector) < 32:
                index = int(selector) - 1
                template = templates[index] if 0 <= index < len(templates) else None
            else:
                matches = [
                    template
                    for template in templates
                    if template.name == selector
                    or (template.template_id and template.template_id.startswith(selector.lower()))
                ]
                template = matches[0] if len(matches) == 1 else None
            if template is None:
                raise ValueError("没有找到唯一匹配的模板；用 /template 查看可用模板。")
        template = CommandActions(conversation).open_template(template)
    except (OSError, RuntimeError, ValueError) as exc:
        console.error(exc)
        return
    show_new_session(conversation, console)
    console.success(
        f"已从模板「{template.name}」开启新会话；仅本次运行的 /new 沿用它，重启默认无模板。"
    )


def create_workflow(conversation: Conversation, console: Console, arguments: list[str]) -> None:
    if getattr(console, "interactive", False) is not True:
        console.info("新建 workflow 需要交互终端；也可用 /workflow add 文件 [名称] 导入步骤。")
        return
    name = arguments[0] if arguments else console.read_value("  Workflow 名称 › ")
    if name is None:
        console.info("已取消新建 workflow，未保存任何文件。")
        return
    name = conversation.workflows.validate_name(name)
    console.info("输入任务步骤，支持粘贴多行内容；回车保存，Esc 取消。保存不会运行任务。")
    instructions = console.read_value("  Workflow 步骤 › ")
    if instructions is None:
        console.info("已取消新建 workflow，未保存任何文件。")
        return
    workflow = CommandActions(conversation).create_workflow(name, instructions)
    console.success(f"已新建 workflow「{workflow.name}」；用 /workflow 选择并运行。")
    console.info(f"文件：{workflow.path}；可直接编辑 Markdown，下一次运行读取最新内容。")


def use_workflow(
    conversation: Conversation,
    console: Console,
    arguments: list[str],
    *,
    ask_task: Callable[[Conversation, Console, str], str],
) -> None:
    usage = (
        "用法：/workflow [编号|名称] [本次任务补充]、/workflow new [名称]、"
        "/workflow add 文件 [名称]、/workflow show 编号|名称"
    )
    try:
        if arguments and arguments[0] == "new":
            if len(arguments) > 2:
                raise ValueError(usage)
            create_workflow(conversation, console, arguments[1:])
            return
        if arguments and arguments[0] == "add":
            if len(arguments) not in {2, 3}:
                raise ValueError(usage)
            source = conversation.settings.workspace / Path(arguments[1]).expanduser()
            workflow = CommandActions(conversation).import_workflow(
                source, arguments[2] if len(arguments) == 3 else None
            )
            console.success(f"已导入 workflow「{workflow.name}」；文件：{workflow.path}")
            return
        show_only = bool(arguments and arguments[0] == "show")
        if show_only:
            if len(arguments) != 2:
                raise ValueError(usage)
            arguments = arguments[1:]
        workflows = conversation.workflows.list()
        for diagnostic in conversation.workflows.diagnostics:
            console.warning(diagnostic)
        options = [
            {"label": f"{index}. {workflow.name}", "description": workflow.description}
            for index, workflow in enumerate(workflows, 1)
        ]
        context = " ".join(arguments[1:])
        if not arguments:
            if getattr(console, "interactive", False) is not True:
                if not workflows:
                    console.info("没有保存的 workflow。")
                for option in options:
                    console.info(f"{option['label']} · {option['description']}")
                console.info(usage)
                return
            selected = console.select_option(
                "选择 workflow 运行，或新建可复用的任务步骤；不会清空当前会话。",
                [
                    *options,
                    {
                        "label": "＋ 新建 workflow",
                        "description": "填写名称和步骤后保存，不会自动运行任务。",
                    },
                ],
                title="选择 workflow",
            )
            if selected is None:
                console.info("已取消选择 workflow，当前会话不变。")
                return
            if selected == len(workflows):
                create_workflow(conversation, console, [])
                return
            workflow = workflows[selected]
            console.info(f"准备运行 workflow「{workflow.name}」；回车确认，Esc 取消。")
            context = console.read_value("  本次任务补充（可留空） › ")
            if context is None:
                console.info("已取消运行 workflow，当前会话不变。")
                return
        else:
            selector = arguments[0]
            matches = [
                workflow
                for workflow in workflows
                if workflow.name.casefold() == selector.casefold()
            ]
            if matches:
                workflow = matches[0] if len(matches) == 1 else None
            elif selector.isdecimal() and len(selector) < 32:
                index = int(selector) - 1
                workflow = workflows[index] if 0 <= index < len(workflows) else None
            else:
                workflow = None
            if workflow is None:
                raise ValueError("没有找到唯一匹配的 workflow；用 /workflow 查看或新建。")
        workflow = conversation.workflows.read(workflow.path)
        if show_only:
            console.info(f"Workflow「{workflow.name}」 · {workflow.path}")
            console.info(workflow.instructions)
            return
    except (OSError, RuntimeError, ValueError) as exc:
        console.error(exc)
        return
    console.info(f"正在运行 workflow「{workflow.name}」，沿用当前会话和推理档位。")
    ask_task(conversation, console, workflow.prompt(context))


def update_effort(conversation: Conversation, console: Console, arguments: list[str]) -> None:
    if not arguments:
        effort = conversation.settings.reasoning_effort or "max"
        if getattr(console, "interactive", False) is True:
            descriptions = {
                "max": "最高推理强度（默认），适合复杂任务。",
                "high": "较高推理强度，平衡深度与响应速度。",
                "low": "轻量推理，优先响应速度。",
                "off": "关闭思考，直接生成回复。",
                "default": "恢复本程序默认的 max。",
            }
            efforts = list(descriptions)
            options = [
                {
                    "label": value + ("（当前）" if value == effort else ""),
                    "description": description,
                }
                for value, description in descriptions.items()
            ]
            selected = console.select_option(
                "从下一次提问生效（含子任务，仅当前进程）。",
                options,
                default_index=efforts.index(effort),
                title="选择推理强度",
            )
            if selected is None:
                console.info(f"已取消选择，推理强度保持 {effort}。")
                return
            arguments = [efforts[selected]]
        else:
            console.info(f"当前推理强度：{effort}")
            console.info("用法：/effort " + "|".join(REASONING_EFFORTS))
            console.info(
                "low 轻量 · high 较高 · max 最高（默认） · off 关闭思考 · default 恢复 max"
            )
            console.info("切换仅当前进程有效；持久配置请在 .env 设置 DEEPSEEK_REASONING_EFFORT。")
            return
    try:
        effort = CommandActions(conversation).set_effort(arguments)
        console.reasoning_effort = effort
        console.success(f"推理强度已设为 {effort}，从下一次提问生效（含子任务，仅当前进程）。")
    except ValueError as exc:
        console.error(exc)


def websearch_status(conversation: Conversation) -> str:
    if not conversation.settings.websearch_enabled:
        return "关闭"
    names = {item["function"]["name"] for item in conversation.definitions}
    return "可用" if "websearch" in names else "未配置或策略禁用"


def use_login(conversation: Conversation, console: Console, arguments: list[str]) -> None:
    store = AuthStore(conversation.settings.auth_file or auth_file())
    login = run_login(console, store)
    if login is None:
        console.info("登录方式未改变。")
        return
    try:
        conversation.switch_login(login, model_client(conversation.settings.with_login(login)))
    except (LoginRequired, RuntimeError, ValueError) as exc:
        console.warning(f"新的登录方式已保存，重启 Aster 后生效；当前会话未切换：{exc}")
        return
    settings = conversation.settings
    console.model = settings.model
    console.account = describe(settings)
    console.websearch_status = websearch_status(conversation)
    console.update_context(conversation.context_status())
    console.success(f"已切换到 {settings.model}（{console.account}），从下一次提问生效。")


def use_logout(conversation: Conversation, console: Console, arguments: list[str]) -> None:
    if logout(console, AuthStore(conversation.settings.auth_file or auth_file())):
        if conversation.settings.provider == "chatgpt":
            console.info("当前会话无法再使用 ChatGPT 账户；用 /login 重新登录或改用第三方 API。")
        else:
            console.info("当前进程仍使用已加载的 API 配置，退出后失效。")


def sync_environment_ui(conversation: Conversation, console: Console) -> None:
    settings = conversation.settings
    if getattr(conversation, "team", None) is None:
        # Leaving a team environment drops its status from the toolbar.
        console.update_team(None)
    console.environment_name = conversation.environment_id or "默认"
    console.last_environment_name = (
        conversation.last_environment_id if conversation.environment_id is None else None
    )
    console.reasoning_effort = settings.reasoning_effort
    console.skill_provider = lambda: (
        {skill.name: skill.description for skill in conversation.skills.skills}
        if conversation.skills is not None
        else {}
    )
    skills = conversation.skills
    console.skill_status = (
        {
            "enabled": skills.enabled,
            "loaded": len(skills.skills),
            "automatic": sum(not skill.disable_model_invocation for skill in skills.skills),
            "manual": sum(skill.disable_model_invocation for skill in skills.skills),
            "diagnostics": len(skills.diagnostics),
        }
        if skills is not None
        else None
    )
    console.mcp_policy_enabled = (
        conversation.tools.policy.permits("mcp") if conversation.tools is not None else False
    )
    console.websearch_status = websearch_status(conversation)
    console.mcp_provider = (
        (lambda: conversation.mcp_manager.status().data["servers"])
        if conversation.mcp_manager is not None
        else None
    )
    console.environment_provider = conversation.environment_health
    if conversation.mcp_manager is not None:
        console.update_mcp(conversation.mcp_manager.status().data["servers"])
    else:
        console.update_mcp([])
        console.mcp_status = "未配置" if settings.mcp_enabled else "关闭"


def use_environment(conversation: Conversation, console: Console, arguments: list[str]) -> bool:
    usage = (
        "用法：/environment [名称 [keep] [--yes]|off|show 名称|trust 名称|deny 名称]；"
        "/env help 查看 create/use/exit/status/skills/mcp/agents 等子命令"
    )
    store = conversation.environments
    request_id = None
    requested_environment = None
    try:
        if arguments and arguments[0] in SUBCOMMANDS:
            return environment_subcommand(conversation, console, list(arguments), use_environment)
        if not arguments and getattr(console, "interactive", False) is True:
            entries = store.list()
            selectors = ["off", *(entry.environment_id for entry in entries)]
            current = conversation.environment_id or "off"
            options = [
                {
                    "label": "默认" + ("（当前）" if current == "off" else ""),
                    "description": "恢复进程启动时的能力配置，并开启新会话。",
                },
                *(
                    {
                        "label": entry.environment_id
                        + ("（当前）" if entry.environment_id == current else ""),
                        "description": (
                            "工作区环境需先显式 trust；切换会开启新会话。"
                            if entry.source == "workspace"
                            else "首次使用需确认能力包；切换会开启新会话。"
                        ),
                    }
                    for entry in entries
                ),
            ]
            for diagnostic in store.diagnostics:
                console.warning(diagnostic)
            selected = console.select_option(
                "选择能力环境；切换默认开启新会话。",
                options,
                default_index=selectors.index(current) if current in selectors else 0,
                title="选择能力环境",
            )
            if selected is None:
                console.info("已取消选择环境，当前环境与会话不变。")
                return True
            arguments = [selectors[selected]]
        if not arguments:
            console.info(f"当前环境：{conversation.environment_id or '默认'}")
            console.info(f"prompt：{conversation.settings.prompt_file}")
            roots = (
                conversation.skills.roots
                if conversation.skills and conversation.skills.enabled
                else []
            )
            console.info("skills：" + (", ".join(map(str, roots)) or "关闭"))
            console.info(
                f"effort：{conversation.settings.reasoning_effort}；"
                f"tools：{', '.join(item['function']['name'] for item in conversation.definitions) or '无'}"
            )
            console.info(
                f"MCP：{conversation.mcp_manager.config_file if conversation.mcp_manager else '未挂载'}"
            )
            console.info(
                "websearch："
                + (
                    "可用"
                    if any(
                        item["function"]["name"] == "websearch" for item in conversation.definitions
                    )
                    else "不可用（关闭、未配置或策略禁用）"
                )
            )
            if conversation.tools is not None:
                console.info(
                    "tools.deny：" + (", ".join(sorted(conversation.tools.policy.deny)) or "无")
                )
            if conversation.mcp_manager is not None:
                for server in conversation.mcp_manager.status().data["servers"]:
                    console.info(
                        f"  MCP {server['name']}：{server['status']} · {server.get('tool_count', 0)} 个工具"
                    )
            for entry in store.list():
                current = "（当前）" if entry.environment_id == conversation.environment_id else ""
                trust = (
                    "；工作区来源需 trust/指纹校验"
                    if entry.source == "workspace"
                    else "；使用前确认"
                )
                console.info(f"{entry.environment_id}{current}{trust} · {entry.path}")
            for diagnostic in store.diagnostics:
                console.warning(diagnostic)
            return True
        yes = "--yes" in arguments
        arguments = [item for item in arguments if item != "--yes"]
        if not arguments:
            raise ValueError(usage)
        action = arguments[0]
        if action == "save":
            raise ValueError("/environment save 暂未实现：需先定义资源复制与秘密信息导出规则。")
        if action in {"show", "trust", "deny"}:
            if len(arguments) != 2:
                raise ValueError(usage)
            if action == "deny":
                CommandActions(conversation).deny_environment(arguments[1])
                console.success("已撤销环境信任；当前若使用该环境，下一次提问将拒绝执行。")
                return True
            environment = store.resolve(arguments[1])
            console.info(store.summary(environment))
            if action == "show":
                return True
        else:
            if len(arguments) > 2 or (len(arguments) == 2 and arguments[1] != "keep"):
                raise ValueError(usage)
            requested_environment = None if action == "off" else action
            request_id = store.audit("requested", requested_environment, keep=len(arguments) == 2)
            environment = None if action == "off" else store.resolve(action)
            if environment is not None:
                console.info(store.summary(environment))
                if store.denied(environment):
                    raise ValueError("环境已 deny；必须先显式 /environment trust 名称 重新确认。")
                if environment.entry.source == "workspace" and not store.approved(environment):
                    raise ValueError("工作区环境未信任或资源改变；请先 /environment trust 名称。")
        needs_confirmation = action == "trust" or (
            environment is not None and not store.approved(environment)
        )
        if needs_confirmation:
            if not yes:
                if getattr(console, "interactive", False) is not True:
                    raise ValueError("环境需要显式确认；非交互模式请检查 show 摘要后添加 --yes。")
                selected = console.select_option(
                    "确认以上来源、资源和 MCP 命令；工具策略不是操作系统沙箱。",
                    [
                        {"label": "取消", "description": "不改变信任或当前环境。"},
                        {"label": "确认能力包", "description": "记录规范路径与当前资源指纹。"},
                    ],
                    title="确认环境能力",
                    default_index=0,
                )
                if selected != 1:
                    if request_id is not None:
                        store.audit("cancelled", requested_environment, request=request_id)
                    console.info("已取消，当前环境与会话不变。")
                    return False
            CommandActions(conversation).approve_environment(environment)
        if action == "trust":
            console.success(
                f"已确认 {environment.environment_id}；未应用，用 /environment 名称 切换。"
            )
            return True
        keep = len(arguments) == 2
        if keep:
            console.warning("keep 保留已有上下文；它可能与新环境不匹配，开场模板不会注入。")
        console.info("正在切换环境…")
        previous_session = conversation.session_id
        warnings = CommandActions(conversation).apply_environment(
            environment, keep=keep, on_connect=console.mcp_connecting
        )
        sync_environment_ui(conversation, console)
        unchanged = previous_session == conversation.session_id and not keep
        if not keep and not unchanged:
            show_new_session(conversation, console)
        for warning in warnings:
            console.warning(warning)
        if unchanged:
            console.info("环境已是当前配置；保留现有会话与工具连接。")
            return True
        report = (
            console.success if conversation.environment_health() == "ready" else console.warning
        )
        report(
            f"已切换到 {conversation.environment_id or '默认'} 环境；"
            + ("保留当前对话。" if keep else "已开启新会话。")
        )
        return True
    except (OSError, RuntimeError, ValueError) as exc:
        if request_id is not None:
            with suppress(OSError):
                store.audit(
                    "rejected",
                    requested_environment,
                    request=request_id,
                    error_type=type(exc).__name__,
                )
        console.error(exc)
        return False


def use_multiagents(conversation: Conversation, console: Console, arguments: list[str]) -> bool:
    try:
        return enter_multiagents(conversation, console, arguments, use_environment)
    except (OSError, RuntimeError, ValueError) as exc:
        console.error(exc)
        return False


def show_skill_diagnostics(conversation: Conversation, console: Console) -> None:
    if conversation.skills is None:
        return
    for diagnostic in conversation.skills.diagnostics:
        console.warning(f"Skill {diagnostic.path}: {diagnostic.message}")


def present_command(
    execution: CommandExecution,
    conversation: Conversation,
    console: Console,
    *,
    help_text: str | Sequence[tuple[str, Sequence[tuple[str, str]]]],
    interactions: dict[CommandName, Callable[[Conversation, Console, list[str]], object]],
) -> None:
    for effect in execution.effects:
        if effect.kind is EffectKind.ERROR:
            console.error(cast(Exception, effect.payload))
        elif effect.kind is EffectKind.NEW_SESSION:
            show_new_session(conversation, console)
        elif effect.kind is EffectKind.TASKS:
            console.tasks()
        elif effect.kind is EffectKind.STATUS:
            sync_environment_ui(conversation, console)
            console.update_context(conversation.context_status())
            console.status(conversation.settings.session_file, conversation_count(conversation))
        elif effect.kind is EffectKind.INFO:
            console.info(cast(str, effect.payload))
        elif effect.kind is EffectKind.CACHE:
            console.cache(cast(dict, effect.payload))
        elif effect.kind is EffectKind.HELP:
            console.help(help_text)
        elif effect.kind is EffectKind.TEXT:
            print(effect.payload)
        elif effect.kind is EffectKind.INTERACTION:
            command = cast(ParsedCommand, effect.payload)
            interactions[command.name](conversation, console, list(command.arguments))
        else:
            raise ValueError(f"Unsupported command presentation: {effect.kind}")
