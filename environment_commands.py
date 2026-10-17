"""/environment subcommands for creating, configuring and inspecting environments.

Switching itself stays in command_ui.use_environment (trust checks, confirmation and the
atomic transition); these handlers edit definitions and report state.
"""

from collections.abc import Callable

from commands import CommandActions
from environment_definitions import SUBCOMMANDS, EnvironmentEditor, parse_server_options
from environment_spec import format_limit
from environments import ResolvedEnvironment

Switch = Callable[..., bool]

USAGE = """  /env list                          列出全部环境
  /env create 名称 [--template single-agent|multi-agent|coding-team] [--description 文本] [--scope user|workspace|program]
  /env use 名称 [keep] [--yes]        切换环境（同 /env 名称）
  /env exit                          退出环境，恢复默认代理（同 /env off）
  /env status                        查看当前环境、会话与团队状态
  /env reload                        重新加载当前环境定义，保留对话
  /env type single-agent|multi-agent 修改环境类型
  /env skills [list]                 查看环境技能与可添加技能
  /env skills add|remove|enable|disable 技能...
  /env mcp [list]                    查看环境 MCP 选择与可添加服务
  /env mcp add 服务 [--tools 模式,…] [--deny 模式,…]  引用已配置服务
  /env mcp add 服务 --command 命令|--url 地址 [--arg 参数] [--set-env K=${VAR}] [--header K=${VAR}]
  /env mcp remove|enable|disable 服务；/env mcp tools 服务 模式… [--deny 模式,…]
  /env agents [list|show 名称]
  /env agents add|set 名称 [--description 文本] [--instructions 文本] [--tools a,b] [--skills a,b]
                     [--mcp 服务[:模式],…] [--write-paths 目录,…] [--effort 档位]
  /env agents remove|enable|disable 名称
  /env limits [键=整数|unlimited|default …]   查看或修改资源限制
  /env log [数量]                    查看团队事件
  /env results                       查看成员在后台完成的工作和发给主代理的结果（不消耗）
  以上编辑命令默认作用于当前环境；用 --env 名称 指定其他环境。"""


def _take(arguments: list[str], option: str) -> str | None:
    if option not in arguments:
        return None
    index = arguments.index(option)
    if index + 1 >= len(arguments):
        raise ValueError(f"{option} 缺少取值")
    value = arguments[index + 1]
    del arguments[index : index + 2]
    return value


def _csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


class EnvironmentCommands:
    def __init__(self, conversation, console, switch: Switch):
        self.conversation = conversation
        self.console = console
        self.switch = switch
        self.store = conversation.environments
        self.editor = EnvironmentEditor(self.store)

    # Dispatch -----------------------------------------------------------------------

    def run(self, arguments: list[str]) -> bool:
        action, rest = arguments[0], list(arguments[1:])
        handler = getattr(self, f"cmd_{action}", None)
        if handler is None:
            raise ValueError(USAGE)
        return handler(rest)

    def target(self, arguments: list[str]) -> str:
        selector = _take(arguments, "--env")
        if selector is not None:
            return self.store.find(selector).environment_id
        if self.conversation.environment_id is None:
            raise ValueError("当前没有启用自定义环境；先 /env use 名称，或用 --env 名称 指定环境")
        return self.conversation.environment_id

    def applied(self, result, message: str) -> bool:
        console = self.console
        console.success(message)
        if not result.approved:
            console.warning(
                f"环境此前未确认或资源已改变，修改已保存但未自动信任；"
                f"用 /env trust {result.entry.environment_id} 确认后生效。"
            )
            return True
        for note in result.environment.notes:
            console.warning(note)
        if result.entry.environment_id == self.conversation.environment_id:
            return self.reapply("修改已应用到当前环境，对话保留。")
        return True

    def reapply(self, message: str) -> bool:
        from command_ui import sync_environment_ui

        environment = self.store.resolve(self.conversation.environment_id)
        warnings = CommandActions(self.conversation).apply_environment(
            environment, keep=True, on_connect=getattr(self.console, "mcp_connecting", None)
        )
        sync_environment_ui(self.conversation, self.console)
        for warning in warnings:
            self.console.warning(warning)
        self.console.info(message)
        return True

    # Lifecycle ----------------------------------------------------------------------

    def cmd_help(self, _arguments) -> bool:
        print(USAGE)
        return True

    def cmd_list(self, arguments) -> bool:
        if arguments:
            raise ValueError("用法：/env list")
        entries = self.store.list()
        current = self.conversation.environment_id
        self.console.info("默认" + ("（当前）" if current is None else "") + " · 进程启动时的能力")
        for entry in entries:
            marker = "（当前）" if entry.environment_id == current else ""
            try:
                environment = self.store.resolve(entry.environment_id)
                trust = "已确认" if self.store.approved(environment) else "待确认"
                detail = f"{environment.team.kind} · {trust} · {environment.description}"
            except (OSError, ValueError) as exc:
                detail = f"无法解析：{exc}"
            self.console.info(f"{entry.environment_id}{marker} · {detail}")
        for diagnostic in self.store.diagnostics:
            self.console.warning(diagnostic)
        if not entries:
            self.console.info(
                "还没有自定义环境；用 /env create 名称 [--template multi-agent] 创建。"
            )
        return True

    def cmd_create(self, arguments) -> bool:
        template = _take(arguments, "--template") or "single-agent"
        description = _take(arguments, "--description")
        scope = _take(arguments, "--scope") or "user"
        if len(arguments) != 1:
            raise ValueError(
                "用法：/env create 名称 [--template single-agent|multi-agent|coding-team] "
                "[--description 文本] [--scope user|workspace|program]"
            )
        result = self.editor.create(
            arguments[0], template=template, description=description, scope=scope
        )
        self.console.success(
            f"已创建环境 {result.entry.environment_id}（{template}）：{result.entry.path}"
        )
        self.console.info(
            f"用 /env use {arguments[0]} 进入；/env skills、/env mcp、/env agents 管理能力。"
        )
        return True

    def cmd_use(self, arguments) -> bool:
        if not arguments or arguments[0] in SUBCOMMANDS:
            raise ValueError("用法：/env use 名称 [keep] [--yes]")
        return self.switch(self.conversation, self.console, arguments)

    def cmd_exit(self, arguments) -> bool:
        if arguments:
            raise ValueError("用法：/env exit")
        if self.conversation.environment_id is None:
            self.console.info("当前已是默认代理，没有需要退出的环境。")
            return True
        return self.switch(self.conversation, self.console, ["off"])

    def cmd_reload(self, arguments) -> bool:
        if arguments:
            raise ValueError("用法：/env reload")
        if self.conversation.environment_id is None:
            raise ValueError("当前没有启用自定义环境")
        return self.reapply("已重新加载环境定义，对话保留。")

    def cmd_type(self, arguments) -> bool:
        selector = self.target(arguments)
        if len(arguments) != 1:
            raise ValueError("用法：/env type single-agent|multi-agent [--env 名称]")
        result = self.editor.set_kind(selector, arguments[0])
        return self.applied(result, f"环境类型已设为 {arguments[0]}。")

    def cmd_status(self, arguments) -> bool:
        if arguments:
            raise ValueError("用法：/env status")
        conversation = self.conversation
        console = self.console
        environment = conversation.active_environment
        if environment is None:
            console.info("当前：默认代理（未启用自定义环境）。用 /env list 查看环境。")
            tools = ", ".join(item["function"]["name"] for item in conversation.definitions)
            console.info(f"工具：{tools or '无'}")
            skills = conversation.skills
            console.info(
                "技能：" + (", ".join(skill.name for skill in skills.skills) or "无")
                if skills is not None and skills.enabled
                else "技能：关闭"
            )
            return True
        console.info(
            f"当前环境：{environment.environment_id} · {environment.team.kind} · "
            f"{environment.description}"
        )
        console.info(f"会话：{conversation.session_id}")
        skills = conversation.skills
        console.info(
            "启用技能：" + (", ".join(skill.name for skill in skills.skills) or "无")
            if skills is not None and skills.enabled
            else "启用技能：关闭"
        )
        if conversation.mcp_manager is not None:
            for server in conversation.mcp_manager.status().data["servers"]:
                console.info(
                    f"  MCP {server['name']}：{server['status']} · {server.get('tool_count', 0)} 个工具"
                )
        else:
            console.info("MCP：未挂载")
        for note in environment.notes:
            console.warning(note)
        team = getattr(conversation, "team", None)
        if team is not None:
            for line in team.status_lines():
                console.info(line)
        elif environment.multi_agent:
            console.warning("团队运行时未启动；用 /env reload 重试。")
        return True

    def cmd_log(self, arguments) -> bool:
        if len(arguments) > 1:
            raise ValueError("用法：/env log [数量]")
        team = getattr(self.conversation, "team", None)
        if team is None:
            raise ValueError("当前环境没有团队运行时（仅 multi-agent 环境记录团队事件）")
        try:
            limit = int(arguments[0]) if arguments else 20
        except ValueError as exc:
            raise ValueError("数量必须是整数") from exc
        lines = team.event_lines(max(1, min(500, limit)))
        if not lines:
            self.console.info("暂无团队事件。")
        for line in lines:
            self.console.info(line)
        return True

    def cmd_results(self, arguments) -> bool:
        if arguments:
            raise ValueError("用法：/env results")
        team = getattr(self.conversation, "team", None)
        if team is None:
            raise ValueError("当前环境没有团队运行时（仅 multi-agent 环境有成员结果）")
        for line in team.result_lines():
            self.console.info(line)
        return True

    # Skills -------------------------------------------------------------------------

    def cmd_skills(self, arguments) -> bool:
        selector = self.target(arguments)
        action = arguments[0] if arguments else "list"
        if action == "list":
            environment = self.store.resolve(selector)
            available = self.editor.available_skills(selector)
            selection = environment.skill_selection
            if selection is None:
                self.console.info("此环境未设置技能选择：使用所有来源中发现的技能。")
            else:
                self.console.info(
                    "环境技能："
                    + (
                        ", ".join(
                            f"{name}{'' if enabled else '（停用）'}"
                            for name, enabled in selection.items()
                        )
                        or "无"
                    )
                )
            addable = [skill for skill in available if not selection or skill.name not in selection]
            for skill in addable:
                self.console.info(f"  可添加 {skill.name}: {skill.description}")
            return True
        names = arguments[1:]
        result = self.editor.edit_skills(selector, action, names)
        return self.applied(result, f"技能已更新：{action} {' '.join(names)}")

    # MCP ----------------------------------------------------------------------------

    def cmd_mcp(self, arguments) -> bool:
        selector = self.target(arguments)
        action = arguments[0] if arguments else "list"
        if action == "list":
            environment = self.store.resolve(selector)
            available = self.editor.available_mcp(selector)
            selection = environment.mcp_selection
            if selection is None:
                self.console.info("此环境未设置 MCP 选择：使用全部已配置服务。")
            for name, item in (selection or {}).items():
                state = "启用" if item.enabled else "停用"
                tools = ",".join(item.tools) if item.tools is not None else "全部工具"
                deny = f"；排除 {','.join(item.deny)}" if item.deny else ""
                missing = "；未在任何来源配置" if name not in available else ""
                self.console.info(f"  {name} · {state} · {tools}{deny}{missing}")
            for name, server in available.items():
                if selection is None or name not in selection:
                    target = server.command or server.url or ""
                    self.console.info(f"  可添加 {name} · {server.transport} · {target}")
            if not available:
                self.console.info("没有可添加的 MCP 服务；用 /env mcp add 名称 --command … 定义。")
            return True
        if len(arguments) < 2:
            raise ValueError("用法：/env mcp add|remove|enable|disable|tools 服务 …")
        name = arguments[1]
        options = arguments[2:]
        tools = _csv(_take(options, "--tools"))
        deny = _csv(_take(options, "--deny"))
        if action == "add" and options:
            server = parse_server_options(
                ["--env" if option == "--set-env" else option for option in options]
            )
            result = self.editor.define_mcp_server(selector, name, server)
            if tools or deny:
                result = self.editor.edit_mcp(selector, "tools", name, tools=tools, deny=deny)
            return self.applied(result, f"已在环境 MCP 文件中定义并启用服务 {name}。")
        if action == "tools":
            tools = [*options, *(tools or [])]
            if not tools and not deny:
                raise ValueError("用法：/env mcp tools 服务 模式… [--deny 模式,…]；* 表示全部")
            tools = None if tools == ["*"] else tools
        elif options:
            raise ValueError(f"无法识别的参数：{' '.join(options)}")
        result = self.editor.edit_mcp(selector, action, name, tools=tools, deny=deny)
        return self.applied(result, f"MCP 已更新：{action} {name}")

    # Agents -------------------------------------------------------------------------

    def cmd_agents(self, arguments) -> bool:
        selector = self.target(arguments)
        action = arguments[0] if arguments else "list"
        environment = self.store.resolve(selector)
        if action == "list":
            if not environment.multi_agent:
                self.console.info("single-agent 环境只有主代理；/env type multi-agent 可添加成员。")
                return True
            if not environment.team.agents:
                self.console.info("还没有成员；用 /env agents add 名称 --description … 添加。")
            for agent in environment.team.agents:
                self.console.info(self.describe_agent(agent))
            return True
        if action == "show":
            if len(arguments) != 2:
                raise ValueError("用法：/env agents show 名称")
            agent = environment.team.agent(arguments[1])
            if agent is None:
                raise ValueError(f"成员 {arguments[1]!r} 不存在")
            self.console.info(self.describe_agent(agent))
            self.console.info(agent.instructions or "（无额外指令）")
            return True
        if len(arguments) < 2:
            raise ValueError("用法：/env agents add|set|remove|enable|disable 名称 [选项]")
        name, options = arguments[1], list(arguments[2:])
        values = self.agent_values(options)
        if options:
            raise ValueError(f"无法识别的参数：{' '.join(options)}")
        result = self.editor.edit_agent(selector, action, name, values)
        return self.applied(result, f"成员已更新：{action} {name}")

    @staticmethod
    def agent_values(options: list[str]) -> dict:
        values: dict = {}
        for option, key in (
            ("--description", "description"),
            ("--instructions", "instructions"),
            ("--prompt-file", "prompt_file"),
            ("--effort", "reasoning_effort"),
        ):
            value = _take(options, option)
            if value is not None:
                values[key] = value
        for option, key in (
            ("--tools", "tools"),
            ("--skills", "skills"),
            ("--write-paths", "write_paths"),
        ):
            value = _csv(_take(options, option))
            if value is not None:
                values[key] = value
        mcp = _csv(_take(options, "--mcp"))
        if mcp is not None:
            scope: dict = {}
            for item in mcp:
                server, _, pattern = item.partition(":")
                patterns = scope.setdefault(server, [])
                if pattern:
                    patterns.append(pattern)
            values["mcp"] = {server: patterns or "*" for server, patterns in scope.items()}
        return values

    @staticmethod
    def describe_agent(agent) -> str:
        summary = agent.summary()
        tools = summary["tools"]
        skills = summary["skills"]
        return (
            f"{agent.name}{'' if agent.enabled else '（停用）'} · {agent.description or '无描述'}"
            f" · tools={','.join(tools) if isinstance(tools, list) else tools}"
            f" · skills={','.join(skills) if isinstance(skills, list) else skills}"
            f" · mcp={summary['mcp']}"
        )

    # Limits -------------------------------------------------------------------------

    def cmd_limits(self, arguments) -> bool:
        selector = self.target(arguments)
        if not arguments:
            environment: ResolvedEnvironment = self.store.resolve(selector)
            for key, value in environment.team.limits.__dict__.items():
                self.console.info(f"  {key} = {format_limit(value)}")
            return True
        result = self.editor.set_limits(selector, arguments)
        return self.applied(result, "资源限制已更新。")


def environment_subcommand(conversation, console, arguments: list[str], switch: Switch) -> bool:
    return EnvironmentCommands(conversation, console, switch).run(arguments)


def enter_multiagents(conversation, console, arguments: list[str], switch: Switch) -> bool:
    """Shortcut: create the default multi-agent environment if needed, then enter it."""
    if arguments:
        console.error(ValueError("用法：/multiagents"))
        return False
    store = conversation.environments
    name = "multi-agent"
    matches = [entry for entry in store.list() if entry.name == name]
    if not matches:
        result = EnvironmentEditor(store).create(name, template="multi-agent")
        console.success(f"已创建默认多代理环境 {result.entry.environment_id}：{result.entry.path}")
        selector = result.entry.environment_id
    elif len(matches) == 1:
        selector = matches[0].environment_id
    else:
        console.error(ValueError("存在多个 multi-agent 环境；请用 /env use 来源:multi-agent 选择"))
        return False
    return switch(conversation, console, [selector])
