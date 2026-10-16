# 框架架构

主链路由 `main.py` 启动，读取 `config.py` 配置，构建模型客户端、会话、工具和终端界面。配置使用既有的 `DEEPSEEK_*` 名称，项目名称为 Aster。

| 职责 | 模块 |
| --- | --- |
| CLI 与交互命令 | `main.py`、`commands.py`、`command_ui.py` |
| 配置、协议与系统提示 | `config.py`、`contracts.py`、`system_prompt.md` |
| 模型、传输、限流与用量 | `deepseek_client.py`、`client_runtime.py`、`http_transport.py`、`model_rate_limit.py`、`usage.py`、`generation_metrics.py` |
| 会话与上下文 | `conversation.py`、`sessions.py`、`context_budget.py`、`context_management.py` |
| 任务编排与验收 | `reasoning.py`、`task_runtime.py`、`task_acceptance.py`、`worker_checkpoint.py` |
| 工具与文件状态 | `tools.py`、`tool_runtime.py`、`file_state.py`、`file_changes.py` |
| 进程与执行记录 | `process_monitor.py`、`execution_history.py`、`execution_trace.py` |
| 终端展示 | `ui.py`、`terminal_rendering.py`、`trace_ui.py` |
| 扩展接口 | `mcp_client.py`、`skills.py`、`environments.py`、`environment_transition.py`、`templates.py`、`workflows.py`、`websearch.py` |
| 环境定义与管理 | `environment_spec.py`（成员、技能/MCP 选择、限制）、`environment_definitions.py`（创建与编辑）、`environment_commands.py`（`/env` 子命令）、`mcp_scope.py`（MCP 权限交集） |
| 多代理运行时 | `team_runtime.py`（调度与共享状态服务）、`team_agent.py`（成员能力与激活循环）、`team_state.py`（会话状态、持久化与恢复）、`team_tools.py`（`team` 工具与提示）、`team_protocols.py`（协作协议） |

会话负责组织消息和上下文，任务运行时调度模型与工具调用。子代理的编辑经过文件状态与冲突处理；执行记录用于界面观察和恢复。MCP、技能、环境定义、模板与 workflow 由用户提供资源，框架仓库只保留加载和管理代码；`builtin/` 中的协作技能与环境模板属于框架默认资源。

环境层组合而非替换原有代理：`EnvironmentStore` 解析定义，`EnvironmentTransition` 原子地切换设置、技能、工具、MCP 和提示，多代理环境额外挂载 `TeamRuntime` 与 `team` 工具。成员复用 `TaskRuntime`、`CodingTools`、`McpManager` 和 `SkillCatalog`，各自持有独立上下文与模型客户端。设计与规则见 [环境](environments.md)。

运行期间产生的会话、日志、命令输出与检查点可能包含任务原文，应保存在本地私密位置。配置加载和 Git 边界分别见 [配置入口](configuration.md) 与 [Git 策略](git-policy.md)。
