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

会话负责组织消息和上下文，任务运行时调度模型与工具调用。子代理的编辑经过文件状态与冲突处理；执行记录用于界面观察和恢复。MCP、技能、环境定义、模板与 workflow 由用户提供资源，框架仓库只保留加载和管理代码。

运行期间产生的会话、日志、命令输出与检查点可能包含任务原文，应保存在本地私密位置。配置加载和 Git 边界分别见 [配置入口](configuration.md) 与 [Git 策略](git-policy.md)。
