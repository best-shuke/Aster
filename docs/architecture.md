# 框架架构

主链路由 `main.py` 启动，读取保存的登录选择（首次启动时询问）与 `config.py` 配置，按登录方式构建模型客户端，再构建会话、工具和终端界面。配置使用既有的 `DEEPSEEK_*` 名称，项目名称为 Aster。

| 职责 | 模块 |
| --- | --- |
| CLI 与交互命令 | `main.py`、`commands.py`、`command_ui.py` |
| 配置、协议与系统提示 | `config.py`、`contracts.py`、`system_prompt.md` |
| 登录 | `auth.py`（登录选择存储、ChatGPT OAuth 与令牌刷新）、`login_ui.py`（登录菜单） |
| 模型、传输、限流与用量 | `model_api.py`（协议和服务兼容规则）、`model_clients.py`（客户端工厂）、`deepseek_client.py`（共享传输与 Chat Completions）、`responses_client.py` / `responses_protocol.py`（Responses 转换）、`anthropic_client.py`（Messages 转换）、`chatgpt_client.py`（ChatGPT 账户鉴权）、`client_runtime.py`、`http_transport.py`、`model_rate_limit.py`、`usage.py`、`generation_metrics.py` |
| 会话与上下文 | `conversation.py`、`sessions.py`、`context_budget.py`、`context_management.py` |
| 任务编排与验收 | `reasoning.py`、`task_runtime.py`、`task_acceptance.py`、`worker_checkpoint.py` |
| 持续目标 | `goals.py`（目标状态与命令参数）、`goal_runtime.py`（多轮调度、证据核对与预算） |
| 共享模型请求执行 | `model_request.py`（传输、重试、限流等待、截止时间、取消与逐次请求预算结算）；协议适配器提供请求构造、响应解析和鉴权 |
| 工具与文件状态 | `tools.py`、`tool_runtime.py`、`file_state.py`、`file_changes.py` |
| 进程与执行记录 | `process_monitor.py`、`execution_history.py`、`execution_trace.py` |
| 终端展示 | `ui.py`、`terminal_rendering.py`、`trace_ui.py` |
| 扩展接口 | `mcp_client.py`、`skills.py`、`environments.py`、`environment_transition.py`、`templates.py`、`workflows.py`、`websearch.py` |
| 环境定义与管理 | `environment_spec.py`（成员、技能/MCP 选择、限制）、`environment_definitions.py`（创建与编辑）、`environment_commands.py`（`/env` 子命令）、`mcp_scope.py`（MCP 权限交集） |
| 多代理运行时 | `team_runtime.py`（调度与共享状态服务）、`team_agent.py`（成员能力与激活循环）、`team_state.py`（会话状态、持久化与恢复）、`team_tools.py`（`team` 工具与提示）、`team_protocols.py`（协作协议） |

会话负责组织消息和上下文，任务运行时调度模型与工具调用。子代理的编辑经过文件状态与冲突处理；执行记录用于界面观察和恢复。MCP、技能、环境定义、模板与 workflow 由用户提供资源，框架仓库只保留加载和管理代码；`builtin/` 中的协作技能与环境模板属于框架默认资源。

启动时 MCP 服务在后台并发连接，首个请求不等待；搜索、描述与调用会等待仍在连接的服务，失败的服务在下次使用时重连，交互模式在下一次输入前报告后台连接失败。环境切换仍并发连接并等待结果，以便检查必需服务。搜索按任一词命中工具名或描述排序，`*` 列出全部工具，无匹配时返回权限范围内的工具路径。MCP 工具目录随 `notifications/tools/list_changed` 自动刷新；新版协议在首次查询前通过 `subscriptions/listen` 订阅，旧版使用通知回调。通知回调只标记目录失效，并合并刷新请求；分页查询完成后整体替换缓存，查询期间的新通知会触发下一次刷新。搜索、描述与调用通过同一连接队列获取当前目录，继续应用环境与成员的权限范围。刷新失败保留完整旧快照，并通过状态中的 `catalog_stale`、`catalog_error` 暴露问题；后续目录查询会重试，订阅中断后重新订阅并刷新，工具调用不会因此自动重放。
模型接入分别处理登录鉴权、通信协议与服务兼容规则。`model_clients.model_client()` 是启动和登录切换的统一入口；`api` 显式选择协议，`api_compat` 覆盖服务差异。各适配器转换请求与原生响应，继续使用 `contracts.Message` 和现有文本、思考、工具增量事件。`model_messages.py` 在发送前转换跨协议工具调用 ID，并同步更新工具结果的引用，不修改保存的历史。任务调度、子代理、重试、限流、取消、输出上限与统计复用同一条执行路径；ChatGPT 的令牌刷新和订阅额度处理独立于 Responses 协议转换。

原生思考块和加密推理保存在模型不可见的 `_harness` 中，附带服务地址、协议、模型与消息内容的身份校验。只在来源和内容均匹配时回传，避免跨服务发送签名状态或在上下文压缩后复用失效内容。请求用量身份也包含协议和兼容选项。SSE 必须收到各协议的完成事件才被接受；仅收到文本或流被截断时仍返回可恢复的失败。


修改代码的任务由 `task_acceptance.py` 验收：最后一次修改之后必须有一次退出状态能证明通过的检查（测试运行器、构建/类型检查、带 `assert` 的 `python -c` 或 here-document 程序、使用 `node:assert` 的 `node -e`、直接运行的测试脚本、`diff`/`cmp` 等）。只用 `&&` 串联的命令行中，退出码为零即证明其中每条命令都已成功，因此 `cd 目录 && 检查` 会被计入；管道、`;`、`||`、`&` 与换行会让退出状态不再属于检查本身，不被计入。答复时若唯一缺少的是这类未计入的检查，运行时会单独重跑一次该检查，由真实退出码决定接受或带输出退回，而不是再请求模型。只有能归因于检查本身的失败才会记为未解决的失败；运行器缺失（退出码 127、`No module named pytest` 等）表示检查没有运行，不记为失败。退回提示会写明需要重跑的失败命令及最后一次检查之后改动过的文件。

环境层组合而非替换原有代理：`EnvironmentStore` 解析定义，`EnvironmentTransition` 原子地切换设置、技能、工具、MCP 和提示，多代理环境额外挂载 `TeamRuntime` 与 `team` 工具。成员复用 `TaskRuntime`、`CodingTools`、`McpManager` 和 `SkillCatalog`，各自持有独立上下文与模型客户端。设计与规则见 [环境](environments.md)。

运行期间产生的会话、日志、命令输出与检查点可能包含任务原文，应保存在本地私密位置。配置加载和 Git 边界分别见 [配置入口](configuration.md) 与 [Git 策略](git-policy.md)。

## 会话持久化与性能边界

任务开始时，`Conversation.compact()` 已保存本轮用户输入与上下文，随后直接开始模型请求，
不再重复保存相同状态。工具执行记录、最终验收和异常恢复仍分别持久化；首次请求之前的
保存失败会中断任务，避免在无法保存恢复数据时继续执行。会话与目标的 token 账本也随保存
更新；后台请求的迟到结算写回原会话及对应目标，不归入后来创建的会话或目标。

2026-11-18 在本机临时目录、约 3 MB 合成会话上测量 12 次无工具工作轮次：每轮保存从
3 次降为 2 次，中位耗时从 31.016 ms 降为 20.741 ms，fsync 总数从 72 次降为 48 次。
这只衡量本地持久化开销，不代表真实模型任务的端到端提速。此前 4,000 文件扫描样本的
冷/热中位耗时为 73.455 / 20.469 ms；保留已有按需扫描、摘要缓存和结构化验收状态，
未据此新增数据库或进一步削弱修订检查。恢复产物继续保留，不自动删除仍可能被引用的文件。
