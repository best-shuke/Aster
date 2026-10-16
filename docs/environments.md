# 环境（Environment）

环境是一个独立的执行空间：它决定代理可用的技能、MCP 服务、工具权限和资源限制，并可以包含多个独立协作的成员代理。没有启用自定义环境时，程序保持原有的单代理行为。

- **环境定义**：可复用的 JSON 配置，位于程序目录 `environments/`、用户目录 `~/.agents/environments/` 或工作区 `.agent/environments/`。资源指纹与信任机制见 [架构](architecture.md)。
- **环境会话**：环境的一次运行实例。每个对话会话（`/new`、`/resume`）对应一个独立的环境会话；多代理环境的任务、消息、会议、成员上下文和事件都保存在会话里，会话之间默认互相隔离。

## 快速开始

```text
/env create research-team --template multi-agent   创建团队环境（默认写入用户目录并记录信任）
/env use research-team                              进入环境，开启新会话
/env skills                                         查看已选技能与可添加技能
/env skills add research                            添加技能；disable/enable/remove 管理
/env mcp                                            查看已选 MCP 服务与可添加服务
/env mcp add github --tools 'get_*,search_*'        引用已配置的服务，并限制工具
/env agents                                         查看成员
/env agents add analyst --description '数据分析' --tools read,grep,exec
/env limits max_meeting_rounds=2                    查看或修改资源限制
/env status                                         当前环境、成员、任务、协议
/env log 50                                         团队事件
/env results                                        成员后台完成的工作与发给主代理的结果
/env exit                                           退出环境，恢复默认代理
/multiagents                                        进入默认团队环境（不存在时自动创建）
```

编辑命令默认作用于当前环境，`--env 名称` 可指定其他环境。修改当前环境后会立即重新加载并保留对话。原有命令 `/env 名称 [keep] [--yes]`、`/env off`、`/env show|trust|deny 名称` 不变。若已有环境恰好与新子命令同名（如 `status`），用 `/env use 名称` 切换。

## 定义文件

原有字段（`prompt_file`、`skills_dirs`、`mcp_config_file`、`inherit_mcp`、`inherit_skills`、`tools`、`worker_tools`、`role_prompts` 等）保持原义。新增字段均为可选，缺省时行为与之前一致：

| 字段 | 含义 |
| --- | --- |
| `type` | `single-agent`（默认）或 `multi-agent` |
| `skills` | 技能名 → 是否启用。存在时只启用其中为 `true` 的技能 |
| `mcp` | 服务名 → `true`/`false` 或 `{"enabled", "tools", "deny"}`；存在时只启用列出的服务，`tools`/`deny` 是工具名通配模式 |
| `agents` | 成员名 → 成员定义（仅 `multi-agent`） |
| `limits` | 资源限制，见下文 |
| `collaboration` | `protocols`（启用的协议）、`dynamic_agents`（是否允许运行时创建成员，默认否）、`broadcast`（是否允许 `all` 广播） |

成员定义字段：`description`、`instructions` 或 `prompt_file`（二选一）、`enabled`、`skills`（技能名数组）、`tools`（工具名数组）、`mcp`（服务名数组，或 服务 → 工具模式数组/`"*"`）、`write_paths`（允许写入的工作区相对路径）、`reasoning_effort`。

示例见 [examples/research-team.json](examples/research-team.json)；一次真实的多代理运行记录见 [多代理演示](multi-agent-demo.md)。

## 继承与覆盖规则

环境只能在进程启动时的能力范围内选择或收窄，不能扩大：

1. **工具**：环境 `tools.allow/deny` 收窄启动工具集；联网搜索与子代理上限不能超过启动设置。
2. **技能来源**：环境 `skills_dirs` → 启动技能（`inherit_skills` 为真时）→ 内置协作技能（仅在存在 `skills` 选择时作为来源）。同名技能以先出现的来源为准，用户技能总是覆盖内置技能。
3. **技能选择**：存在 `skills` 时只启用选中且为 `true` 的技能；技能仍按需加载——提示词只列名称、描述和位置，代理需要时读取全文。
4. **MCP 来源**：环境自己的 MCP 文件与启动 MCP 配置（`inherit_mcp`）合并，同名以环境为准；`mcp` 选择只在这些来源中按名称挑选并添加工具模式。
5. **成员能力 = 环境能力 ∩ 成员声明**：
   - 工具：未声明时只有只读工具（read、grep、find、ls、websearch）；`ask_user`、`monitor`、`update_plan` 始终只属于主代理；`team` 始终授予成员。
   - MCP：未声明时没有 MCP；声明后权限是环境 `mcp` 选择与成员声明的交集，两边的工具模式都必须满足。环境未启用的服务即使成员声明也不可用。
   - 技能：未声明时使用环境启用的全部技能；声明后只取其中环境已启用的部分。
6. **动态成员**：开启 `dynamic_agents` 后，成员可用 `spawn_agent` 创建会话级成员，其工具不能超过创建者自身的工具，且不授予 MCP。

## 凭据

环境定义不保存凭据。MCP 服务的令牌、密码和认证头继续使用原有机制：在 MCP 配置中写 `${环境变量}`，值来自进程环境或被 Git 忽略的 `.env`。`/env mcp add 名称 --command …|--url …` 新建环境专属服务时，认证头、看起来是凭据的环境变量与参数必须是 `${VAR}` 引用，URL 中不能内嵌用户名密码；该文件权限为 600。

## 多代理运行时

`multi-agent` 环境中，主代理（`main`）是用户的唯一对接者，同时也是团队成员；其他成员是真正独立的代理：

- 每个成员有自己的模型客户端、私有上下文（跨激活持久保存）、系统提示、工具集和预算。成员的生成循环复用主代理的 `TaskRuntime`（上下文压缩、输出恢复、限流、截止时间、取消）。
- 成员之间直接通信，不经过主代理。主代理通过 `team action=wait` 接收回复、任务结果和协议结论。
- **调度**：投递到空闲成员的消息立即启动一次激活（独立线程）；成员忙碌时消息留在邮箱，下一次激活合并处理；超过并发上限的成员排队。
- **轮次与后台**：成员的工作随主代理处理用户请求的轮次开始。轮次结束时：
  - 交互会话（终端或通过标准输入逐行提问）中，仍在进行的工作转入后台继续：成员之间照常投递、唤醒和完成任务，直到团队空闲、达到 `limits.background_seconds`（默认 30 分钟，0 表示关闭）或用户发出下一条消息——下一轮直接接管仍在运行的激活，不重做。后台工作仍计入上一轮的预算与次数限制。
  - 单次命令（`agent "提示"`）、Esc/Ctrl-C 中断、后台时限到达、切换会话或环境、退出进程时，进行中的激活被取消、其输入重新排队，下一轮自动继续。
  - 后台期间底部状态栏显示工作中的成员与新结果数量；每个成员结束、以及团队空闲或暂停时，在输入框上方打印一行提示。`/env results` 查看后台期间各成员的报告和发给主代理的结果，但不消耗它们；下一轮主代理收到“后台期间发生了什么”的提示，并用 `team action=wait` 读取结果。
  - 主代理在成员仍在工作时给出最终答复会被提醒一次；交互会话中它可以选择不等待，并告知用户成员在后台继续。此时本轮结果记为“后台继续”而不是“已完成”，成员尚未完成的修改不要求在本轮验证。
- **义务结算**：成员一轮结束时的最终消息会作为它未显式处理的请求的回复、所分配任务的结果（标记为推断完成）以及协议轮次的发言。若它仍在等待他人（未获回复的请求、委派的任务、自己发起的协议）或邮箱里还有未读消息，这些义务顺延到之后的激活；若形成等待环（A 等 B、B 等 A）则立即结算以避免死锁。成员失败或预算耗尽时也会自动答复请求者，任何代理都不会无限等待。

### team 工具

| 类别 | 动作 |
| --- | --- |
| 成员与状态 | `list_members`、`get_state`（只含本代理可见的内容） |
| 通信 | `send_message`（`kind=request` 需要回复；`reply_to` 回复；`to` 可为成员、`main`、`group:<id>`、`all`）、`create_group` |
| 任务 | `create_task`（`assignee`、`depends_on`）、`update_task`、`get_task`、`request_review` |
| 产物与决策 | `share_artifact`（team/group/private 可见性）、`get_artifact`、`record_decision` |
| 协议 | `start_protocol`、`contribute`、`conclude` |
| 其他 | `load_skill`（只能加载授予自己的技能）、`spawn_agent`（需开启）、主代理专用 `wait`、`cancel` |

### 协作协议

协议是可选的。所有协议由同一个引擎在相同的邮箱、产物和决策之上运行，区别只在轮数、哪些轮次为盲轮（参与者互相看不到发言、彼此消息被阻止）以及是否由主持人总结：

| 协议 | 默认轮数 | 盲轮 | 总结 |
| --- | --- | --- | --- |
| `meeting` | 2 | 0 | 主持人 `conclude` |
| `debate` | 3 | 0 | 主持人 `conclude`，可用 `stances` 指定立场 |
| `brainstorm` | 2 | 1 | 主持人 `conclude` |
| `independent_research` | 1 | 1 | 运行时汇总全部发现交给发起者 |
| `review_decision` | 1 | 1 | 主持人 `conclude` |
| `coordinate_work` | 1 | 0 | 主持人 `conclude` 并创建任务 |

轮数受 `limits.max_meeting_rounds` 限制；全部参与者 `ready=true` 时非盲轮可提前结束。结论生成团队可见的产物，`decisions` 记录到决策板。

### 内置协作技能

`builtin/skills/` 提供 `team-collaboration`、`task-delegation`、`independent-debate`、`brainstorming`、`conflict-resolution`、`team-retrospective`，描述何时求助、何时先独立调查、何时正式讨论、如何解决分歧、如何并行与控制成本。它们只是可选策略：环境通过 `skills` 选择启用哪些，成员用 `load_skill` 按需加载。

## 安全与资源限制

| 限制 | 默认 | 作用 |
| --- | --- | --- |
| `max_concurrent_agents` | min(3, 启动子代理上限) | 同时运行的成员数 |
| `max_agents` | 8 | 成员总数（含动态成员） |
| `max_messages_per_turn` | 200 | 每轮团队消息总数 |
| `max_activations_per_agent` | 8 | 每个成员每轮激活次数 |
| `max_message_hops` | 10 | 消息链长度，阻止循环转发（回复不计入） |
| `max_meeting_rounds` | 4 | 协议轮数上限 |
| `max_task_depth` | 3 | 任务嵌套深度，阻止无限委派 |
| `max_tool_calls_per_agent` | 80 | 每个成员每轮工具调用 |
| `token_budget_per_agent` | 400000 | 每个成员每轮的非缓存 token：输入中未命中提示缓存的部分加输出。使用服务返回的用量；服务未返回用量时按整个请求估算（偏高，不会偏低） |
| `activation_timeout_seconds` | 600 | 单次激活时限，轮次内同时受主任务时限约束 |
| `background_seconds` | 1800 | 交互会话中主代理答复后，成员最多在后台继续工作的秒数；0 表示轮次结束即暂停 |

其他边界：成员的文件工具不能访问环境状态目录、信任记录和主对话的会话文件（含原子写入的临时文件）；`write_paths` 可限制成员写入范围；代理之间看不到彼此的私信与私有上下文；同一环境会话同一时间只能被一个进程打开；状态文件绑定所属环境，跨环境加载会被拒绝；需要用户决定的事项由成员转交主代理，只有主代理能使用 `ask_user`。工具策略不是操作系统沙箱，`exec` 仍以运行用户的权限执行命令。

## 持久化与恢复

```text
<DEEPSEEK_ENVIRONMENT_STATE_DIR>/sessions/<来源-名称-路径摘要>/<会话ID>/
  state.json        共享状态：成员、邮箱、消息、任务、群组、产物、决策、协议、最近事件
  events.jsonl      完整事件日志
  agents/<成员>.json 每个成员的私有上下文
  lock              进程锁
```

每次状态变化都会持久化。进程中断后重新打开会话（`/resume`）时：进行中的激活输入重新排队并标记为“中断后重投”，已完成的任务、消息和决策原样保留；成员私有上下文中缺少结果的工具调用会补上“结果未知”的记录，提醒代理先核实实际状态再重复操作。

## 已知限制

- 后台工作只在进程运行期间进行；进程退出后剩余工作在下次打开会话的第一轮继续。
- **并发编辑**：成员直接写共享工作区。团队内所有代理（含主代理）共享一份编辑记录：某代理基于旧读取执行 `edit` 或整文件 `write` 时，若文件此后只被团队成员的文件工具改过，运行时把它的修改与最新内容按行三方合并，结果中的 `merged` 字段说明合并已发生；修改重叠时仍报 `file_changed`，代理需重新读取后再改。用户、`exec` 命令或其他程序造成的变更不会被自动合并，同样要求重新读取；单代理环境保持原有的读取版本检查。
- 预算与时限是协作式的：长时间的单个工具调用在其自身超时或取消前不会被打断。
- 代理行为由模型决定；运行时保证投递、调度、隔离和限制，但不保证模型选择了最优的协作方式。
