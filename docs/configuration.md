# 配置入口

所有默认值与解析规则以 `config.py` 为准。仓库不包含 `.env`、MCP 配置、个人技能或环境定义；请按自己的服务和工作区创建。

## 登录方式

模型来源有三种，首次在终端启动时选择，之后用 `/login`、`/logout` 或 `agent --login` 管理：

| 方式 | 说明 |
| --- | --- |
| ChatGPT 账户（Plus / Pro） | OAuth 登录（浏览器回调 `localhost:1455`，或 `auth.openai.com/codex/device` 设备码），请求发往 ChatGPT 的 Codex 后端并计入订阅额度；令牌过期前自动刷新，`/status` 显示额度窗口用量 |
| API 服务 | 支持 OpenAI Chat Completions、Responses、Anthropic Messages；保存地址、API Key、模型与协议 |
| 环境变量 | 使用下表中的 `DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL` |

选择与凭据保存在 `DEEPSEEK_AUTH_FILE`（默认 `~/.agents/aster/auth.json`，权限 0600）。保存的选择优先于环境变量；没有保存的选择时，单次执行和非交互运行直接使用环境变量。ChatGPT 刷新令牌只能使用一次，多个 Aster 进程通过同目录的锁文件共享刷新结果；请不要把该文件复制给其他程序同时使用。

| 环境变量 | 用途 |
| --- | --- |
| `ASTER_HOME` | 用户配置与状态根目录，默认 `~/.agents/aster`；在启动前设置 |
| `DEEPSEEK_AUTH_FILE` | 登录选择与凭据文件，默认 `~/.agents/aster/auth.json` |
| `DEEPSEEK_API_KEY` | 环境变量方式的模型凭据 |
| `DEEPSEEK_BASE_URL` | 模型服务地址，默认 `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | 模型名称，默认 `deepseek-chat` |
| `DEEPSEEK_API` | 通信协议：`auto`（默认）、`openai-completions`、`openai-responses`、`anthropic-messages` |
| `DEEPSEEK_API_COMPAT` | 可选 JSON 对象，覆盖模型服务的兼容规则，见下文 |
| `DEEPSEEK_ENV_FILE` | 私密配置文件位置，默认 `$ASTER_HOME/.env`；两个启动入口均尊重此覆盖 |
| `DEEPSEEK_SESSION_FILE` | 会话文件位置，默认 `$ASTER_HOME/sessions/<工作区路径 SHA-256>/session.json` |
| `DEEPSEEK_PROMPT_FILE` | 自定义系统提示文件 |
| `DEEPSEEK_MCP_ENABLED` / `DEEPSEEK_MCP_CONFIG_FILE` | MCP 开关与本地配置路径 |
| `DEEPSEEK_SKILLS_ENABLED` / `DEEPSEEK_SKILLS_DIR` | 技能发现开关与目录 |
| `DEEPSEEK_ENVIRONMENTS_DIR` | 用户能力环境定义目录 |
| `DEEPSEEK_ENVIRONMENT_STATE_DIR` | 本地能力环境信任、审计与多代理会话状态目录 |
| `DEEPSEEK_TEMPLATES_DIR` / `DEEPSEEK_WORKFLOWS_DIR` | 本地模板与 workflow 目录 |
| `DEEPSEEK_WEBSEARCH_ENABLED` | 联网搜索开关 |
| `DEEPSEEK_WEBSEARCH_PROVIDER` | 搜索协议：`auto`（默认）、`deepseek` 或 `cline`；`auto` 根据搜索地址（未填时根据模型地址）识别 DeepSeek 官方服务，其余沿用 Cline |
| `DEEPSEEK_WEBSEARCH_BASE_URL` / `DEEPSEEK_WEBSEARCH_API_KEY` | 搜索服务地址与独立凭据；地址留空自动选择，同源时复用模型 Key |
| `DEEPSEEK_REASONING_EFFORT` | 推理档位：`off`、`low`、`high`、`max`、`default` |
| `DEEPSEEK_MAX_CONTEXT_CHARS` / `DEEPSEEK_CONTEXT_WINDOW_TOKENS` | 上下文预算 |
| `DEEPSEEK_MAX_OUTPUT_TOKENS` / `DEEPSEEK_MAX_TASK_SECONDS` | 可选输出上限与任务时限，默认不限 |
| `ASTER_TASK_TOKEN_BUDGET` / `ASTER_SESSION_TOKEN_BUDGET` / `ASTER_GOAL_TOKEN_BUDGET` | 单次任务、会话和持续目标的可选 token 额度；默认不限 |
| `DEEPSEEK_MAX_SUBAGENTS` | 可选子代理并发上限，默认不限 |

普通生成默认由服务端决定输出上限。Anthropic Messages 原生接口要求提供 `max_tokens`，未配置时使用 8192 作为兼容值，可按模型能力显式配置。DeepSeek 官方 Messages 入口允许省略该字段。设置 token 预算也可能为单次请求指定可用输出额度。

`exec` 和 `delegate` 省略 `timeout` 时沿用父任务的截止时间。长命令可以用 `exec(background=true)` 启动，再通过 `monitor` 等待、读日志或停止。后台命令只在显式指定 `timeout` 时有自己的运行时限，退出应用或切换环境会清理这些进程。

启用 token 额度后，每次模型请求先预留估算输入与输出上限。主代理、团队成员、委派与摘要共享所属任务、会话和目标的额度；成员还受自己的单轮额度限制。已经开始的请求仍在原账本结算。响应有用量时，使用未命中缓存的输入与输出 token；正常响应缺少用量时使用估算，失败且用量未报告的请求单独记为未知。只报告部分用量时，已知部分仍计入，缺失部分保留未知标记。预留不可替代供应商计量，不是严格的费用上限。会话和目标恢复时延续已结算用量；同一进程内尚未结束的请求保留原目标账本及预留，迟到结算写回所属会话。

配置先从进程环境读取；本地文件只补充尚未设置的值。文件每行一个 `KEY=value`，可带 `export` 前缀，值可加引号，未加引号的值中 ` #` 之后视为注释。数值项写错时启动会指出具体变量，不会继续运行。Python 入口与 `bin/agent` 默认都读取用户目录中的 `.env`。

## 用户目录与迁移

默认目录为 `~/.agents/aster/`，可在启动前设置 `export ASTER_HOME=/自己的目录`。目录中的布局：

```text
~/.agents/aster/
  .env                       模型、功能开关及路径设置
  auth.json                  登录选择与凭据
  mcp.json                   MCP 配置
  skills/                    用户技能
  templates/                 对话模板
  workflows/                 Markdown workflow
  environments/              用户环境定义与能力资源
  environment-state/         信任、审计与团队运行状态
  sessions/<工作区路径摘要>/
    session.json             当前会话
    session.json.lock        并发锁
    session.sessions/        历史快照
    session.templates/       会话关联的旧版模板设置
```

同一个工作区重启后继续使用同一份会话；不同路径的同名项目各自隔离，通过符号链接访问同一工作区会使用相同会话。配置和会话目录在实际写入时按需创建。内置系统提示词和框架资源仍随程序发布；自定义提示词可放入用户目录并用 `DEEPSEEK_PROMPT_FILE` 指定。已有的工作区技能和显式选择的工作区环境继续参与发现。

升级后，请将旧的程序目录 `.env`、`mcp.json`、个人 `skills/`、`templates/`、`workflows/` 和环境资源复制到对应用户目录；旧的 `~/.agents/environments/` 与 `~/.agents/environment-state/` 分别对应新的 `environments/` 与 `environment-state/`。保留资源的相对目录结构；已有目标文件应先比较再合并。旧登录文件的默认位置没有改变。

旧会话不会自动移动或覆盖。迁移前退出该工作区中的 Aster 进程，在原工作区运行以下命令查看新会话路径：

```sh
/程序目录/.venv/bin/python -c 'import sys; sys.path.insert(0, "/程序目录"); from config import default_session_file; from pathlib import Path; print(default_session_file(Path.cwd()))'
```

将 `.deepseek-session.json` 复制为输出路径的 `session.json`，将 `.deepseek-session.sessions/` 和 `.deepseek-session.templates/` 分别复制到同级的 `session.sessions/` 和 `session.templates/`；无需复制锁文件。确认恢复正常后可自行清理旧文件。需要继续沿用自定义路径时，原有 `DEEPSEEK_ENV_FILE`、`DEEPSEEK_SESSION_FILE` 等覆盖仍生效；显式指定的相对会话路径仍以当前工作区为基准。

## 模型通信协议

登录方式与通信协议分别管理。ChatGPT 账户使用自身的 Codex Responses 后端；API Key 登录按所选协议构造请求。`/login` 可以选择协议，`/status` 的账户信息会显示当前 API 协议。保存的 API 登录同时保存协议与兼容设置，不受环境变量中其他服务的协议配置影响。

`auto` 的默认选择：DeepSeek 官方及未知代理使用 Chat Completions；OpenAI 官方使用 Responses；Anthropic 官方或基础地址包含 `/anthropic` 路径时使用 Messages。代理支持多种协议时应显式选择。这里的 `openai-completions` 指 `/chat/completions`，不是旧的 `/completions`。

DeepSeek 官方可使用同一个 API Key 与模型，通过以下配置切换：

| `DEEPSEEK_API` | `DEEPSEEK_BASE_URL` | 实际模型接口 |
| --- | --- | --- |
| `openai-completions` | `https://api.deepseek.com` | `/chat/completions` |
| `openai-responses` | `https://api.deepseek.com` | `/responses` |
| `anthropic-messages` | `https://api.deepseek.com/anthropic` | `/anthropic/v1/messages` |

```dotenv
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-flash
DEEPSEEK_API=openai-responses
```

修改 `.env` 后重启。也可以使用 `/login` 切换，新的模型客户端和搜索凭据从下一次提问生效。现有会话、子任务和工具继续使用统一内部消息格式。DeepSeek 的独立 Search API 在三种模型协议下均可使用，不依赖模型接口的内置 `web_search`。

协议适配支持文本、思考和自定义函数工具的同步及流式响应；不会因为选择了某种协议就自动启用其全部内置工具。DeepSeek Responses 每次回传对话历史，不使用 `previous_response_id`，也不请求其未实现的加密推理。DeepSeek Messages 会保留中途的 system 消息；其他 Messages 服务收到带明确标签的运行时 user 消息。

### 服务兼容设置

默认规则根据实际服务地址选择。自建代理可通过 `DEEPSEEK_API_COMPAT` 显式声明行为；只接受以下布尔选项，未知选项或非布尔值会报配置错误：

| 选项 | 用途 |
| --- | --- |
| `supports_developer_role` | Responses 的中途系统指令使用 `developer`，否则使用 `system` |
| `supports_encrypted_reasoning` | Responses 是否请求及回传加密推理；DeepSeek 默认为 `false` |
| `supports_store` | Responses 是否接受 `store: false`；DeepSeek 不发送此字段 |
| `supports_reasoning_effort` | OpenAI 格式的服务是否接受推理强度参数 |
| `requires_reasoning_content` | Chat Completions 的 assistant 历史是否需要 `reasoning_content`；DeepSeek 默认为 `true` |
| `adaptive_thinking` | Messages 使用 `thinking.type=adaptive` 与 `output_config.effort` |
| `allow_empty_thinking_signature` | Messages 是否允许回传空的思考签名；DeepSeek 默认为 `true` |

例如，Responses 代理未实现加密推理与 developer 角色时：

```dotenv
DEEPSEEK_API=openai-responses
DEEPSEEK_API_COMPAT={"supports_encrypted_reasoning":false,"supports_developer_role":false}
```

原生推理状态只向同一服务地址、协议与模型回传；正文被压缩或修改后也不复用旧状态。切换服务时保留普通对话和工具记录，避免把其他服务的签名或加密状态发送给新服务。

## 运行与扩展配置

`DEEPSEEK_MAX_TASK_SECONDS` 默认 `unlimited`，主任务不因固定总时长结束。已有配置中的数字仍生效；
希望持续工作时将该项改为 `unlimited` 并重启。网络空闲、单个工具操作的超时和用户取消分别处理。

环境定义的字段、技能与 MCP 的继承规则、团队成员与资源限制见 [环境](environments.md)；示例在 [docs/examples](examples/)。

默认资源目录均位于用户目录下：`skills/`、`environments/`、`templates/`、`workflows/` 和 `mcp.json`。程序目录中的同名扩展目录仅保留说明或显式安装的兼容资源。自定义路径与其他参数的具体解析方式见源码；为避免相对路径歧义，可使用自己机器上的绝对路径。

MCP 支持本地子进程与远程服务，接入的服务、参数、认证头和环境变量均由使用者自行管理。联网搜索按以下顺序选择后端：设置了 DeepSeek / Cline 搜索凭据（`DEEPSEEK_WEBSEARCH_API_KEY`，或模型服务与搜索服务同源时复用的模型凭据）时使用对应服务的原生搜索；否则以 ChatGPT 账户登录时，使用 Codex 后端托管的 `web_search`，由当前模型以低推理档位执行搜索，返回带引用的摘要、实际执行的查询和来源链接，消耗 ChatGPT 套餐额度。两者都没有时显示“未配置”。`/login` 切换登录方式后，搜索后端随之切换。无需搜索时可以禁用。没有外部服务或技能配置时仍可使用核心编程工具。

直连 DeepSeek 官方模型时，搜索默认使用 [官方 Search API](https://api-docs.deepseek.com/zh-cn/guides/search/)（`POST https://api.deepseek.com/v1/tools/search`），复用原有 DeepSeek API Key，无需 Cline Key。配置示例：

```dotenv
DEEPSEEK_WEBSEARCH_ENABLED=1
DEEPSEEK_WEBSEARCH_PROVIDER=auto
DEEPSEEK_WEBSEARCH_BASE_URL=https://api.deepseek.com/v1/tools
DEEPSEEK_WEBSEARCH_API_KEY=
```

也可把搜索地址留空，启动时根据所选模型服务自动选择。DeepSeek 地址可填写 `https://api.deepseek.com`、`https://api.deepseek.com/v1` 或完整的 `/v1/tools` 基础地址。若旧 `.env` 仍指定 Cline 地址，请修改或清空该项并重启。

DeepSeek 搜索将域名筛选转换为 `site:` / `-site:` 操作符，查询词加筛选条件总计最多 128 个 Unicode 字符。工具保留官方返回的长短摘要、链接、发布时间与排序分数；搜索请求不额外获取网页全文。搜索按官方 Search API 规则单独计费。Cline/ClinePass 继续使用 `/search/websearch` 接口；自建 DeepSeek 搜索代理需设置 `DEEPSEEK_WEBSEARCH_PROVIDER=deepseek`、搜索基础地址和独立 Key。
