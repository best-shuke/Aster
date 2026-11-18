# Aster · 星芒

Aster 是一个支持研究、分析、写作、规划和编程等任务的终端助手框架，提供流式模型调用、文件和命令工具、任务编排、子代理协作、上下文管理、会话恢复及终端界面。

本仓库只包含框架源码、通用测试、依赖锁文件和文档。用户自行选择模型服务、MCP、skills 和环境配置；仓库不附带任何实际服务配置、凭据、会话、个人技能或具体能力包。

## 启动

需要 Python 3.10+、uv 和 ripgrep（`rg`）。终端入口及测试主要面向 macOS / Linux。

```sh
git clone git@github.com:best-shuke/Aster.git
cd Aster
uv sync --locked
./bin/agent --help
```

首次在终端启动时会询问登录方式：

- **ChatGPT 账户（Plus / Pro）**：用 ChatGPT 订阅额度运行。可选浏览器登录，或在手机/任意电脑上输入设备码登录（适合 SSH 和无图形界面的服务器）；登录后从账户可用的模型中选择。
- **API 服务（OpenAI / Anthropic 兼容）**：支持 Chat Completions、Responses、Anthropic Messages；填写服务地址、API Key、协议与模型名称，保存前会尝试验证 Key。
- **沿用环境变量配置**：已设置 `DEEPSEEK_API_KEY`（及可选的 `DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL`）时出现，保持原有用法。

选择与凭据保存在 `~/.agents/aster/auth.json`（仅当前用户可读，不在仓库内）。之后可随时用 `/login` 切换登录方式或模型、`/logout` 退出登录，或用 `./bin/agent --login` 在启动前重新选择。

```sh
./bin/agent
./bin/agent "说明当前工作区的代码结构"
```

也可以只用环境变量：在终端环境或 `~/.agents/aster/.env` 中设置 `DEEPSEEK_API_KEY` 等变量；仓库不提供实际配置文件。单次执行（带任务参数）且已配置环境变量时不会弹出登录菜单。

`DEEPSEEK_API` 可显式选择 `openai-completions`、`openai-responses` 或 `anthropic-messages`；默认 `auto` 按服务地址选择，DeepSeek 官方默认沿用 Chat Completions。协议和服务兼容设置见[配置入口](docs/configuration.md#模型通信协议)。

`bin/agent` 与直接运行 `uv run --locked python main.py` 都默认读取 `~/.agents/aster/.env`，保留当前调用目录作为工作区。配置、登录凭据、技能、模板、workflow、环境状态和会话统一保存在 `~/.agents/aster/`；会话按工作区路径隔离，工作区内不再生成默认会话文件或锁文件。可通过进程环境变量 `ASTER_HOME` 更改整个用户目录，原有 `DEEPSEEK_*` 路径配置仍可单独覆盖。已有进程环境变量优先于文件配置，旧配置的迁移见[配置入口](docs/configuration.md#用户目录与迁移)。

联网搜索：直连 DeepSeek 官方模型时默认使用官方 Search API，复用原有 API Key；以 ChatGPT 账户登录时直接使用账户自带的搜索，无需额外配置；其他第三方 API 可配置 DeepSeek / Cline 搜索凭据（`DEEPSEEK_WEBSEARCH_API_KEY`），详见[配置入口](docs/configuration.md#运行与扩展配置)。

如不需要扩展，可通过自己的环境变量设置 `DEEPSEEK_MCP_ENABLED=0`、`DEEPSEEK_SKILLS_ENABLED=0` 和 `DEEPSEEK_WEBSEARCH_ENABLED=0`。缺少 MCP 配置或技能目录不会要求安装具体服务。

## 框架能力

- 流式响应、重试、限流协调、用量与提示缓存统计。
- 文件读取、搜索、编辑、命令执行及后台进程观察。
- 任务计划、子代理调度、文件修改冲突检测与检查点恢复。
- `/goal` 持续目标：按验收条件自动推进多轮工作，支持总轮次/时间预算、暂停恢复及会话持久化。
- 上下文预算、压缩、会话保存、历史恢复及执行轨迹。
- MCP 客户端、skills 发现、环境切换、模板与 workflow 扩展接口。
- 环境：按环境独立选择技能、MCP 服务与工具权限；`multi-agent` 环境运行多个独立成员代理，支持直接通信、任务依赖、共享产物与会议/辩论/独立调查等可选协议；交互会话中成员可在两次提问之间继续后台工作，并发编辑同一文件时自动三方合并。
- 交互式终端、命令补全和 Markdown 输出。

常用命令：`/help`、`/status`、`/login`、`/new`、`/history`、`/resume`、`/trace`、`/tasks`、`/goal`、`/skills`、`/environment`（`/env`）、`/multiagents`。完整说明可在运行后用 `/help` 与 `/env help` 查看。

快捷键：行尾输入 `\` 再回车换行，`Tab` 补全命令，`Ctrl-T` 打开执行轨迹，任务运行中按 `Esc` 或 `Ctrl-C` 暂停。

```text
/env create research-team --template multi-agent
/env use research-team
/env skills add brainstorming
/env status
根据我提供的论文材料完成一份研究综述，让团队自主分工、比较方法、核查引用并独立评审。
/env exit
```

## 开发与测试

```sh
uv sync --locked
TERM=xterm-256color uv run --locked python -m unittest discover -s tests
```

离线任务评测使用固定模型响应和临时工作区，不读取真实凭据或调用模型服务。单独重跑并输出指标：

```bash
TERM=xterm-256color ASTER_EVAL_METRICS=1 uv run --locked python -m unittest tests.test_task_evaluation
```

指标包含完成状态、修复次数、工具与模型请求数、上下文压缩、耗时和已报告用量。耗时仅用于观察，不作为 CI 门槛。

终端渲染测试需要正常终端类型，上述命令显式设置 `TERM`，也可用于无交互执行环境。

测试使用合成数据、临时目录和模拟模型客户端；部分协议测试会启动本地 fixture 进程或回环服务，不需要真实模型密钥。具体 MCP/技能能力包及其专用测试不在本框架仓库内。

## 文档

- [架构](docs/architecture.md)：模块划分与执行链路。
- [环境](docs/environments.md)：环境定义、能力继承、多代理运行时与协作协议。
- [持续目标](docs/goals.md)：自动续跑、验收、预算及暂停恢复。
- [多代理演示](docs/multi-agent-demo.md)：一次真实运行的分工、直接求助与会议记录。
- [配置入口](docs/configuration.md)：模型、目录与扩展设置。
- [Git 与私密数据](docs/git-policy.md)：提交边界。
- [贡献指南](CONTRIBUTING.md)：开发和验证方式。

应用可执行命令并修改工作区文件；实际权限取决于运行用户和接入的外部工具。请在自己的工作区和授权范围内配置服务与任务。
