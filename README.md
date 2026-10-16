# Aster · 星芒

Aster 是一个终端编程助手框架，提供流式模型调用、文件和命令工具、任务编排、子代理协作、上下文管理、会话恢复及终端界面。

本仓库只包含框架源码、通用测试、依赖锁文件和文档。用户自行选择模型服务、MCP、skills 和环境配置；仓库不附带任何实际服务配置、凭据、会话、个人技能或具体能力包。

## 启动

需要 Python 3.10+、uv 和 ripgrep（`rg`）。终端入口及测试主要面向 macOS / Linux。

```sh
git clone git@github.com:best-shuke/Aster.git
cd Aster
uv sync --locked
./bin/agent --help
```

在自己的终端环境中设置 `DEEPSEEK_API_KEY`，并按服务需要设置 `DEEPSEEK_BASE_URL` 和 `DEEPSEEK_MODEL`。也可以自行在项目目录创建私密的 `.env` 文件；该文件被 Git 忽略，仓库不提供实际配置文件。

```sh
./bin/agent
./bin/agent "说明当前工作区的代码结构"
```

`bin/agent` 固定读取程序目录的 `.env`，并保留当前调用目录作为工作区。直接运行 `uv run --locked python main.py` 时，默认从当前工作目录读取 `.env`。已有进程环境变量优先于文件配置。

如不需要扩展，可通过自己的环境变量设置 `DEEPSEEK_MCP_ENABLED=0`、`DEEPSEEK_SKILLS_ENABLED=0` 和 `DEEPSEEK_WEBSEARCH_ENABLED=0`。缺少 MCP 配置或技能目录不会要求安装具体服务。

## 框架能力

- 流式响应、重试、限流协调、用量与提示缓存统计。
- 文件读取、搜索、编辑、命令执行及后台进程观察。
- 任务计划、子代理调度、文件修改冲突检测与检查点恢复。
- 上下文预算、压缩、会话保存、历史恢复及执行轨迹。
- MCP 客户端、skills 发现、环境切换、模板与 workflow 扩展接口。
- 环境：按环境独立选择技能、MCP 服务与工具权限；`multi-agent` 环境运行多个独立成员代理，支持直接通信、任务依赖、共享产物与会议/辩论/独立调查等可选协议。
- 交互式终端、命令补全和 Markdown 输出。

常用命令：`/help`、`/status`、`/new`、`/history`、`/resume`、`/trace`、`/tasks`、`/skills`、`/environment`（`/env`）、`/multiagents`。完整说明可在运行后用 `/help` 与 `/env help` 查看。

```text
/env create product-team --template multi-agent
/env use product-team
/env skills add brainstorming
/env status
设计并开发一个 AI 产品，让团队自主决定如何调研、讨论、分工和执行。
/env exit
```

## 开发与测试

```sh
uv sync --locked
TERM=xterm-256color uv run --locked python -m unittest discover -s tests
```

终端渲染测试需要正常终端类型，上述命令显式设置 `TERM`，也可用于无交互执行环境。

测试使用合成数据、临时目录和模拟模型客户端；部分协议测试会启动本地 fixture 进程或回环服务，不需要真实模型密钥。具体 MCP/技能能力包及其专用测试不在本框架仓库内。

## 文档

- [架构](docs/architecture.md)：模块划分与执行链路。
- [环境](docs/environments.md)：环境定义、能力继承、多代理运行时与协作协议。
- [多代理演示](docs/multi-agent-demo.md)：一次真实运行的分工、直接求助与会议记录。
- [配置入口](docs/configuration.md)：模型、目录与扩展设置。
- [Git 与私密数据](docs/git-policy.md)：提交边界。
- [贡献指南](CONTRIBUTING.md)：开发和验证方式。

应用可执行命令并修改工作区文件；实际权限取决于运行用户和接入的外部工具。请在自己的工作区和授权范围内配置服务与任务。
