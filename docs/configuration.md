# 配置入口

所有默认值与解析规则以 `config.py` 为准。仓库不包含 `.env`、MCP 配置、个人技能或环境定义；请按自己的服务和工作区创建。

| 环境变量 | 用途 |
| --- | --- |
| `DEEPSEEK_API_KEY` | 模型凭据，运行模型功能需要 |
| `DEEPSEEK_BASE_URL` | 模型服务地址，默认 `https://api.deepseek.com` |
| `DEEPSEEK_MODEL` | 模型名称，默认 `deepseek-chat` |
| `DEEPSEEK_ENV_FILE` | 私密配置文件位置；启动脚本固定使用程序目录的 `.env` |
| `DEEPSEEK_SESSION_FILE` | 会话文件位置 |
| `DEEPSEEK_PROMPT_FILE` | 自定义系统提示文件 |
| `DEEPSEEK_MCP_ENABLED` / `DEEPSEEK_MCP_CONFIG_FILE` | MCP 开关与本地配置路径 |
| `DEEPSEEK_SKILLS_ENABLED` / `DEEPSEEK_SKILLS_DIR` | 技能发现开关与目录 |
| `DEEPSEEK_ENVIRONMENTS_DIR` | 用户能力环境定义目录 |
| `DEEPSEEK_ENVIRONMENT_STATE_DIR` | 本地能力环境信任与运行状态目录 |
| `DEEPSEEK_TEMPLATES_DIR` / `DEEPSEEK_WORKFLOWS_DIR` | 本地模板与 workflow 目录 |
| `DEEPSEEK_WEBSEARCH_ENABLED` | 联网搜索开关 |
| `DEEPSEEK_WEBSEARCH_BASE_URL` / `DEEPSEEK_WEBSEARCH_API_KEY` | 搜索服务地址与独立凭据 |
| `DEEPSEEK_REASONING_EFFORT` | 推理档位：`off`、`low`、`high`、`max`、`default` |
| `DEEPSEEK_MAX_CONTEXT_CHARS` / `DEEPSEEK_CONTEXT_WINDOW_TOKENS` | 上下文预算 |
| `DEEPSEEK_MAX_OUTPUT_TOKENS` / `DEEPSEEK_MAX_TASK_SECONDS` | 输出上限与任务时限 |
| `DEEPSEEK_MAX_SUBAGENTS` | 子代理并发上限 |

配置先从进程环境读取；本地文件只补充尚未设置的值。直接运行 Python 入口时，未指定配置文件则读取当前目录的 `.env`；`bin/agent` 则使用程序目录的 `.env`。

默认资源目录位于程序目录下：`skills/`、`environments/`、`templates/`、`workflows/` 和 `mcp.json`。本仓库的这些目录只放说明文件，实际资源默认被 Git 忽略。自定义路径与其他参数的具体解析方式见源码；为避免相对路径歧义，可使用自己机器上的绝对路径。

MCP 支持本地子进程与远程服务，接入的服务、参数、认证头和环境变量均由使用者自行管理。联网搜索有独立服务设置，模型凭据仅在服务同源时自动复用；无需搜索时可以禁用。没有外部服务或技能配置时仍可使用核心编程工具。
