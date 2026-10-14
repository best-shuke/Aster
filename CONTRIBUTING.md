# 贡献指南

使用 `uv sync --locked` 安装依赖，使用 `TERM=xterm-256color uv run --locked python -m unittest discover -s tests` 运行框架测试；显式设置 `TERM` 可避免无交互环境的 `dumb` 终端关闭渲染。

仓库通过 `.python-version` 固定使用 Python 3.12。`requires-python` 仅声明 `>=3.10`，若不固定版本，`uv` 会自动选用系统上最新的解释器；过新的 Python 可能还没有依赖（如 `pydantic-core`）的预编译包，从而退化为耗时的源码编译。如需其他受支持版本，改写该文件即可。

修改前阅读相邻模块，保持配置解析、执行逻辑和终端展示的边界。修改工具协议时同步检查 schema、处理器及权限约束；修改任务运行时保持取消、截止时间、冲突检测和恢复语义。

测试应使用临时目录、合成消息、模拟模型和本地 fixture；不要依赖真实账户、外部 MCP、个人 skills 或本地会话。新增功能按影响范围补充相邻测试，更新文档和必要的依赖锁文件。

提交前运行 `git diff --check`，检查暂存文件及差异。私密配置、密钥、会话、运行产物和用户扩展资源应留在本机，详见 [Git 策略](docs/git-policy.md)。忽略规则不替代内容审查。
