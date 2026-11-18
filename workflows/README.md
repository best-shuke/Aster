# 本地 workflow 目录

框架提供 workflow 创建、导入与运行接口，仓库不附带个人任务流程。用户可通过 `/workflow` 管理本地 Markdown 流程；实现见 `workflows.py`。

workflow 默认保存在 `~/.agents/aster/workflows/`，随 `ASTER_HOME` 更改，也可用 `DEEPSEEK_WORKFLOWS_DIR` 单独指定。本目录中的兼容资源默认被 Git 忽略。
