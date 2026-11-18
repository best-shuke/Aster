# 本地模板目录

框架提供模板保存与导入接口，仓库不附带个人对话模板。模板可能包含历史消息、来源路径和任务信息，应保留在本机；实现见 `templates.py`。

模板默认保存在 `~/.agents/aster/templates/`，随 `ASTER_HOME` 更改，也可用 `DEEPSEEK_TEMPLATES_DIR` 单独指定。本目录中的兼容资源默认被 Git 忽略。
