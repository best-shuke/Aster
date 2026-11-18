# Skills 扩展目录

框架保留技能发现与加载接口，但不附带具体 skills。请在本机配置自己的技能目录，或通过 `--skill` 指定资源；实现见 `skills.py`。

用户技能默认位于 `~/.agents/aster/skills/`，随 `ASTER_HOME` 更改，也可用 `DEEPSEEK_SKILLS_DIR` 单独指定；工作区与其他共享技能目录继续参与发现。本目录中的兼容资源默认被 Git 忽略。不要提交含账号、内部路径、服务凭据或真实任务数据的技能文件。
