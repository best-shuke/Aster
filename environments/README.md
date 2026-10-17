# 能力环境扩展目录

框架保留能力环境解析、切换和信任管理，但不附带具体环境定义、提示包、MCP 桥或技能包。用户可在本机配置自己的环境目录，或用 `/env create 名称 [--template multi-agent] --scope program` 在此创建；实现见 `environments.py` 与 `environment_transition.py`，格式见 [docs/environments.md](../docs/environments.md)。

本目录中的用户资源默认被 Git 忽略。
