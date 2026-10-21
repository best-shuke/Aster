# Git 与私密数据

本仓库只发布框架源码、通用测试、依赖声明、锁文件和文档。具体模型凭据、MCP 配置、skills、环境能力包及个人模板和 workflow 不属于发布内容。

默认忽略本地环境配置、凭据文件、会话、命令输出、检查点、数据库、缓存、虚拟环境和编辑器文件。`skills/`、`environments/`、`templates/` 和 `workflows/` 中的用户资源默认忽略，仅保留目录说明。

在其他仓库中运行 Aster 时，工作区里的 `.agent-artifacts/`（命令输出与检查点）和会话快照目录 `.deepseek-session.sessions/` 会自带只含 `*` 的 `.gitignore`，不会出现在 `git status` 中；会话文件 `.deepseek-session.json` 及其锁文件仍在工作区根目录，可加入该仓库的忽略规则，或用 `DEEPSEEK_SESSION_FILE` 放到别处。

提交前检查：

```sh
git status --short --untracked-files=all
git diff --check
git diff --cached --check
git diff --cached --name-status
```

逐项检查暂存内容，确保没有真实密钥、认证头、内部服务地址、个人路径、账号、用户数据、历史对话或命令输出。不要通过 `git add -f` 绕过私密资源规则。

`.gitignore` 不会停止跟踪已经提交的文件，也不会检查源码中的硬编码秘密。公开前还需要检查提交历史、作者身份和实际 Git 对象。若凭据曾经公开，应先撤销或轮换，再处理仓库内容。
