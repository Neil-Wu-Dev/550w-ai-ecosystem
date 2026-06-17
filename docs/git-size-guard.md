# Git 大文件安全门

本仓库使用两道 Git Hook 防止 `.gitignore` 遗漏导致大文件被提交或推送：

1. 单个新增或修改文件的完整大小不得超过 `10 MiB`。
2. 单次提交中所有新增或修改文件的完整大小合计不得超过 `50 MiB`。
3. `pre-commit` 检查暂存区。
4. `pre-push` 重新检查即将推送的每个提交。

## 安装

在仓库根目录右键运行或通过 PowerShell 执行：

```powershell
.\scripts\install_git_size_guard.ps1
```

安装脚本会为当前仓库设置：

```text
core.hooksPath=.githooks
```

该配置不会影响其他 Git 仓库。每位新拉取仓库的开发者都需要运行一次安装脚本。

## 设计说明

安全门只会拒绝提交或推送，不会截断、删除或修改文件。被阻止后，应把模型、
Adapter、数据集、数据库、日志、压缩包或生成物移出 Git，并按需补充 `.gitignore`。

`git commit --no-verify` 可以跳过 `pre-commit`，但普通 `git push` 仍会触发
`pre-push` 复检。若要让任何客户端都无法绕过限制，还应在 GitHub 仓库或 CI 中
增加同样的服务端检查。
