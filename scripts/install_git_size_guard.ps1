# 为当前仓库启用版本化 Git Hooks。
# 该设置只修改当前仓库的 .git/config，不影响电脑上的其他 Git 项目。

$ErrorActionPreference = "Stop"

$repoRoot = git rev-parse --show-toplevel
if (-not $repoRoot) {
    throw "The current directory is not inside a Git repository."
}

git -C $repoRoot config --local core.hooksPath .githooks

$configuredPath = git -C $repoRoot config --local --get core.hooksPath
if ($configuredPath -ne ".githooks") {
    throw "Failed to configure the Git hooks path."
}

Write-Host "[Git Size Guard] Installed successfully." -ForegroundColor Green
Write-Host "  Single file limit: 10 MiB"
Write-Host "  Per-commit content limit: 50 MiB"
Write-Host "  Hooks path: $configuredPath"
