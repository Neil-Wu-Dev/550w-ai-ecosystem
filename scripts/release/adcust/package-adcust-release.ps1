# 生成 AdCust 正式二进制发布包。
# 输入：.build/adcust/<version>/app
# 输出：release/adcust/<version>
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..\..")
$version = if ($env:ADCUST_RELEASE_VERSION) { $env:ADCUST_RELEASE_VERSION } else { "1.0.0" }
$buildRoot = Join-Path $repoRoot ".build\adcust\$version"
$sourceRoot = Join-Path $buildRoot "app"
$workRoot = Join-Path $buildRoot "package-work"
$releaseVersionRoot = Join-Path $repoRoot "release\adcust\$version"
$packageName = "AdCust-$version-windows-portable"
$packageRoot = Join-Path $workRoot $packageName
$zipPath = Join-Path $releaseVersionRoot "$packageName.zip"

function Assert-Exists($path, $message) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw $message
    }
}

function Reset-Directory($path) {
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Recurse -Force
    }
    New-Item -ItemType Directory -Path $path -Force | Out-Null
}

function Copy-CleanTree($source, $destination) {
    Reset-Directory $destination
    Copy-Item -Path (Join-Path $source "*") -Destination $destination -Recurse -Force

    Get-ChildItem -LiteralPath $destination -Recurse -Force -Directory |
        Where-Object { $_.Name -eq "__pycache__" } |
        ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force }

    Get-ChildItem -LiteralPath $destination -Recurse -Force -File |
        Where-Object {
            $_.Extension -eq ".pyc" -or
            $_.Extension -eq ".pyo" -or
            $_.Extension -eq ".pyd"
        } |
        ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force }
}

function Write-PackageReadme($targetRoot) {
    $readme = @(
        "# AdCust $version Windows Portable",
        "",
        "This is the production binary package for AdCust $version.",
        "",
        "## How to run",
        "",
        "Double-click:",
        "",
        "````text",
        "AdCust Launcher.exe",
        "````",
        "",
        "The launcher will:",
        "",
        "1. Check whether the local backend is ready.",
        "2. Start the bundled backend runtime when needed.",
        "3. Wait for http://127.0.0.1:8000/health.",
        "4. Open the AdCust desktop app.",
        "",
        "## Important",
        "",
        "This portable package contains the AdCust application runtime, but it does not bundle large AI models, adapters, datasets, or your private compute-node configuration.",
        "",
        "Your local data will be created under data/.",
        "Backend logs will be written under logs/.",
        "",
        "## Python runtime",
        "",
        "This build expects the Python runtime configured when the release was built. If the Python path is different on another machine, rebuild the release with ADCUST_RELEASE_PYTHON set to your Python executable."
    ) -join [Environment]::NewLine

    Set-Content -LiteralPath (Join-Path $targetRoot "README.md") -Value $readme -Encoding UTF8 -ErrorAction Stop
}

function Write-ReleaseNotes($targetRoot) {
    $notes = @(
        "# AdCust $version",
        "",
        "Production binary release for Windows portable use.",
        "",
        "## Asset",
        "",
        "- AdCust-$version-windows-portable.zip",
        "",
        "## Run",
        "",
        "Extract the zip and double-click:",
        "",
        "````text",
        "AdCust Launcher.exe",
        "````",
        "",
        "## Notes",
        "",
        "This package contains the AdCust desktop app, launcher, and backend runtime copy. It does not include user datasets, model files, adapters, logs, or private compute-node configuration."
    ) -join [Environment]::NewLine

    Set-Content -LiteralPath (Join-Path $targetRoot "release-notes.md") -Value $notes -Encoding UTF8 -ErrorAction Stop
}

Write-Host "[AdCust Package] Source: $sourceRoot"
Write-Host "[AdCust Package] Release directory: $releaseVersionRoot"

Assert-Exists (Join-Path $sourceRoot "AdCust Launcher.exe") "Missing AdCust Launcher.exe. Run scripts\release\adcust\build-adcust-release.ps1 first."
Assert-Exists (Join-Path $sourceRoot "AdCust.exe") "Missing AdCust.exe. Run scripts\release\adcust\build-adcust-release.ps1 first."
Assert-Exists (Join-Path $sourceRoot "runtime\backend\app\main.py") "Missing backend runtime. Run scripts\release\adcust\build-adcust-release.ps1 first."
Assert-Exists (Join-Path $sourceRoot "runtime\adcust-logic\adcust_logic") "Missing adcust-logic runtime. Run scripts\release\adcust\build-adcust-release.ps1 first."

Reset-Directory $workRoot
New-Item -ItemType Directory -Path $releaseVersionRoot -Force | Out-Null
New-Item -ItemType Directory -Path $packageRoot -Force | Out-Null

Copy-Item -LiteralPath (Join-Path $sourceRoot "AdCust Launcher.exe") -Destination (Join-Path $packageRoot "AdCust Launcher.exe") -Force
Copy-Item -LiteralPath (Join-Path $sourceRoot "AdCust.exe") -Destination (Join-Path $packageRoot "AdCust.exe") -Force
Copy-CleanTree (Join-Path $sourceRoot "runtime") (Join-Path $packageRoot "runtime")

New-Item -ItemType Directory -Path (Join-Path $packageRoot "data") -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $packageRoot "logs") -Force | Out-Null
Set-Content -LiteralPath (Join-Path $packageRoot "data\.keep") -Value "" -Encoding ASCII -ErrorAction Stop
Set-Content -LiteralPath (Join-Path $packageRoot "logs\.keep") -Value "" -Encoding ASCII -ErrorAction Stop
Write-PackageReadme $packageRoot

$runtimeData = Join-Path $packageRoot "runtime\backend\app\data"
if (Test-Path -LiteralPath $runtimeData) {
    Remove-Item -LiteralPath $runtimeData -Recurse -Force
}

if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}
Compress-Archive -LiteralPath $packageRoot -DestinationPath $zipPath -CompressionLevel Optimal

$packageFiles = Get-ChildItem -LiteralPath $packageRoot -Recurse -Force -File
$forbidden = $packageFiles | Where-Object {
    $_.FullName -match "__pycache__|\.pyc$|providers\.json|backend\.log$|release-manifest\.json$"
}

if ($forbidden) {
    $forbiddenList = ($forbidden | Select-Object -First 20 -ExpandProperty FullName) -join "`n"
    throw "Package contains forbidden local/cache files:`n$forbiddenList"
}

$zipItem = Get-Item -LiteralPath $zipPath
$hash = Get-FileHash -LiteralPath $zipPath -Algorithm SHA256
Set-Content -LiteralPath (Join-Path $releaseVersionRoot "SHA256SUMS.txt") -Value "$($hash.Hash.ToLower())  $packageName.zip" -Encoding ASCII
Write-ReleaseNotes $releaseVersionRoot

Write-Host "[AdCust Package] Done."
Write-Host "[AdCust Package] File: $zipPath"
Write-Host "[AdCust Package] Size: $([Math]::Round($zipItem.Length / 1MB, 2)) MB"
