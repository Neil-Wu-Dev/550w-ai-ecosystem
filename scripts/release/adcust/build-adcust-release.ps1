# 构建 AdCust 本机中间产物。
# 注意：这个脚本不直接生成正式发布目录；正式发布包由 package-adcust-release.ps1 生成。
$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $scriptDir "..\..\..")
$version = if ($env:ADCUST_RELEASE_VERSION) { $env:ADCUST_RELEASE_VERSION } else { "1.0.0" }
$buildRoot = Join-Path $repoRoot ".build\adcust\$version"
$appBuildRoot = Join-Path $buildRoot "app"
$developerBuildRoot = Join-Path $buildRoot "developer-portable-release"
$desktopRoot = Join-Path $repoRoot "apps\adcust-desktop"
$backendRoot = Join-Path $repoRoot "apps\adcust-backend"
$logicRoot = Join-Path $repoRoot "packages\adcust-logic"
$launcherRoot = Join-Path $repoRoot "apps\adcust-launcher"
$pythonExe = if ($env:ADCUST_RELEASE_PYTHON) { $env:ADCUST_RELEASE_PYTHON } else { "D:\runtime\miniconda3\envs\heavy-common-env\python.exe" }

function Assert-Command($commandName, $message) {
    if (-not (Get-Command $commandName -ErrorAction SilentlyContinue)) {
        throw $message
    }
}

function Invoke-Checked($commandName, [string[]]$arguments) {
    & $commandName @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$commandName failed with exit code $LASTEXITCODE."
    }
}

function Reset-Directory($path) {
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Recurse -Force
    }
    New-Item -ItemType Directory -Path $path -Force | Out-Null
}

function Copy-Runtime($targetRoot) {
    $runtimeRoot = Join-Path $targetRoot "runtime"
    Reset-Directory $runtimeRoot
    New-Item -ItemType Directory -Path (Join-Path $runtimeRoot "backend") -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $runtimeRoot "adcust-logic") -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $backendRoot "app") -Destination (Join-Path $runtimeRoot "backend\app") -Recurse -Force
    Copy-Item -LiteralPath (Join-Path $logicRoot "adcust_logic") -Destination (Join-Path $runtimeRoot "adcust-logic\adcust_logic") -Recurse -Force
}

function Prepare-AppDirectory($targetRoot) {
    Reset-Directory $targetRoot
    New-Item -ItemType Directory -Path (Join-Path $targetRoot "logs") -Force | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $targetRoot "data") -Force | Out-Null
    Copy-Runtime $targetRoot

    # 本机中间产物可以复制你的本地算力配置，方便在本机直接试运行。
    $existingProviderFile = Join-Path $backendRoot "app\data\providers.json"
    $targetProviderFile = Join-Path $targetRoot "data\providers.json"
    if (Test-Path -LiteralPath $existingProviderFile) {
        Copy-Item -LiteralPath $existingProviderFile -Destination $targetProviderFile -Force
        Write-Host "[AdCust Build] Existing compute node configuration copied into $targetProviderFile."
    }
}

function Write-DeveloperStartScript($targetRoot) {
    $startScript = @"
@echo off
setlocal
cd /d "%~dp0"

set "ADCUST_RELEASE_ROOT=%~dp0"
set "ADCUST_BACKEND_HOST=127.0.0.1"
set "ADCUST_BACKEND_PORT=8000"
set "ADCUST_DATA_DIR=%~dp0data"
set "PYTHONPATH=%~dp0runtime\backend;%~dp0runtime\adcust-logic"
set "ADCUST_PYTHON=$pythonExe"

if not exist "%ADCUST_PYTHON%" (
  echo [AdCust] Python executable not found: %ADCUST_PYTHON%
  echo Rebuild with ADCUST_RELEASE_PYTHON if your Python path is different.
  pause
  exit /b 1
)

powershell -NoProfile -ExecutionPolicy Bypass -Command "try { Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 1 ^| Out-Null; exit 0 } catch { exit 1 }"
if errorlevel 1 (
  echo [AdCust] Starting backend on http://127.0.0.1:8000 ...
  start "AdCust Backend" /min "%ADCUST_PYTHON%" -m uvicorn app.main:app --app-dir "%~dp0runtime\backend" --host 127.0.0.1 --port 8000

  powershell -NoProfile -ExecutionPolicy Bypass -Command "for (`$i=0; `$i -lt 40; `$i++) { try { Invoke-RestMethod -Uri 'http://127.0.0.1:8000/health' -TimeoutSec 1 ^| Out-Null; exit 0 } catch { Start-Sleep -Milliseconds 500 } }; exit 1"
  if errorlevel 1 (
    echo [AdCust] Backend did not become ready. Check logs or run the backend manually.
    pause
    exit /b 1
  )
) else (
  echo [AdCust] Backend is already ready on http://127.0.0.1:8000.
)

echo [AdCust] Launching desktop app...
start "" "%~dp0AdCust.exe"
exit /b 0
"@

    Set-Content -LiteralPath (Join-Path $targetRoot "Start-AdCust.cmd") -Value $startScript -Encoding ASCII
}

function Write-Manifest($targetRoot, $releaseKind, $entryPoint) {
    $manifest = @{
        product = "AdCust"
        version = $version
        release_kind = $releaseKind
        generated_at = (Get-Date).ToString("o")
        backend_url = "http://127.0.0.1:8000"
        python = $pythonExe
        desktop_exe = "AdCust.exe"
        entry_point = $entryPoint
        data_dir = "data"
        runtime_dir = "runtime"
    }
    $manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $targetRoot "release-manifest.json") -Encoding UTF8
}

Write-Host "[AdCust Build] Repository: $repoRoot"
Write-Host "[AdCust Build] Build root: $buildRoot"

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Python executable not found: $pythonExe. Set ADCUST_RELEASE_PYTHON to override it."
}

Assert-Command "pnpm.cmd" "pnpm.cmd is required to build the Tauri frontend."
Assert-Command "cargo" "Rust cargo is required to build the Tauri app and launcher."

& $pythonExe -c "import fastapi, uvicorn, pydantic; print('Backend dependency check: PASS')"

Write-Host "[AdCust Build] Building Tauri desktop app..."
Push-Location $desktopRoot
try {
    $env:CI = "true"
    $tauriCli = Join-Path $desktopRoot "node_modules\.bin\tauri.cmd"
    if ((Test-Path -LiteralPath $tauriCli) -and -not ($env:ADCUST_FORCE_INSTALL -eq "1")) {
        Write-Host "[AdCust Build] Frontend dependencies already exist; skipping pnpm install."
    }
    else {
        Write-Host "[AdCust Build] Installing frontend dependencies..."
        Invoke-Checked "pnpm.cmd" @("install", "--frozen-lockfile")
    }
    Invoke-Checked "pnpm.cmd" @("run", "tauri", "build", "--no-bundle")
}
finally {
    Pop-Location
}

$tauriExe = Join-Path $desktopRoot "src-tauri\target\release\adcust-desktop.exe"
if (-not (Test-Path -LiteralPath $tauriExe)) {
    throw "Tauri release executable was not found: $tauriExe"
}

Write-Host "[AdCust Build] Building AdCust Launcher..."
Invoke-Checked "cargo" @("build", "--release", "--manifest-path", (Join-Path $launcherRoot "Cargo.toml"))
$launcherExe = Join-Path $launcherRoot "target\release\adcust_launcher.exe"
if (-not (Test-Path -LiteralPath $launcherExe)) {
    throw "Launcher executable was not found: $launcherExe"
}

Write-Host "[AdCust Build] Preparing app build..."
Prepare-AppDirectory $appBuildRoot
Copy-Item -LiteralPath $tauriExe -Destination (Join-Path $appBuildRoot "AdCust.exe") -Force
Copy-Item -LiteralPath $launcherExe -Destination (Join-Path $appBuildRoot "AdCust Launcher.exe") -Force
Write-Manifest $appBuildRoot "local-app-build" "AdCust Launcher.exe"

Write-Host "[AdCust Build] Preparing developer portable build..."
Prepare-AppDirectory $developerBuildRoot
Copy-Item -LiteralPath $tauriExe -Destination (Join-Path $developerBuildRoot "AdCust.exe") -Force
Write-DeveloperStartScript $developerBuildRoot
Write-Manifest $developerBuildRoot "developer-portable-build" "Start-AdCust.cmd"

Write-Host "[AdCust Build] Done."
Write-Host "[AdCust Build] App build: $appBuildRoot\AdCust Launcher.exe"
Write-Host "[AdCust Build] Developer portable build: $developerBuildRoot\Start-AdCust.cmd"
