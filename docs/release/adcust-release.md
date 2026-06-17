# AdCust Release Structure

这个文档说明 AdCust 的专业发布目录结构。

## 正式发布目录

`release/` 只存放稳定发布物。它不存放构建脚本、源码、临时 staging 目录，也不存放本机散装运行目录。

```text
release/
  adcust/
    1.0.0/
      AdCust-1.0.0-windows-portable.zip
      SHA256SUMS.txt
      release-notes.md
```

其中：

```text
AdCust-1.0.0-windows-portable.zip
```

是 Production Binary Release，应该上传到 GitHub Releases。

## 本机构建目录

`.build/` 存放本机中间产物，整个目录不提交 Git。

```text
.build/
  adcust/
    1.0.0/
      app/
      developer-portable-release/
      package-work/
```

其中：

```text
.build/adcust/1.0.0/app/
```

是本机散装运行目录，可以在本机试运行，但不是正式发布物。

```text
.build/adcust/1.0.0/package-work/
```

是打 zip 前的临时 staging 目录，不具备发布意义。

## 构建脚本

构建脚本放在：

```text
scripts/release/adcust/
  build-adcust-release.ps1
  package-adcust-release.ps1
```

执行顺序：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\release\adcust\build-adcust-release.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\release\adcust\package-adcust-release.ps1
```

第一步生成 `.build/adcust/<version>/app`。

第二步生成 `release/adcust/<version>/AdCust-<version>-windows-portable.zip`。

## Launcher 源码

Launcher 是正式应用组件源码，放在：

```text
apps/adcust-launcher/
```

它会被构建成：

```text
AdCust Launcher.exe
```

但 exe 只进入 `.build` 和最终 zip，不提交到 Git。
