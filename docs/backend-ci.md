# AdCust 后端 CI

后端使用 GitHub Actions 免费托管的 Ubuntu runner 执行持续集成。当前项目仍处于
开发构建阶段，因此流水线不包含 Continuous Delivery 或 Continuous Deployment。

## 触发方式

以下操作会运行 `.github/workflows/backend-ci.yml`：

1. 向任意分支执行 `git push`。
2. 创建或更新 Pull Request。
3. 在 GitHub Actions 页面点击 `Run workflow` 手动运行。

同一分支短时间内连续推送时，旧的未完成任务会自动取消，只保留最新提交的 CI。

## 三个阶段

### 1. Build

1. 使用 Python 3.12。
2. 编译检查 FastAPI 后端和 `adcust_logic` 源码。
3. 构建 `adcust_logic` wheel 与 source distribution。
4. 使用 Twine 检查包元数据。
5. 检查 wheel 是否包含英文/中文 locale 和远程训练脚本。
6. 将构建产物保存为保留七天的 GitHub Actions artifact。

该阶段不会发布 package、创建 Release 或部署应用。

### 2. Unit Tests

1. 仅在 Build 成功后运行。
2. 下载并安装 Build 阶段生成的 wheel。
3. 运行 `apps/adcust-backend/tests/unit` 下的全部单元测试。
4. 不连接真实服务器，不加载真实模型，不执行 GPU 训练。

### 3. Integration Tests

1. 仅在 Unit Tests 成功后运行。
2. 下载并安装相同的 wheel。
3. 使用 FastAPI `TestClient` 和内存 fake server 运行全部后端集成测试。
4. 验证 API、DTO、依赖注入、流式训练响应及任务状态契约。

## GitHub 显示结果

一次成功运行会显示三个绿色检查：

```text
1 - Build
2 - Unit Tests
3 - Integration Tests
```

任一阶段失败时，后续阶段不会运行。进入失败 job 后可以展开具体 step 查看完整日志。

## 分支保护建议

首次将 workflow 推送到 GitHub 并成功运行后，可以在仓库分支保护规则中将以下检查
设为合并前必需：

```text
1 - Build
2 - Unit Tests
3 - Integration Tests
```

分支保护属于 GitHub 仓库设置，不能仅靠仓库内的 workflow 文件强制启用。
