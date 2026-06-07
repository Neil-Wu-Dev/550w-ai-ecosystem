# AdCust 后端测试说明

测试分为两个标准 Python `unittest` package，不连接真实云服务器，不运行真实 GPU 训练，也不会留下 adapter 或持久化数据。

## IDE 运行方式

1. 不要运行 `run_unit_tests.py` 或 `run_integration_tests.py` 来观察绿色勾；它们是终端兼容入口。
2. 在顶部运行配置中选择 `AdCust Unit Tests` 或 `AdCust Integration Tests`，点击 Run。
3. 也可以在项目树中右键 `tests/unit` 或 `tests/integration`，选择 `Run pytest`。
4. PyCharm 的 Test Results 工具窗口会逐条显示结果；成功项目显示圆形绿色勾。

不要直接运行 `fake_server` 中的文件。它们是单元测试和集成测试使用的内存工具，不是测试入口。

## Package 职责

1. `unit`：测试 rich model、mapper、service、manifest、训练数据构造和推理消息等内部逻辑。
2. `integration`：通过 FastAPI `TestClient` 测试 API、DTO、依赖注入和响应契约。
3. `fake_server`：为上述测试提供临时 SSH/远端文件系统替身，不是独立测试 package。

## 脚本入口

- `run_unit_tests.py`：运行全部单元测试。
- `run_integration_tests.py`：运行全部集成测试。
- `run_backend_tests.py`：运行以上两类测试。

所有测试数据均写入系统临时目录，并在测试结束后自动清理。
