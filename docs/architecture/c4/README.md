# AdCust C4 架构图

本目录包含基于当前代码结构绘制的三级 C4 图，图内文字全部使用英文：

1. `adcust-c1-system-context.svg` / `.png`：系统上下文，以及用户、AdCust、RunPod GPU Server、Hugging Face Hub 的关系。
2. `adcust-c2-container.svg` / `.png`：Tauri 桌面端、FastAPI 后端、`adcust_logic` package、本地推理与存储、RunPod 远程训练容器。
3. `adcust-c3-component.svg` / `.png`：后端 API 与 headless package 内部的服务、rich models、DTO、引擎、SSH driver 和异常边界。
4. `adcust-runpod-user-workflow.svg` / `.png`：从 RunPod 创建 Pod、AdCust 注册 Node、训练、手动停机，到端口更新和 Template 恢复的完整用户工作流。
5. `adcust-runpod-user-workflow.md`：与工作流图对应的逐步操作表。

图中明确区分了两个事实：

1. RunPod 是当前实际使用的远程 GPU 服务示例。
2. AdCust 只通过 SSH/SFTP 使用服务器，不负责创建、启动、停止、销毁 Pod，也不控制云厂商计费。
