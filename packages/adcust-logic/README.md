# adcust_logic

`adcust_logic` 是 AdCust 的可复用 headless Python package，包含：

1. Rich domain models 与 DTO schemas。
2. 数据集、模型、训练、推理和算力节点服务。
3. Provider-agnostic SSH/SFTP 远程执行能力。
4. LoRA/QLoRA 远程训练脚本与 Adapter 签名规则。
5. 本地 Hugging Face Transformers + PEFT 推理能力。

该 package 不负责创建、启动、停止或销毁云厂商实例。远程算力只需满足可通过
SSH 访问的 Linux + NVIDIA GPU 主机契约。
