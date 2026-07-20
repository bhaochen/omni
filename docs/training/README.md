# training/ · 训练系统

训练相关代码分散在两处：

- `utils/training.py`：训练通用工具（配置加载、checkpoint、优化器辅助、采样器）。
- `trainers/`：按模态组织的训练脚本（`lm/` `vlm/` `vam/`）。

详见：

- [config-and-cli.md](config-and-cli.md) — YAML 配置如何驱动训练
- [trainers.md](trainers.md) — 各 trainer 模块概览
