# trainers/ · 训练脚本

按模态组织的训练入口，每个脚本暴露 `main(default_config=None)`（可由 `python -m trainers.<mod>` 或 `runs/*.sh` 调用）。

- `trainers/lm/`：pretrain / full_sft / lora / dpo / distillation / ppo / grpo / agent / rollout_engine / train_tokenizer
- `trainers/vlm/`：pretrain / full_sft
- `trainers/vam/`：full_sft

相关：配置与 CLI 机制见 [../utils/config-and-cli.md](../utils/config-and-cli.md)（基于 `utils/training.apply_config`）。

详见：

- [trainers.md](trainers.md) — 各 trainer 模块概览与通用训练循环
