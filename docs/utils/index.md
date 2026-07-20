# utils/ · 训练工具

训练与分布式通用工具，被 `trainers/` 复用。

| 文件 | 内容 | 作用 |
| --- | --- | --- |
| `training.py` | `apply_config` / `lm_checkpoint` / `init_model` / `SkipBatchSampler` / `get_lr` | YAML 配置驱动、checkpoint 读写、采样器 |
| `checkpoint.py` | `save_checkpoint` / `load_checkpoint` / `iter_module_state_dict` | 权重序列化辅助 |
| `distributed.py` | 分布式初始化辅助 | 多卡训练 |
| `multimodal.py` | `init_vlm_model` / `vlm_checkpoint` / `init_omni_model` / `omni_checkpoint` | VLM / VAM 模型装配与 checkpoint |

详见：

- [配置与命令行](config-and-cli.md) — YAML 配置如何驱动训练、`apply_config` 机制、tokenizer 训练、启动示例
