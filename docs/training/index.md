# training/ · 训练

训练相关的数据集、训练脚本、配置工具。

## 数据集（Dataset）

`dataset/` 下**每类数据集一个文件**（已去掉 `_dataset` 后缀），公共辅助函数在 `common.py`。

| 文件 | 数据集 | 用途 |
| --- | --- | --- |
| `pretrain.py` | `PretrainDataset` | 预训练（纯文本） |
| `sft.py` | `SFTDataset` | 全量 SFT（chat 模板 + loss mask） |
| `dpo.py` | `DPODataset` | DPO 偏好数据 |
| `rlaif.py` | `RLAIFDataset` | RLHF/RLAIF prompt 采样 |
| `agent_rl.py` | `AgentRLDataset` | Agent 强化学习轨迹 |
| `vlm.py` | `VLMDataset` | 图文对（parquet） |
| `vam.py` | `VAMDataset` | 全模态（图文 + 音频，parquet） |

### 共同模式

- 都继承 `torch.utils.data.Dataset`，实现 `__len__` / `__getitem__`。
- 文本类通过 `tokenizer.apply_chat_template` 渲染对话；用 `bos_id/eos_id` 在 `generate_labels` 里**只对 assistant 回复计算损失**（prompt 部分 label=-100）。
- 多模态（`vlm`/`vam`）额外加载图像/音频，并产出 `(input_ids, labels, 视觉/音频特征)` 元组。

## 训练脚本（Trainers）

按模态组织的训练入口，每个脚本暴露 `main(default_config=None)`（可由 `python -m trainers.<mod>` 或 `runs/*.sh` 调用）。

- `trainers/lm/`：pretrain / full_sft / lora / dpo / distillation / ppo / grpo / agent / rollout_engine / train_tokenizer
- `trainers/vlm/`：pretrain / full_sft
- `trainers/vam/`：full_sft

详见：[trainers.md](trainers.md) — 各 trainer 模块概览与通用训练循环

## 工具（Utils）

### 配置与命令行

YAML 配置驱动训练、`apply_config` 机制、tokenizer 训练、启动示例。

详见：[config-and-cli.md](config-and-cli.md)

## 面试要点

- **loss mask 为什么只标 assistant？** 让模型只学习生成回复，不拟合用户输入/系统提示，避免「学用户说话」。
- 多模态数据常用 parquet + `HFDataset.from_parquet`，便于大批量流式读取。
- `pre_processing_chat` 随机插入 system prompt，`post_processing_chat` 随机去掉空 think 段，做数据增广。
- **配置系统优先级**：`CLI 参数 > YAML 默认值 > 代码默认`；`apply_config` 把 YAML 扁平化注入 argparse 默认值。
- **续训怎么保证精确**：`SkipBatchSampler` 在分布式下跳过已训 step；`from_resume` 从 `checkpoint/` 恢复 optimizer/scheduler 状态。
