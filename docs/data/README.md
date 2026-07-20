# data/ · 数据集

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

统一从 `dataset`（本仓库包）导入，例如 `from dataset import SFTDataset`。

## 共同模式

- 都继承 `torch.utils.data.Dataset`，实现 `__len__` / `__getitem__`。
- 文本类通过 `tokenizer.apply_chat_template` 渲染对话；用 `bos_id/eos_id` 在 `generate_labels` 里**只对 assistant 回复计算损失**（prompt 部分 label=-100）。
- 多模态（`vlm`/`vam`）额外加载图像/音频，并产出 `(input_ids, labels, 视觉/音频特征)` 元组。

## 训练时拼 batch

- `trainers/lm/*` 用 `SkipBatchSampler` + `DataLoader`；
- `vlm` 用 `vlm_collate_fn`（把变长视觉特征 pad/stack）；
- `vam` 的 `VAMDataset.__getitem__` 直接返回定长张量（文本+音频各层 label + 音频特征 + spk_emb）。

## 要点（面试）

- **loss mask 为什么只标 assistant？** 让模型只学习生成回复，不拟合用户输入/系统提示，避免「学用户说话」。
- 多模态数据常用 parquet + `HFDataset.from_parquet`，便于大批量流式读取。
- `pre_processing_chat` 随机插入 system prompt，`post_processing_chat` 随机去掉空 think 段，做数据增广。
