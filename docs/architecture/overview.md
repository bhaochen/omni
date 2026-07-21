# 01 · 架构总览

## 1. 设计目标

把「纯文本 / 视觉多模态 / 全模态（语音）」三套模型统一到一套代码里，按**能力分层**：

- `core/` 只放**与模态无关的纯 Transformer 组件**，可被任意模型复用。
- `models/` 把 `core` 组件**拼装**成成品模型，按模态分 `lm` / `vlm` / `vam` 三个子包。
- `encoders/` + `projectors/` 负责把外部模态信号接入 LLM 主干。
- `trainers/` 按模态组织训练脚本；`dataset/` 按数据格式组织数据集。

## 2. 模型能力矩阵

| 子包 | 模态 | 主干 | 额外组件 |
| --- | --- | --- | --- |
| `models/lm` | 文本 | `LM`（`LMForCausalLM`） | — |
| `models/vlm` | 文本 + 图像 | `VLM`(继承 `LMForCausalLM`) | `SiglipVisionEncoder` + `MMVisionProjector` |
| `models/vam` | 文本 + 图像 + 语音 | `VAM`(继承 `LMForCausalLM`) | 上述 + `SenseVoice` + `MMAudioProjector` + `TalkerModule` |

继承关系：`VLM` 和 `VAM` 都继承自 `LMForCausalLM`，因此**共享 `LM` 主干 + `lm_head` + `generate()`**，只在 `forward` 里额外拼接视觉/音频特征。

## 3. 一次前向的数据流（以 LM 为例）

```
input_ids
  └─ embed_tokens ──► dropout
        │
        ├─ (RoPE 频率 precompute 一次，缓存为 buffer freqs_cos/freqs_sin)
        │
        └─ N × Block:
              residual = x
              x = x + Attention(RMSNorm(x), rope, mask)   # 带 QK-Norm + GQA
              x = x + MLP(RMSNorm(x))                      # SwiGLU 或 MoE
        │
        └─ RMSNorm(x)
              └─ lm_head ──► logits
```

损失 = `CrossEntropy(logits[:, :-1], labels[:, 1:], ignore_index=-100)` + `aux_loss`(仅 MoE)。

## 4. 训练入口

```
python -m trainers.lm.full_sft   --config configs/lm/lm_full_sft.yaml
python -m trainers.vlm.full_sft  --config configs/vlm/vlm_sft.yaml
python -m trainers.vam.full_sft  --config configs/vam/vam.yaml
```

配置读取：`utils.training.apply_config(parser, default_config)` 把 YAML 的
`model/train/paths` 三段扁平化后作为 argparse 默认值；**CLI 显式参数覆盖 YAML**。

## 5. 关键设计取舍（面试可聊）

1. **Pre-Norm + 残差**：`Block` 用 RMSNorm 包裹子层，残差直接相加（标准 LLaMA 风格）。
2. **QK-Norm**：attention 里对 q/k 每个 head 做 `RMSNorm(head_dim)`，稳定训练（Qwen 风格）。
3. **GQA**：`num_key_value_heads` 可小于 `num_attention_heads`，k/v 通过 `repeat_kv` 复制。
4. **权重绑定**：`tie_word_embeddings=True` 时 `lm_head.weight` 与 `embed_tokens.weight` 共享。
5. **MoE 可插拔**：`Block` 根据 `config.use_moe` 在 `FeedForward` / `MOEFeedForward` 间切换；
   `aux_loss` 在 `MLP.forward` 内累加，主模型 `forward` 汇总。
6. **配置驱动**：所有超参集中在 `LMConfig`（及其子类），trainer 用 `**vars(args)` 构造，
   所以 YAML 能完整控制模型结构。
