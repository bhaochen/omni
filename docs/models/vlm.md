# models/vlm · 视觉多模态 (VLM)

`VLM` 继承自 `LMForCausalLM`，复用 `LM` 主干，只在 `forward` 里把**视觉特征投影后拼到文本 embedding 序列**对应位置。

## 结构

```
VLM(LMForCausalLM)
  ├─ model: LM                       # 复用主干
  ├─ vision_encoder: SiglipVisionEncoder
  └─ vision_proj: MMVisionProjector  # 图像特征 → LLM hidden
```

## 视觉注入方式（面试重点）

- 文本序列里用特殊 token（如 `<|image_pad|>` × `image_token_len`）占位。
- 图像经 `vision_encoder`(SigLIP) 得到 patch 特征 → `vision_proj` 映射到 `hidden_size` 与目标 token 数。
- `forward` 把投影后的图像特征**替换**掉占位 token 位置的 embedding，再走标准 `LM` 主干。
- 因此 LLM 看到的「图像」就是一段连续的视觉 token，与文本 token 一视同仁。

## 关键字段（`VLMConfig`）

| 字段 | 含义 |
| --- | --- |
| `image_special_token` | 图像占位符，如 `<|image_pad|>` |
| `image_ids` | 占位 token 的 id（如 `[12]`） |
| `image_hidden_size` | 视觉编码器输出维度 |
| `image_token_len` | 单张图压缩成的视觉 token 数 |

## 要点（面试）

- **为什么投影 + 占位而不是 concat 新模态？** 复用现成 LLM 的词嵌入位置，训练稳定，且位置编码天然连续。
- **冻结策略**：训练时常冻结 `vision_encoder`，只训练 `vision_proj` 与 LLM 部分（`freeze_llm` 参数控制），避免灾难性遗忘。
- 视觉 token 数与图像分辨率/分块数解耦，由 `vision_proj` 重采样到固定 `image_token_len`。
