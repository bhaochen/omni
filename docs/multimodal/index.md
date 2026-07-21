# multimodal/ · 多模态

把「外部模态」（图像/音频）编码并投影到 LLM 隐藏空间，实现多模态理解。

## 数据流

```
原始输入(图/音)
   └─ encoder      → 模态特征
        └─ projector → 投影到 hidden_size
             └─ 替换 input_ids 中的占位 token 位置的 embedding
                  └─ 进入共享 LM 主干
```

## 编码器（Encoders）

把外部模态的原始信号编码为模态特征，冻结预训练权重、不参与 LLM 主干训练。

| 文件 | 类 | 作用 |
| --- | --- | --- |
| `encoders/vision/siglip.py` | `SiglipVisionEncoder` | 图像 → patch 特征（SigLIP） |
| `encoders/audio/sensevoice.py` | `SenseVoiceAudioEncoder` | 语音 → 语义/声学特征（SenseVoice） |

## 投影器（Projectors）

把 encoder 输出的模态特征投影到 LLM 隐藏维度（`hidden_size`），随后替换 `input_ids` 中的占位 token 位置。

| 文件 | 类 | 作用 |
| --- | --- | --- |
| `projectors/vision.py` | `MMVisionProjector` | 视觉特征 → LLM `hidden_size` |
| `projectors/audio.py` | `MMAudioProjector` | 音频特征 → LLM `hidden_size` |

## 设计要点（面试）

- **encoder 冻结 + 只训 projector**：大模型主干易灾难性遗忘，冻结预训练视觉/音频 encoder、只训轻量投影层，是高效多模态对齐的常用做法。
- **`encoders/` 与 `projectors/` 分离**：encoder 负责「理解模态」，projector 负责「对齐到 LLM 空间」，二者职责清晰、可独立替换。
- **可扩展性**：加新模态 = 加一对 encoder+projector + 占位 token，主干不动（符合本仓库分层初衷）。
- 文本侧用特殊占位 token（`<|vision_start|>` / `會員註冊`）预留位置；
- `forward` 把投影特征写到这些位置，LLM 对文本/视觉/音频 token 一视同仁。
