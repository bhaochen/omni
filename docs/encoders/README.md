# encoders/ · 多模态编码器

把「外部模态」的原始信号编码为模态特征，冻结预训练权重、不参与 LLM 主干训练。

| 文件 | 类 | 作用 |
| --- | --- | --- |
| `encoders/vision/siglip.py` | `SiglipVisionEncoder` | 图像 → patch 特征（SigLIP） |
| `encoders/audio/sensevoice.py` | `SenseVoiceAudioEncoder` | 语音 → 语义/声学特征（SenseVoice） |

## 设计要点（面试）

- **encoder 冻结 + 只训 projector**：大模型主干易灾难性遗忘，冻结预训练视觉/音频 encoder、只训轻量投影层，是高效多模态对齐的常用做法。
- **`encoders/` 与 `projectors/` 分离**：encoder 负责「理解模态」，projector 负责「对齐到 LLM 空间」，二者职责清晰、可独立替换。
- **可扩展性**：加新模态 = 加一对 encoder+projector + 占位 token，主干不动（符合本仓库分层初衷）。
