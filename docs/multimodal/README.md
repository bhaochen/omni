# multimodal/ · 多模态扩展

把「外部模态」接入 LLM 主干的三段式组件：

| 目录 | 内容 | 作用 |
| --- | --- | --- |
| `encoders/vision` | `SiglipVisionEncoder` | 图像 → patch 特征（SigLIP） |
| `encoders/audio` | `SenseVoiceAudioEncoder` | 语音 → 语义/声学特征（SenseVoice） |
| `projectors/vision` | `MMVisionProjector` | 视觉特征 → LLM `hidden_size` |
| `projectors/audio` | `MMAudioProjector` | 音频特征 → LLM `hidden_size` |
| `serve` | `SileroVAD` / `RealtimeSession` | 实时语音会话（VAD + 全双工） |

## 注入范式（回顾）

```
原始输入(图/音)
   └─ encoder      → 模态特征
        └─ projector → 投影到 hidden_size
             └─ 替换 input_ids 中的占位 token 位置的 embedding
                  └─ 进入共享 LM 主干
```

- 文本侧用特殊占位 token（`<|image_pad|>` / `<|audio_pad|>`）预留位置；
- `forward` 把投影特征写到这些位置，LLM 对文本/视觉/音频 token 一视同仁。

## Talker（语音生成，仅 VAM）

`VAM` 的 `TalkerModule` 在主干某层之后把文本隐状态解码为 8 层音频 code，实现「文本→语音」生成，与理解侧共用主干。

## 要点（面试）

- **为何 encoder 冻结 + 只训 projector？** 大模型主干易灾难性遗忘，冻结预训练视觉/音频 encoder、只训轻量投影层，是高效多模态对齐的常用做法。
- **`encoders/` 与 `projectors/` 分离**：encoder 负责「理解模态」，projector 负责「对齐到 LLM 空间」，二者职责清晰、可独立替换。
- **可扩展性**：加新模态 = 加一对 encoder+projector + 占位 token，主干不动（符合本仓库分层初衷）。
- `serve/realtime.py` 提供端到端语音对话链路（VAD 检测 → ASR → LLM → TTS/Talker）。
