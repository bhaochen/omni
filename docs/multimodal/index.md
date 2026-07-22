# multimodal/ · 多模态

把「外部模态」（图像/音频）编码并投影到 LLM 隐藏空间，实现多模态理解与生成。

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
| `encoders/audio/sensevoice.py` | `SenseVoiceAudioEncoder` | 语音 → 帧级别 fbank 特征（SenseVoice） |

### SigLIP 视觉编码器

- 基于 `transformers.SiglipVisionModel`（siglip2-base-p32-256）
- 图像被分割为 32×32 像素的 patch，输出 patch 特征序列
- `processor` 负责图像预处理（resize、normalize）
- 输出通过 `MMVisionProjector` 重采样到 `image_token_len=64` 个视觉 token

### SenseVoice 音频编码器

- 基于 FunASR 的 `SenseVoiceEncoderSmall`（阿里达摩院）
- 输入：16kHz 单声道音频 → fbank 特征
- 输出：**帧级别**声学特征（每帧对应约 10ms 音频）
- 编码器输出维度：`d_model=512`（`audio_hidden_size`）
- 内部包含 frontend（fbank 提取）+ encoder（Conformer 架构）
- 使用 `funasr.AutoModel` 加载，`trust_remote_code=True`

> 注意：FunASR 的 C 库初始化与 PyArrow 存在冲突，需确保 pyarrow 在 FunASR 模型创建后导入（VAMDataset 使用 lazy import）。

### 音频特征长度

音频经 SenseVoice 编码后的帧数取决于输入长度。由于帧数与文本 token 数不一致，需用 `audio_proj` 重采样到固定长度匹配占位 token：

```
音频 3 秒 @ 16kHz = 48000 采样点
  → fbank (80 维) × ~300 帧
  → audio_proj 重采样到目标帧数（占位 token 数）
  → 替换 input_ids 中的 <|audio_pad|> 位置
```

## 投影器（Projectors）

把 encoder 输出的模态特征投影到 LLM 隐藏维度（`hidden_size`），随后替换 `input_ids` 中的占位 token 位置。

| 文件 | 类 | 作用 |
| --- | --- | --- |
| `projectors/vision.py` | `MMVisionProjector` | 视觉特征 → LLM `hidden_size` |
| `projectors/audio.py` | `MMAudioProjector` | 音频特征 → LLM `hidden_size` |

### MMAudioProjector

```
输入: (B, T_audio, audio_hidden_size=512)
  └─ Linear(512 → hidden_size)
  └─ 2D 位置编码（补偿音频帧与文本 token 的位置差异）
  └─ 重采样到目标 token 数
输出: (B, target_tokens, hidden_size)
```

## VAM 双模态架构

VAM 同时处理视觉和音频：

```
VAM(LMForCausalLM)
  ├─ model: LM                    # 共享主干
  ├─ vision_encoder + proj        # 图像理解
  ├─ audio_encoder + proj         # 语音理解
  └─ talker: TalkerModule          # 语音生成
```

### 多模态特征注入

在 `bridge_layer`（默认第 3 层）之后注入多模态特征：

```python
# Forward 中的多模态注入
hidden_states = self.model.embed_tokens(input_ids)

# Thinker 层
h = self.thinker(hidden_states)

# 注入视觉特征
if pixel_values is not None:
    vision_features = self.vision_proj(self.vision_encoder(pixel_values))
    h = inject_at_positions(h, vision_features, vision_positions)

# 注入音频特征
if audio_inputs is not None:
    audio_features = self.audio_proj(self.audio_encoder(audio_inputs))
    h = inject_at_positions(h, audio_features, audio_positions)

# Talker 层
h = self.talker(h)

# 双 head 输出
logits = self.lm_head(h)
audio_logits = self.talker.decode(h)  # 8 层 Mimi code
```

### 模态组合

VAM 支持灵活的模态输入组合：

| 输入模态 | 设置 | 用例 |
| --- | --- | --- |
| 文本 + 音频 | 提供 `audio_inputs`，`pixel_values=None` | 语音对话 |
| 文本 + 图像 | 提供 `pixel_values`，`audio_inputs=None` | 图像理解 |
| 文本 + 图像 + 音频 | 同时提供两者 | 全模态交互 |
| 纯文本 | 两者均为 None | 回退到 LM |

## 设计要点（面试）

- **encoder 冻结 + 只训 projector**：大模型主干易灾难性遗忘，冻结预训练视觉/音频 encoder、只训轻量投影层，是高效多模态对齐的常用做法。
- **`encoders/` 与 `projectors/` 分离**：encoder 负责「理解模态」，projector 负责「对齐到 LLM 空间」，二者职责清晰、可独立替换。
- **可扩展性**：加新模态 = 加一对 encoder+projector + 占位 token，主干不动（符合本仓库分层初衷）。
- 文本侧用特殊占位 token（`<|image_pad|>` / `<|audio_pad|>`）预留位置；
- `forward` 把投影特征写到这些位置，LLM 对文本/视觉/音频 token 一视同仁。
- **3 阶段 SFT**：T2A→audio_proj→full，逐步激活文本生成、音频理解、协同优化能力。
