# models/vam · 全模态 (VAM)

`VAM`（原 MiniMindOmni）在 `VLM` 基础上再接入**语音**：既能理解音频/图像输入，也能**生成语音**（Talker 模块）。同样继承 `LMForCausalLM`，复用 `LM` 主干。

## 结构

```
VAM(LMForCausalLM)
  ├─ model: LM                                  # 复用主干
  ├─ vision_encoder + vision_proj               # 同 VLM（图像理解）
  ├─ audio_encoder (SenseVoice) + audio_proj    # 语音理解
  └─ talker: TalkerModule                       # 语音生成（文本→音频 code）
```

## 双输出（理解 + 生成语音）

`VAM.forward` 同时产出：

- `logits`：文本 token 分布（与 LM 一致）；
- `audio_logits`：语音 code 分布（8 层 Mimi 音频 code，维度 `audio_vocab_size=2112`）。

即模型**并行预测文本和语音**，训练时两份交叉熵分别监督。

## 关键字段（`VAMConfig`）

| 字段 | 含义 |
| --- | --- |
| `num_talker_hidden_layers` / `talker_hidden_size` | Talker 子网络规模 |
| `audio_ids` / `audio_special_token` | 音频占位符（如 `<|audio_pad|>`） |
| `audio_hidden_size` / `audio_vocab_size` | 音频特征维 / 音频词表（2048 code + 64 special） |
| `audio_pad/stop/spk_token` | 音频特殊 token id |
| `spk_emb_size` | 说话人 embedding 维 |
| `bridge_layer` | 多模态特征注入主干的层（通常 `num_layers//2 - 1`） |

## TalkerModule

- 在 LLM 某层（`bridge_layer`）之后接一个小型 Transformer，**把文本隐状态解码成 8 层音频 code**（Mimi 风格声码器前置表示）。
- 训练目标：文本与音频 code 的对齐序列。

## 要点（面试）

- **统一主干 + 多 head**：理解用共享 `LM` 主干；感知侧挂 encoder+projector，生成侧挂 Talker。新增模态 = 新增 encoder/projector，不改主干。
- **音频 code 多层级**：语音用 8 层离散 code 表示，模型并行预测每一层，贴近 SoundStorm/Mimi 思路。
- **占位 + 投影**范式与 VLM 一致，保证跨模态位置对齐。
- `bridge_layer` 控制多模态信息「插入」主干的深度，是平衡早/晚融合的超参。
