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

## 语音（Talker）模块详解

### 音频表示：Mimi Codec

语音不直接用波形，而是用 **Mimi 声码器** 把音频编码为 8 层离散 code：

```
音频波形
  └─ Mimi Encoder → 8×T 离散 token（每层 codebook 大小 2048 + 64 special tokens = 2112）
       └─ VAM talker 在每一层并行预测
```

- `audio_vocab_size=2112`：2048 个 Mimi code + 64 个特殊 token（含 `<|audio_pad|>`、`<|audio_stop|>`、`<|audio_spk|>`）
- `audio_pad_token=2049` / `audio_stop_token=2050` / `audio_spk_token=2051`
- talker 输出 8 个 `(B, T, 2112)` 的 logits 张量，每层独立计算 CE 损失
- `stop_mask` 对 `<|audio_stop|>` token 做 10× 加权，鼓励模型及时停止

### TalkerModule

在 LLM 某层（`bridge_layer`）之后接一个小型 Transformer 解码器：

```
LM 主干输出 h (B, T, hidden)
  └─ TalkerModule:
       └─ 第 1 层 cross-attn（以 h 为 query，h 为 key/value）→ 预测第 1 层 code
       └─ 第 2 层 cross-attn → 预测第 2 层 code
       └─ ...（共 num_talker_hidden_layers 层）
```

- 每层使用独立参数，**并行**预测（非自回归，类似 SoundStorm）
- 训练时 8 层 CE 损失取均值
- Talker 初始化策略：从 `thinker` 的后几层复制权重

### Thinker / Talker 分工

`VAM` 的 LM 主干有 `num_hidden_layers` 层（默认 8），分为：

| 角色 | 层范围 | 功能 |
| --- | --- | --- |
| **Thinker** | `layers[0:bridge_layer]` | 文本理解与推理 |
| **Bridge** | `bridge_layer`（默认 3） | 多模态特征注入点 |
| **Talker** | `layers[bridge_layer+1:]` | 语音生成解码 |

```python
VAM.forward(hidden_states):
  # Thinker 层
  for layer in self.thinker.layers: h = layer(h)
  # 注入多模态特征（vision_proj / audio_proj）
  h = inject_multimodal(h)
  # Talker 层
  for layer in self.talker.layers: h = layer(h)
  # 文本 head + 语音 head
  logits = self.lm_head(h)
  audio_logits = self.talker.decode(h)
```

## 关键字段（`VAMConfig`）

| 字段 | 含义 |
| --- | --- |
| `num_talker_hidden_layers` / `talker_hidden_size` | Talker 子网络规模 |
| `audio_ids` / `audio_special_token` | 音频占位符（如 `<|audio_pad|>`） |
| `audio_hidden_size` / `audio_vocab_size` | 音频特征维 / 音频词表（2048 code + 64 special） |
| `audio_pad/stop/spk_token` | 音频特殊 token id |
| `spk_emb_size` | 说话人 embedding 维（默认 192） |
| `bridge_layer` | 多模态特征注入主干的层（通常 `num_layers // 2 - 1`） |
| `image_token_len` | 单张图像占用的 token 数 |
| `use_moe` | 是否在主干 FFN 使用 MoE |

## 3 阶段 SFT 训练流程

参考 MiniMind-O 的设计，VAM 的 SFT 分为 3 个阶段逐步激活各能力：

### Stage 1：T2A（文本→音频对齐）

```
配置: mode=all, batch_size=4, lr=5e-4, max_samples=2000
数据: sft_t2a_mini.parquet（515k 条，无音频输入，无 spk_emb）
```

- 从预训练 checkpoint（`omni.pth` 或 `omni-v.pth`）初始化
- `mode=all`：所有参数参与训练（113M trainable）
- 数据只有文本（用户问题 + 文本答案 + 音频标签），`question_audios` 列为空
- 目标：让模型学会**生成声学 token**，文本+音频双损失下降
- 损失变化举例：12.79 → 9.80（500 steps）

### Stage 2：A2A audio_proj（音频特征对齐）

```
配置: mode=audio_proj, batch_size=8, lr=5e-4, max_samples=2000
数据: sft_a2a_mini.parquet（77k 条，含 question_audios + spk_emb）
```

- 从 Stage 1 输出初始化
- `mode=audio_proj`：**冻结除 `audio_proj` 外的所有参数**（仅训练 1.0M / 113M 参数）
- 数据包含真实音频输入，需要 `librosa` / `torchaudio` 重采样到 16kHz
- 目标：训练 `audio_proj` 将 SenseVoice 的音频特征映射到 LLM 隐藏空间
- 无视觉数据时 `vision_proj` 梯度为 0，不受影响

### Stage 3：A2A mode=all（全参数微调）

```
配置: mode=all, batch_size=4, lr=2e-5, max_samples=2000
数据: sft_a2a_mini.parquet
```

- 从 Stage 2 输出初始化
- 恢复全参数训练（113M trainable），但**学习率降低至 2e-5**（Stage 1 的 1/25）
- 目标：在已对齐的音频特征基础上，精细调优全部参数
- 损失变化举例：9.72 → 9.27（500 steps）

### 为什么分 3 阶段？

| 阶段 | 解决的问题 | 训练参数 | LR |
| --- | --- | --- | --- |
| 1: T2A mode=all | 冷启动：从头学音频 code 生成 | 全部 | 5e-4 |
| 2: A2A audio_proj | 对齐：让音频特征进入 LLM 空间 | 仅 proj (1%) | 5e-4 |
| 3: A2A mode=all | 精调：全参数协同优化 | 全部 | 2e-5 |

- 若跳过 Stage 1 直接 A2A，模型未见过音频 code 分布，生成质量差
- 若跳过 Stage 2 直接全参数，`audio_proj` 远未收敛，梯度方向主次不分
- Stage 2 用高 LR 只训 proj，是**多模态对齐的标准做法**

## 训练要点

### 损失函数

训练总损失 = 文本 CE + 音频 CE + aux_loss（仅 MoE 时非 0）：

```python
# 文本损失（与 LM 一致）
text_loss = CE(logits, labels, ignore_index=-100)

# 音频损失（每层独立 CE，对 stop token 加权）
audio_loss = 0
for i, al in enumerate(audio_logits):  # 8 层
    layer_loss = CE(al.view(-1, 2112), targets[:, i, :].reshape(-1))
    stop_mask = (targets == audio_stop_token).float()  # 2050
    weighted = layer_loss * valid_mask * (1 + stop_mask * 9)
    audio_loss += weighted.sum() / valid_mask.sum()
audio_loss = audio_loss / 8  # 8 层均值
```

### 优化器与梯度

```python
optimizer = AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),  # 只训 trainable
    lr=learning_rate
)
```

- `mode=audio_proj` 时，`model.audio_proj` 约 1.0M 参数，其余 112M 冻结
- `filter(requires_grad)` 避免 optimizer 持有冻结参数的动量（省显存）

### 数据集

| 数据 | 格式 | 行数 | 大小 | 特点 |
| --- | --- | --- | --- | --- |
| T2A | parquet | 515k | 1.5 GB | 无 question_audios，无 spk_emb |
| A2A | parquet | 77k | 841 MB | 含 question_audios + spk_emb + ref_audios |

- 使用 `pyarrow.parquet.iter_batches(batch_size=4096)` 流式读取
- `max_samples` 控制加载行数，适合快速验证
- 音频列（`question_audios`）为二进制 bytes，在 `__getitem__` 时解码

### 说话人嵌入（spk_emb）

- A2A 数据包含预计算的 `spk_emb` 维（192 维，campplus 模型提取）
- T2A 数据无此列 → 回退到 `torch.zeros(192)`
- `spk_emb` 在 forward 中与 hidden states 拼接，协助模型区分说话人

## 检查点管理

训练过程保存两种检查点：

```
save_dir/sft_omni_768.pth           # 推理权重（仅 LLM 部分，fp16）
../checkpoints/sft_omni_768.pth     # 完整检查点（含 optimizer state 用于续训）
../checkpoints/sft_omni_768_resume.pth  # 带 optimizer 的续训文件
```

推理权重过滤掉 `audio_encoder.` 和 `vision_encoder.` 前缀（编码器需单独加载）。

## 要点（面试）

- **统一主干 + 多 head**：理解用共享 `LM` 主干；感知侧挂 encoder+projector，生成侧挂 Talker。新增模态 = 新增 encoder/projector，不改主干。
- **音频 code 多层级**：语音用 8 层离散 code 表示，模型并行预测每一层，贴近 SoundStorm/Mimi 思路。
- **占位 + 投影**范式与 VLM 一致，保证跨模态位置对齐。
- **3 阶段 SFT**：T2A→audio_proj→full，逐步激活文本生成、音频理解、协同优化能力。
- `bridge_layer` 控制多模态信息「插入」主干的深度，是平衡早/晚融合的超参。
