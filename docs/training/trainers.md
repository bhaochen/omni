# trainers/trainers.md · 训练脚本

`trainers/` 按模态分子包，每个脚本暴露 `main(default_config=None)`（通过 `python -m trainers.<mod>` 调用）。

## 文本（lm/）

| 脚本 | 任务 | 关键损失/算法 |
| --- | --- | --- |
| `pretrain.py` | 预训练 | 下一 token CE |
| `full_sft.py` | 全量 SFT | CE（loss mask 仅 assistant） |
| `lora.py` | LoRA 微调 | 低秩适配，仅训 A/B |
| `dpo.py` | DPO | 偏好对齐（参考比损失） |
| `distillation.py` | 知识蒸馏 | 师生 KL |
| `ppo.py` | PPO | Actor-Critic + 奖励 |
| `grpo.py` | GRPO | 分组相对策略优化 |
| `agent.py` | Agent RL | 工具调用强化学习 |
| `rollout_engine.py` | — | torch / sglang 推理引擎（被 ppo/grpo/agent 复用） |
| `train_tokenizer.py` | — | tokenizer 训练（学习用），结果保存到 `checkpoint/tokenizer/` |

## 视觉（vlm/）

- `pretrain.py`：视觉预训练
- `full_sft.py`：视觉 SFT（含 `vlm_collate_fn`）

## 全模态（vam/）

- `full_sft.py`：全模态 SFT（文本 + 视觉 + 音频，双 head 损失）

## VAM SFT 详解

### 模型初始化（`init_omni_model`）

```python
model, tokenizer = init_omni_model(omni_config,
    from_weight='omni-v',
    tokenizer_path='checkpoint/omni/native_hf',
    audio_encoder_path='checkpoint/sensevoice',
    vision_model_path='checkpoint/siglip',
    model_dir='checkpoint/omni-v')  # 权重源目录
```

- `from_weight` + `model_dir` 决定加载哪个 checkpoint
- 优先加载 `{model_dir}/sft_omni_{hidden_size}.pth`
- 若加载权重不含 talker 参数，自动从 thinker 后几层复制初始化
- 编码器（SenseVoice / SigLIP）从独立路径初始化，不在 checkpoint 中保存

### 训练模式（mode）

| mode | trainable params | 用途 |
| --- | --- | --- |
| `all` | 全部（113M） | 全参数 SFT |
| `audio_proj` | 仅 audio_proj（1.0M） | 音频特征对齐 |
| `vision_proj` | 仅 vision_proj（1.2M） | 视觉特征对齐 |

```python
if args.mode == 'audio_proj':
    for p in model.parameters(): p.requires_grad = False
    for p in model.audio_proj.parameters(): p.requires_grad = True
```

optimizer 使用 `filter(requires_grad)` 避免为冻结参数维护动量：

```python
optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.learning_rate
)
```

### 损失计算（双 head）

```python
# 文本损失
text_loss = CE(logits, labels, ignore_index=-100)

# 音频损失（对 8 层 Mimi code 逐层 CE，stop token 10× 加权）
audio_loss = 0
for i, al in enumerate(res.audio_logits):
    layer_loss = CE(al.view(-1, al.size(-1)), audio_labels[:, i, :].reshape(-1))
    stop_mask = (targets == audio_stop_token).float()  # 2050
    weighted = layer_loss * valid_mask * (1 + stop_mask * 9)
    audio_loss += weighted.sum() / valid_mask.sum()
audio_loss = audio_loss / 8

# 总损失（accumulation_steps 用于梯度累积）
loss = (text_loss + audio_loss + res.aux_loss) / args.accumulation_steps
```

### DataLoader 与 collate_fn

VAM 的 `omni_collate_fn` 处理**变长**的音频和视觉输入：

```python
def omni_collate_fn(batch):
    # batch 包含：input_ids, labels, audio_labels, audio_inputs, audio_lens, pixel_values, spk_emb

    # 1. 文本：直接 stack（已 padding）
    input_ids = torch.stack(input_ids)

    # 2. 音频：padding 到 batch 内最大长度
    valid_audios = [a for a in audio_inputs if a is not None]
    max_t = max(a.size(1) for a in valid_audios)
    padded = [pad(a, max_t) for a in valid_audios]
    audio_inputs = torch.cat(padded, dim=0)

    # 3. 视觉：SigLIP 返回 dict（pixel_values + attention_mask）
    valid_images = [p for p in pixel_values if p is not None]
    pixel_values = {k: torch.cat([d[k] for d in valid_images], dim=0) for k in keys}
```

### 音频处理流程

A2A 数据包含 `question_audios`（二进制音频文件），在 `__getitem__` 中按需解码：

```python
def load_audio_inputs(self, audio_bytes):
    wav, sr = sf.read(io.BytesIO(audio_bytes))
    if wav.ndim > 1: wav = wav.mean(axis=1)
    if sr != 16000:
        wav_t = torch.from_numpy(wav).unsqueeze(0)
        wav_t = AF.resample(wav_t, sr, 16000)  # torchaudio
        wav = wav_t.squeeze(0).numpy()
    inputs = self.audio_processor(wav, sampling_rate=16000, ...)
    return inputs.input_features, valid_len
```

- 使用 `soundfile` 解码音频 bytes
- `torchaudio.functional.resample` 重采样到 16kHz（SenseVoice 要求）
- `SenseVoiceAudioProcessor` 提取 fbank 特征

### 3 阶段训练配置

运行示例：

```bash
# Stage 1: T2A mode=all（从预训练初始化）
python -m trainers.vam.full_sft --config configs/vam/vam_t2a_all_mini_omni-v.yaml

# Stage 2: A2A audio_proj（从 Stage 1 初始化）
python -m trainers.vam.full_sft --config configs/vam/vam_a2a_audio_proj_mini.yaml

# Stage 3: A2A mode=all（从 Stage 2 初始化）
python -m trainers.vam.full_sft --config configs/vam/vam_a2a_all_mini.yaml
```

各配置文件的 `model_dir` 与 `from_weight` 构成训练链：
- Stage 1 → 从 `omni-v.pth` 初始化 → 输出到 `vam_t2a_all_mini_omni-v/`
- Stage 2 → 从 Stage 1 输出初始化 → 输出到 `vam_a2a_audio_proj_mini/`
- Stage 3 → 从 Stage 2 输出初始化 → 输出到 `vam_a2a_all_mini/`

### 检查点保存

```python
# 推理权重（仅 LLM 部分，fp16）
torch.save({k: v.half().cpu() for k, v in clean_state_dict.items()}, ckp)

# 续训检查点（含 optimizer + scaler 状态）
omni_checkpoint(omni_config, weight=..., model=..., optimizer=..., ...)
```

推理权重过滤掉 `audio_encoder.` 前缀（编码器需在各训练脚本中单独加载），确保 checkpoint 格式兼容。

## 通用训练循环（以 full_sft 为例）

```python
for epoch in range(epochs):
    loader = DataLoader(ds, batch_sampler=SkipBatchSampler(...))
    for step, (input_ids, labels) in enumerate(loader):
        loss = model(input_ids, labels=labels).loss + res.aux_loss
        loss = loss / accumulation_steps
        scaler.scale(loss).backward()
        if step % accumulation_steps == 0:
            clip_grad_norm_; scaler.step(optimizer); zero_grad()
        # 定期保存权重到 save_dir + 保存 optimizer/ckpt 到 checkpoint/
```

支持：分布式（`init_distributed_mode` + `DistributedDataParallel`）、混合精度（`autocast` + `GradScaler`）、梯度累积、断点续训（`from_resume`）、可选 wandb/swanlab。

## 要点（面试）

- **`SkipBatchSampler`**：分布式下跳过已训 step，配合 `from_resume` 实现精确续训。
- **`aux_loss`**：MoE 路由均衡损失，只在 `use_moe` 时非 0，需显式加到总损失。
- **RL trainer 复用 `rollout_engine`**：生成样本与训练解耦，可换 torch / sglang 后端。
- 保存分两份：`save_dir`（最终权重 `.pth`）+ `checkpoint/`（optimizer/scheduler 状态用于续训）。
- **VAM 特殊点**：双 head 损失、变长 collate_fn、audio_proj 模式只训 1% 参数、filter(requires_grad) 优化器。
