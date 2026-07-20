# training/trainers.md · 训练脚本

`trainers/` 按模态分子包，每个脚本暴露 `main(default_config=None)`（可由 `python -m trainers.<mod>` 或 `runs/*.sh` 调用）。

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
- 保存分两份：`save_dir`（最终权重 `.pth`）+ `checkpoint/`（optimizer/scheduler 状态用于续训）。`train_tokenizer.py` 额外把训练出的 tokenizer 写到 `checkpoint/tokenizer/`。
