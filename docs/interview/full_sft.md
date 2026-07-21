# 面试：全量微调（Full SFT）深度

> 本仓库 `src/trainers/lm/full_sft.py`，对应 config：`configs/lm/lm_full_sft.yaml`

## 0. 整体流程

```
数据集 (sft_t2t_mini.jsonl)
    │  60,000 条多轮对话
    ▼
SFTDataset
    ├── 解析 JSON 对话格式
    ├── 拼接多轮对话（含 <s>system</s> <s>user</s> <s>assistant</s> 标记）
    ├── input_ids：全部 token 的 id
    ├── labels：prompt 段 → -100，assistant 段 → input_ids
    └── 截断 / 填充到 max_seq_len
    │
    ▼
DataLoader → model(input_ids, labels)
    │
    ├── 🔥 加载预训练权重（from_weight=pretrain）
    │    └── model_dir 指定权重源目录
    │
    ├── 全量参数更新（所有参数参与训练）
    │
    └── Loss = CE(shift_logits, shift_labels)
        只计算 labels != -100 的位置
    │
    ▼
    AdamW + Cosine LR → 保存 checkpoint
```

---

## Q1. SFTDataset 如何处理多轮对话？

### 输入格式

```json
{
  "conversations": [
    {"from": "system", "value": "你是一个 AI 助手"},
    {"from": "user",  "value": "你好"},
    {"from": "assistant", "value": "你好！有什么可以帮你的？"},
    {"from": "user",  "value": "什么是 LLM？"},
    {"from": "assistant", "value": "LLM 是大型语言模型..."}
  ]
}
```

### 编码拼接（`src/dataset/sft.py`）

```python
def _build_conversation(self, conversation):
    input_ids, labels = [], []
    for turn in conversation:
        speaker = turn['from']
        text = turn['value']
        # 用特殊标记包裹每条发言
        tokens = self.tokenizer.encode(f'<s>{speaker}</s>\n{text}\n')
        if speaker == 'assistant':
            input_ids += tokens
            labels   += tokens                     # 回复段参与损失
        else:
            input_ids += tokens
            labels   += [-100] * len(tokens)       # prompt 段忽略损失
    return input_ids[:self.max_length], labels[:self.max_length]
```

### 为什么 prompt 段 label 置 -100？

- `nn.CrossEntropyLoss(ignore_index=-100)` 自动忽略这些位置的损失
- **只学习回复内容**：模型只需要学会生成 assistant 的回答，不需要拟合用户的输入
- **保持 prompt 长度灵活性**：不需要 mask attention，只是不算损失

### 极端情况

如果整条对话 `max_seq_len` 截断后只包含 prompt 段（user 说的话），那么所有的 labels 都是 -100，loss = 0。这种情况虽然罕见但需要注意——数据预处理时应该过滤掉这种样本。

> 面试点：如果一条对话全部被截断为 prompt 端怎么办？→ loss=0，该样本对训练无贡献；需要在数据预处理时过滤或截断时尽量保留 assistant 段

---

## Q2. 加载预训练权重机制

### 权重加载流程

```
1. from_weight=pretrain
       │
2. 确定权重路径：
   model_dir / from_weight_hidden_size.pth
   → checkpoint/lm_pretrain/pretrain_512.pth
       │
3. torch.load(..., map_location=device)
       │
4. model.load_state_dict(weights, strict=False)
       │
5. 选择不加载 lm_head.weight（可选）
```

### 代码实现

```python
# src/utils/training.py:145-168
def init_model(config, save_dir, weight_type='pretrain', model_dir=None):
    if weight_type != 'none':
        weight_dir = model_dir or save_dir              # model_dir 优先
        weight_path = f'{weight_dir}/{weight_type}_{config.hidden_size}.pth'
        weights = torch.load(weight_path, map_location=device)
        # 可选择跳过 lm_head（如新增词表时）
        # weights = {k: v for k, v in weights.items() if 'lm_head' not in k}
        model.load_state_dict(weights, strict=False)
```

### model_dir vs save_dir

| 参数 | 作用 | 默认值 |
|---|---|---|
| `save_dir` | checkpoint 写入目录 | 配置中的 `paths.save_dir` |
| `model_dir` | checkpoint 读取目录 | 未设置时 = save_dir |

为什么要分开？
- pretrain 权重存在 `checkpoint/lm_pretrain/`
- SFT 权重存在 `checkpoint/lm/`
- SFT 需要从 pretrain 目录**加载**，但**保存**到自己目录
- 没有 `model_dir` 时会在 `checkpoint/lm/` 下找 `pretrain_512.pth`，找不到

### 不同 from_weight 的语义

| from_weight | 用途 | 加载的文件 |
|---|---|---|
| `none` | 从头训练 | 不加载（随机初始化） |
| `pretrain` | pretrain → SFT 持续训练 | `pretrain_{h}.pth` |
| `full_sft` | SFT 继续训练/增量 SFT | `full_sft_{h}.pth` |

### strict=False 的注意事项

- 允许权重文件和模型结构**不完全一致**
- 常见的 mismatch 来源：
  1. embedding 和 lm_head 的 weight tying（两个 key 映射到同一个参数字典）
  2. 词表大小变化（新增 special token 后加载旧权重）
  3. 模型架构微调（如增减层数）
- `strict=False` 会静默忽略多出的键（缺少的键会随机初始化）

> 面试点：strict=False 实际可能导致模型部分随机初始化而不报错，如何确保没有遗漏？→ 加载后比较 `model.state_dict().keys()` 和 `weights.keys()`，打印 missing_keys 和 unexpected_keys

---

## Q3. SFT 训练流程详解

### 训练循环（`src/trainers/lm/full_sft.py`）

```python
def train_epoch(epoch, model, loader, optimizer, scheduler, scaler, args):
    model.train()
    total_loss = 0
    for step, (input_ids, labels) in enumerate(loader):
        input_ids = input_ids.cuda()
        labels = labels.cuda()

        with autocast_ctx:
            logits = model(input_ids)
            loss = cross_entropy(logits[..., :-1, :].contiguous(),
                                 labels[..., 1:].contiguous(),
                                 ignore_index=-100)
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += loss.item()
```

### 与 Pretrain 训练循环的区别

| | Pretrain | Full SFT |
|---|---|---|
| 是否加载预训练权重 | 否（from_weight=none） | 是（from_weight=pretrain） |
| 学习率 | 5e-4（从零学习） | 1e-5（微调，小 lr） |
| 训练轮数 | 2 | 2 |
| 数据集 | 纯文本 1.27M 条 | 指令对话 60K 条 |
| loss 计算 | 全 token | 仅 assistant 段 |
| weight_decay | 有 | 有（默认 0.1） |

### 为什么 SFT 学习率要小？

- 预训练权重已经学到很好的语言表示
- 大学习率会破坏/覆盖预训练知识（灾难性遗忘）
- 目标是在已有知识基础上"微调"指令跟随能力
- 一般 pretrain LR : SFT LR ≈ 10:1 ~ 50:1

---

## Q4. 灾难性遗忘（Catastrophic Forgetting）

### 什么灾难性遗忘？

模型在 SFT 阶段学会生成对话回复的同时，**丢失**了在预训练阶段学到的通用知识（如常识推理、知识问答能力）。

### 为什么 SFT 会导致遗忘？

```
预训练：    语言 P(x₁...xₙ)  ← 通用分布
SFT：       条件 P(回复|指令)  ← 狭窄分布
     ──────────────────────→
     训练分布偏移 → 覆盖预训练权重
```

### 缓解策略

1. **小学习率**：1e-5 比 5e-4 小 50 倍，梯度更新量小
2. **少轮数**：1-2 轮足够，更多轮次会导致过拟合和遗忘
3. **保留预训练数据**：混合 SFT 数据 + 10-20% 预训练数据（本仓库未实现）
4. **EWC / LwF**：正则化方法，限制重要参数的大幅更新（本仓库未实现）
5. **LoRA**：增量微调，冻结原权重（本仓库另有 `src/trainers/lm/lora_sft.py`）

> 面试点：什么情况下灾难性遗忘最严重？→ 大量 SFT 数据 + 高学习率 + 多轮训练 + 领域单一的数据集

---

## Q5. SFT 训练问题诊断

### Loss 正常值范围

| 阶段 | Loss | 说明 |
|---|---|---|
| 初始（第 1 步） | ~7.0-8.5 | 刚加载 pretrain 权重，但换数据集后分布不同 |
| 收敛 | ~1.5-2.5 | 模型学会生成合理回复 |
| 过拟合 | < 1.0 | 训练 loss 极低但生成质量差（记忆而不是泛化） |

### Loss 异常分析

```
Loss 行为                 可能原因                          建议
────────────────────────────────────────────────────────────────────
初始 loss 极低 (<2.0)     pretrain 数据集和 SFT 高度重叠    check 数据分布
loss 不下降 (<30 步)      学习率太小 / 模型冻结              check 梯度
loss 突增                 学习率太大 / 梯度爆炸              减少 lr / 加强 grad_clip
loss 震荡                  batch 太小 / lr 太高              调大 batch / 减小 lr
loss 降到 0.0             所有 label 为 -100                 检查数据截断
val loss 上升但 train     train loss 下降过拟合              early stopping / 正则化
```

---

## Q6. 推理时生成差异

### 训练 vs 推理行为对比

```python
# 训练时
model.train()
logits = model(input_ids)                          # 全部序列
loss = cross_entropy(shift_logits, shift_labels)   # 不采样，计算损失

# 推理时
model.eval()
generated = model.generate(input_ids, max_new_tokens=256, do_sample=True, temperature=0.7)
```

### 推理超参

| 参数 | 作用 | SFT 推荐值 |
|---|---|---|
| `do_sample` | 是否采样（否则 greedy） | `True` |
| `temperature` | 采样温度，越高越随机 | 0.7-0.9 |
| `top_k` | 只从前 K 个 token 采样 | 40-50 |
| `top_p` | 核采样（累积概率 p） | 0.9 |
| `repetition_penalty` | 重复惩罚 | 1.05-1.15 |

### 温度对比

```
Temperature=0.1: "今天天气真好，我们去散步吧。"
Temperature=0.7: "今天天气真好，要不出去走走？"
Temperature=1.5: "天气不错，散步散步散步吧...哦不对不对呵呵呵"
```

- 温度太低 → 输出机械、重复
- 温度太高 → 输出发散、语无伦次
- 0.7 是创造性 + 连贯性的良好平衡点

---

## Q7. YAML 配置详解

### 结构说明

```yaml
model:
  hidden_size: 512            # 模型容量：影响参数量和激活值大小
  num_hidden_layers: 8        # Transformer 层数
  use_moe: 0                  # MoE 开关
  vocab_size: 6400            # 词表大小
  max_seq_len: 768            # SFT 通常需要更长上下文
  num_attention_heads: 8      # Q 头数
  num_key_value_heads: 4     # K/V 头数（GQA，4:8=2x 压缩）
  dropout: 0.0                # SFT 一般不加 dropout

train:
  epochs: 2
  batch_size: 16              # 受显存限制
  learning_rate: 1.0e-5       # 微调用小 lr
  accumulation_steps: 1
  grad_clip: 1.0
  dtype: bfloat16
  save_interval: 1000
  log_interval: 100
  from_weight: pretrain       # 加载预训练权重
  model_dir: checkpoint/lm_pretrain  # 预训练权重来源目录
  from_resume: 0

paths:
  save_dir: checkpoint/lm     # 训练产物存放目录
  data_path: dataset/lm/sft.jsonl
```

### max_seq_len 为什么比 pretrain 大？

| | Pretrain | SFT |
|---|---|---|
| max_seq_len | 340 | 768 |
| 原因 | 预训练数据多为短文本（如 BERT 风格片段） | 多轮对话需要更多空间 |

对话拼接后 token 数≈ sum of turns，通常比单篇文本长。

---

## Q8. AdamW 优化器

### 与 Adam 的区别

```python
# Adam:     w_{t+1} = w_t - lr * m_hat / (sqrt(v_hat) + eps)   # 无 weight decay
# AdamW:    w_{t+1} = w_t - lr * (m_hat / (sqrt(v_hat) + eps) + λ*w_t)
#                     └─────────────────────────────┬──────────────┘
#                                                   └ weight decay 与梯度更新解耦
```

Adam 将 weight decay 和 L2 正则化混在一起（L2 = 在 loss 上加 λ/2 × ||w||²），而 AdamW 将 weight decay 从自适应学习率中解耦出来。

### 为什么 AdamW 更好？

| | Adam (L2) | AdamW |
|---|---|---|
| Decay 位置 | loss 函数中（对梯度贡献） | optimizer 更新时独立加 |
| 自适应影响 | decay 也被 m_hat/v_hat 缩放 | decay 不受影响 |
| 实际效果 | 大学习率下 decay 被自适应削弱 | 稳定的 decay 效果 |
| 业界标准 | 旧方法 | GPT/LLaMA 等现代模型标配 |

### 本仓库配置

```python
optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
```

- `weight_decay=0.1` 是常见推荐值
- 一般不对 bias 和 norm 参数做 weight decay（但此项目未做区分)
- PyTorch 的 AdamW 默认 `betas=(0.9, 0.999)`, `eps=1e-8`

---

## Q9. SFT 评估方法

### 评估维度

| 维度 | 评估方式 | 指标 |
|---|---|---|
| 指令跟随 | 人工/模型评估 | 是否按指令完成 |
| 生成质量 | 人工评分 | 连贯性/有用性/安全性 |
| 多样性 | 统计 | distinct-1/2, ngram 重复率 |
| 知识正确性 | 基准测试 | MMLU, CEval, CMMLU |

### 本仓库评估脚本（`scripts/eval_llm.py`）

```bash
# 原生 torch 格式
python scripts/eval_llm.py --native --save_dir checkpoint/lm_full_sft_mini \
                           --weight full_sft --hidden_size 128

# HuggingFace 格式
python scripts/eval_llm.py --load_from checkpoint/omni/native_hf \
                           --tokenizer_path checkpoint/omni/native_hf
```

```
生成评估结果（示例）：
────────────────────────────────────
User: 讲个笑话
Assistant: 为什么程序员总把万圣节和圣诞节搞混？
因为 Oct 31 == Dec 25！
────────────────────────────────────
User: 用 Python 写一个快速排序
Assistant: def quicksort(arr):
    if len(arr) <= 1: return arr
    pivot = arr[len(arr)//2]
    left  = [x for x in arr if x < pivot]
    mid   = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + mid + quicksort(right)
────────────────────────────────────
```

### 常见问题

- **回复过短**：「是的」「好的」→ 数据集过于简单或数据量不够
- **回复重复**：不断生成相同短语 → 温度太低或 repetition_penalty 太小
- **偏离主题**：模型开始乱说 → 训练不足或 temperature 太高
- **不能按格式输出**：要求 JSON 但输出自然语言 → 数据集中缺乏格式化示例

---

## Q10. SFT vs RLHF 的关系

### SFT 的局限性

1. **模仿而非优化**：SFT 只是让模型模仿人工回复分布，不是优化最终效果
2. **暴露偏差**：训练时使用 teacher forcing（每步输入真实 token），推理时输入是自生成的 token，分布偏移
3. **缺乏偏好对齐**：所有训练样本被视为同等正确，区分不出"好回答"和"更好回答"

### RLHF 如何解决？

```
SFT 阶段：模仿示范数据
    │
    ▼
Reward 模型训练：学习偏好排序
    │
    ▼
PPO 阶段：以 reward 为信号优化策略
    │
    ▼
结果：模型知道什么"更好"，不仅仅是"像什么"
```

### 本仓库的 RL 系列

- `src/trainers/lm/dpo.py`：Direct Preference Optimization（PPO 的简化替代）
- `src/trainers/lm/ppo.py`：Proximal Policy Optimization（标准 RLHF）
- `src/trainers/lm/grpo.py`：Group Relative Policy Optimization（DeepSeek 方案）
- `src/trainers/lm/distill.py`：知识蒸馏

> 面试点：SFT 和 RLHF 的核心区别是什么？→ SFT 是监督学习（模仿示范），RLHF 是从偏好信号中学习优化（区分好与更好）

---

## Q11. Teacher Forcing 与 Exposure Bias

### Teacher Forcing

```python
# 训练时：每次输入真实 token
for t in range(seq_len):
    logit = model(input_ids[:, :t+1])
    loss  = CE(logit[:, t, :], labels[:, t])

# 等价于一次性算全部
logits = model(input_ids)
loss   = CE(shift_logits, shift_labels)
```

### 问题：Exposure Bias

```
训练时：
  input:  "中国的首都是"  → 模型预测 → "北京"
  实际输入下一时间步: "北京" (真实 token)

推理时：
  input:  "中国的首都是"  → 模型预测 → "上海" (错误!)
  实际输入下一时间步: "上海" (自己的预测, 错上加错)

        训练分布 ≠ 推理分布 → 累积误差
```

### 缓解方法

1. **Scheduled Sampling**：推理时以一定概率用模型自己的预测替换真实 token（本仓库未实现，但面试常考）
2. **强化学习**（RLHF阶段）：直接在自生成序列上优化
3. **Beam Search**：推理时维护候选路径，减少单步错误的累积影响

---

## Q12. 实际训练资源估算

### 30M 模型 SFT 成本

| 项目 | 估算 |
|---|---|
| 参数量 | ~30M（hidden_size=512, L=8） |
| 总步数 | `ceil(60000/16) × 2 = 7500` |
| 每步时间 | ~250ms (RTX 4060) |
| 总时间 | `7500 × 0.25 ≈ 31 分钟` |
| 峰值显存 | ~4-5 GB（bf16, bs=16, seq=768） |
| 权重大小 | ~60 MB（fp16 保存） |

### 大数据全量 SFT 估算（实际生产）

| 数据量 | batch_size | 步数 | 每步时间 | 总时间 |
|---|---|---|---|---|
| 10K | 16 | 1250 | ~250ms | ~5 分钟 |
| 60K | 16 | 7500 | ~250ms | ~31 分钟 |
| 500K | 16 | 62500 | ~250ms | ~4.3 小时 |

> 面试点：如何加速 SFT 训练？→ 增大 batch_size（需更大显存或多卡）→ 减少步数；使用梯度累积补偿显存不足；使用 DeepSpeed ZeRO 节省显存

---

## 面试高频题汇总

### 基础

1. **SFT 和 Pretrain 训练的核心区别？** → 数据格式（纯文本 vs 对话）、loss 计算（全 token vs assistant only）、学习率（5e-4 vs 1e-5）、权重初始化
2. **为什么 label 要置 -100？** → `CrossEntropyLoss(ignore_index=-100)` 忽略该位置损失，只计算 assistant 段
3. **Teacher Forcing 是什么？** → 训练时每步输入真实 token 而非模型预测
4. **灾难性遗忘怎么避免？** → 小 lr、少轮数、混合预训练数据、LoRA 增量微调

### 进阶

5. **Exposure Bias 是什么？** → 训练（teacher forcing）和推理（自回归）的输入分布不一致导致的误差累积
6. **SFT 后为什么需要 RLHF？** → SFT 只是模仿，RLHF 从偏好信号中学习"什么更好"
7. **AdamW 比 Adam 好在哪？** → weight decay 与自适应学习率解耦，更有效的正则化
8. **weight tying 在微调时有用吗？** → 有用，嵌入层和输出头共享权重能提升泛化和收敛
9. **strict=False 的潜在风险？** → 部分参数随机初始化而不报错，需要手动验证加载结果
10. **如何处理超长对话？** → 截断（丢失信息）、滑动窗口（窗口训练）、压缩（使用长上下文模型）
