# 面试：训练系统深度

> 本仓库 `src/trainers/` 的训练逻辑，覆盖 pretrain、SFT、DPO、PPO、GRPO、蒸馏

## 0. 训练流程概览

```
数据集 (dataset/)
    │
    ▼
SkipBatchSampler → DataLoader → 批次拼接
    │
    ▼
Trainer (trainers/)
    ├── 模型初始化（支持 checkpoint 恢复）
    ├── 混合精度训练（bfloat16/float16）
    ├── 梯度累积 + 梯度裁剪
    ├── 学习率调度（cosine）
    └── 定期保存 checkpoint
```

---

## Q1. 损失里 label 为什么要平移？

### 标准 Next-Token Prediction

```python
# src/models/lm/model.py:69-70
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
```

### 为什么平移？

- `logits[:, :-1]` 预测的是 `labels[:, 1:]` 的内容
- 标准 next-token 预测：给定前缀，预测下一个 token
- 不平移的话，模型会学到"预测自己"，没有意义

> 面试点：如果 label 不平移会怎样？→ 模型会学到恒等映射，训练 loss 很低但生成质量很差

---

## Q2. SFT 为什么只标 assistant 段？

### Loss Mask 机制

```python
# src/dataset/sft.py
def generate_labels(self, input_ids):
    labels = input_ids.clone()
    # prompt/system 段 label 置 -100
    labels[:prompt_end_pos] = -100
    # 只对 assistant 回复计算损失
    return labels
```

### 为什么只标 assistant？

1. **避免学用户说话**：模型应该学习生成回复，不拟合用户输入
2. **提高效率**：只计算有意义部分的损失
3. **符合实际使用**：推理时只生成 assistant 回复

> 面试点：如果标全部 token 会怎样？→ 模型会学用户输入的模式，生成时可能重复用户说话风格

---

## Q3. 配置系统优先级？

### 三级优先级

```
CLI 参数 > YAML 默认值 > 代码默认
```

### 实现（`src/utils/training.py:25-38`）

```python
def apply_config(parser, default_config=None):
    # 1. 先加载 YAML 默认值
    if default_config:
        with open(default_config) as f:
            config = yaml.safe_load(f)
        # 2. 扁平化注入 argparse 默认值
        for key, value in config.items():
            if key in [a.dest for a in parser._actions]:
                parser.set_defaults(**{key: value})
    # 3. CLI 参数覆盖
    return parser.parse_args()
```

### 为什么这样设计？

1. **灵活性**：可以用 YAML 配置常用参数，CLI 临时覆盖
2. **可复现**：YAML 文件可以版本控制
3. **向后兼容**：代码默认值保证基本功能

---

## Q4. 续训怎么保证精确？

### SkipBatchSampler（`src/utils/training.py:177-200`）

```python
class SkipBatchSampler:
    def __init__(self, dataset, batch_size, step):
        self.step = step  # 已完成的 step 数
    
    def __iter__(self):
        # 跳过前 step 个 batch
        indices = list(range(len(self.dataset)))
        indices = indices[self.step * self.batch_size:]
        ...
```

### Checkpoint 恢复

```python
# src/utils/training.py:105-158
def lm_checkpoint(model, optimizer, scheduler, step, path):
    # 原子保存
    tmp_path = path + ".tmp"
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, tmp_path)
    os.replace(tmp_path, path)
```

### 世界规模自适应（`src/utils/training.py:153-156`）

```python
# GPU 数量变化时自动调整 step
if world_size != saved_world_size:
    step = int(step * saved_world_size / world_size)
```

> 面试点：为什么需要世界规模自适应？→ 多机训练时 GPU 数量可能变化，需要按比例调整 step 保证训练进度一致

---

## Q5. DPO Loss 实现

### 标准 DPO Loss（`src/trainers/lm/dpo.py:32-48`）

```python
def dpo_loss(policy_logratios, reference_logratios, beta=0.15):
    # DPO Loss = -logsigmoid(β * (π_logratios - ref_logratios))
    loss = -F.logsigmoid(beta * (policy_logratios - reference_logratios))
    return loss.mean()
```

### 关键参数

- `beta=0.15`：控制策略偏离参考模型的程度
- `lr=4e-8`：极小学习率防止遗忘

### 参考模型（`src/trainers/lm/dpo.py:187-189`）

```python
# 冻结的 ref_model
self.ref_model = init_model(args, config)
for param in self.ref_model.parameters():
    param.requires_grad = False
```

> 面试点：为什么 DPO 学习率这么小？→ DPO 直接优化策略，太大的学习率会导致策略偏离参考模型太远，生成质量下降

---

## Q6. GRPO 实现（`src/trainers/lm/grpo.py:119-142`）

### 核心思想

Group Relative Policy Optimization：每个 prompt 生成多个候选，组内标准化优势。

### 实现细节

```python
def grpo_loss(self, logprobs, old_logprobs, advantages, clip_epsilon=0.2):
    # 每个 prompt 生成 num_generations=6 个候选
    # 优势估计：组内标准化
    advantages = (reward - mean) / (std + 1e-4)
    
    # PPO Clip
    ratio = torch.exp(logprobs - old_logprobs)
    clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon)
    loss = -torch.min(ratio * advantages, clipped * advantages)
    
    # KL 惩罚
    kl = logprobs - old_logprobs
    kl_penalty = torch.exp(kl) - kl - 1
    
    return loss.mean() + self.kl_coef * kl_penalty.mean()
```

### 两种 Loss 模式

1. **"cispo"**：高端 clamped ratio（`epsilon_high=5.0`）
2. **"grpo"**：标准 PPO clip（`epsilon=0.2`）

### 奖励函数（`src/trainers/lm/grpo.py:35-66`）

```python
def reward_fn(self, text):
    reward = 0
    
    # 长度奖励：20-800 字符 +0.5
    if 20 < len(text) < 800:
        reward += 0.5
    
    # thinking 奖励：20-300 字符 +1.0
    if '<think>' in text and 20 < think_len < 300:
        reward += 1.0
    
    # thinking 次数奖励：恰好 1 次 +0.25
    if text.count('<think>') == 1:
        reward += 0.25
    
    # 重复惩罚：3-gram 重复度
    reward -= repetition_penalty(text)
    
    # Reward Model 分数
    rm_score = self.reward_model(text)
    reward += rm_score
    
    return reward
```

> 面试点：GRPO 和 PPO 的区别？→ GRPO 不需要 critic model，直接用组内标准化估计优势，更简单高效

---

## Q7. PPO 实现（`src/trainers/lm/ppo.py`）

### CriticModel（`src/trainers/lm/ppo.py:35-47`）

```python
class CriticModel(LMForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        # lm_head 替换为 value_head (hidden_size -> 1)
        self.value_head = nn.Linear(config.hidden_size, 1, bias=False)
        del self.lm_head
```

### GAE（`src/trainers/lm/ppo.py:138-145`）

```python
def compute_gae(rewards, values, gamma=1.0, lam=0.95):
    advantages = []
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    return advantages
```

### 早停机制（`src/trainers/lm/ppo.py:181-188`）

```python
def ppo_update(self, ...):
    approx_kl = (logprobs - old_logprobs).mean()
    if approx_kl > 0.25:
        # 早停，但保持 DDP 通信闭环
        loss = loss * 0.0
    return loss
```

> 面试点：为什么早停时要 `loss * 0.0`？→ DDP 要求所有 rank 都参与前向/反向传播，`loss * 0.0` 保持计算图连通，避免死锁

---

## Q8. 蒸馏实现（`src/trainers/lm/distillation.py`）

### KL 散度蒸馏（`src/trainers/lm/distillation.py:23-34`）

```python
def kl_divergence(teacher_logits, student_logits, temperature=1.5):
    # KL(teacher || student) * temperature^2
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    return kl * temperature ** 2
```

### 混合损失（`src/trainers/lm/distillation.py:91`）

```python
loss = alpha * ce_loss + (1 - alpha) * kl_loss
# 默认 alpha=0.5, temperature=1.5
```

### Teacher/Student 独立配置

```python
# Teacher 和 Student 可以有不同的配置
teacher_config = LMConfig(hidden_size=768, num_hidden_layers=8)
student_config = LMConfig(hidden_size=384, num_hidden_layers=4)
```

> 面试点：为什么蒸馏要乘以 temperature²？→ 保持梯度尺度一致，避免温度变化影响学习率

---

## Q9. Rollout Engine 设计（`src/trainers/lm/rollout_engine.py`）

### 策略模式

```python
class RolloutEngine(ABC):
    @abstractmethod
    def generate(self, prompts, **kwargs):
        pass

class TorchRolloutEngine(RolloutEngine):
    def generate(self, prompts, **kwargs):
        # 原生 PyTorch 推理
        ...

class SGLangRolloutEngine(RolloutEngine):
    def generate(self, prompts, **kwargs):
        # 通过 HTTP API 调用 SGLang 服务
        ...
```

### 权重同步（`src/trainers/lm/rollout_engine.py:165-188`）

```python
def sync_weights(self, model):
    # 仅 rank 0 执行保存
    if self.rank == 0:
        # 保存到磁盘
        torch.save(model.state_dict(), 'tmp_model.pt')
        # HTTP 请求 SGLang 热加载
        requests.post('http://localhost:8000/load_model', ...)
    # 广播成功标志
    dist.broadcast(success_flag, src=0)
```

> 面试点：为什么要权重同步？→ RL 训练需要最新的策略生成 rollout，必须确保推理服务使用最新权重

---

## Q10. 混合精度训练

### 本仓库的混合精度策略

```python
# src/trainers/lm/pretrain.py
scaler = GradScaler(enabled=(args.dtype == 'float16'))
with autocast(device_type='cuda', dtype=dtype):
    logits, _, aux_loss = model(input_ids, labels=labels)
    loss = criterion(logits, labels) + aux_loss

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
scaler.step(optimizer)
scaler.update()
```

### 为什么用 bfloat16 而不是 float16？

- **bfloat16**：8 位指数，范围大，不易溢出
- **float16**：5 位指数，范围小，容易溢出

> 面试点：GradScaler 的作用？→ 防止 float16 梯度下溢，动态调整 loss 缩放因子

---

## Q11. 梯度累积

### 为什么需要梯度累积？

显存有限时，可以用小 batch 大 accumulation 模拟大 batch。

### 实现（`src/trainers/lm/pretrain.py`）

```python
accumulation_steps = 8  # 默认
for i, batch in enumerate(dataloader):
    with autocast(device_type='cuda', dtype=dtype):
        logits, _, aux_loss = model(batch)
        loss = criterion(logits, labels) + aux_loss
        loss = loss / accumulation_steps  # 梯度累积需要除以步数
    
    scaler.scale(loss).backward()
    
    if (i + 1) % accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
```

> 面试点：为什么 loss 要除以 accumulation_steps？→ 保持梯度尺度一致，避免累积步数影响学习率

---

## Q12. 学习率调度（`src/utils/training.py:82-83`）

### Cosine 调度

```python
def get_lr(step, lr, total):
    # 最低衰减到 10%
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * step / total)))
```

### 为什么用 Cosine？

1. **平滑衰减**：避免学习率骤降
2. **实验效果好**：比 step decay 和 linear decay 更稳定
3. **标准做法**：几乎所有大模型都用 cosine

---

## Q13. Checkpoint 原子写入（`src/utils/checkpoint.py:7-10`）

### 问题

训练过程中保存 checkpoint 时，如果中途崩溃，可能导致 checkpoint 损坏。

### 解决方案

```python
def save_checkpoint(model, optimizer, scheduler, path):
    # 先保存到临时文件
    tmp_path = path + ".tmp"
    torch.save({...}, tmp_path)
    # 原子替换
    os.replace(tmp_path, path)
```

### 为什么用 os.replace？

`os.replace` 是原子操作，要么完成要么不发生，不会出现部分写入的情况。

---

## Q14. DDP 初始化（`src/utils/distributed.py`）

### 标准 DDP 初始化

```python
def init_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank
```

### DDP 死锁防护

```python
# PPO 早停时保持通信闭环
if early_stop:
    loss = loss * 0.0  # 保持计算图连通
```

> 面试点：为什么 DDP 会死锁？→ 如果某个 rank 不参与前向/反向传播，其他 rank 会等待它，导致死锁

---

## Q15. 世界规模自适应（`src/utils/training.py:153-156`）

### 问题

多机训练时 GPU 数量可能变化，需要按比例调整 step。

### 实现

```python
def lm_checkpoint(model, optimizer, scheduler, step, path):
    # 保存世界规模
    torch.save({
        'step': step,
        'world_size': dist.get_world_size(),
    }, path)
    
    # 恢复时检查世界规模
    saved_world_size = checkpoint['world_size']
    if world_size != saved_world_size:
        step = int(step * saved_world_size / world_size)
```

> 面试点：为什么需要这个？→ 保证训练进度一致，避免某些 rank 重复训练或跳过训练
