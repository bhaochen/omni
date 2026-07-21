# 面试：预训练（Pretrain）深度

> 本仓库 `src/trainers/lm/pretrain.py`，对应 config：`configs/lm/lm_pretrain.yaml`

## 0. 整体流程

```
数据集 (pretrain_t2t_mini.jsonl)
    │  1,270,238 条纯文本
    ▼
PretrainDataset
    ├── 读取 jsonl，每条取 text 字段
    ├── tokenizer.encode() → input_ids
    ├── labels = input_ids 克隆（完整序列监督）
    └── 截断 / 填充到 max_seq_len
    │
    ▼
SkipBatchSampler → DataLoader
    │  batch_size 个样本 / 组
    │  SkipBatchSampler 支持断点续训跳过前 N 步
    ▼
Forward: model(input_ids, labels)
    ├── TokenEmbed → N×Block → RMSNorm → lm_head
    └── CE Loss(input=logits[..., :-1, :], target=labels[..., 1:])
    │
    ▼
Backward → 梯度累积 → 梯度裁剪 → Optimizer.step()
    │
    ▼
  Cosine LR 调度 → 定期保存 checkpoint
```

---

## Q1. PretrainDataset 怎么处理数据？

### 代码（`src/dataset/pretrain.py`）

```python
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=340):
        self.data = self.load_data(data_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_data(self, data_path):
        with open(data_path, 'r') as f:
            return [json.loads(line)['text'] for line in f]

    def __getitem__(self, index):
        text = self.data[index]
        input_ids = self.tokenizer.encode(text, max_length=self.max_length,
                                          truncation=True, padding='max_length',
                                          return_tensors='pt')[0]
        labels = input_ids.clone()
        return input_ids, labels
```

### 与 SFT 数据集的关键区别

| | PretrainDataset | SFTDataset |
|---|---|---|
| 数据格式 | 纯文本 | 多轮对话 JSON |
| label 构造 | `labels = input_ids.clone()`（全监督） | prompt 段置 `-100`，只标 assistant |
| 损失计算 | 所有 token 都参与 | 只有 assistant 回复参与 |
| 训练目标 | 下一个 token 预测 | 指令跟随回复生成 |

> 面试点：预训练时所有 token 都计算损失 vs SFT 只算 assistant 段，为什么？→ 预训练目标是学习语言分布，所有 token 都提供统计信息；SFT 是学习指令跟随能力，只需要拟合回复

---

## Q2. 损失函数具体怎么算？

### Next-Token Prediction + 标签平移

```python
# src/models/lm/model.py:63-70
def forward(self, input_ids, labels=None):
    hidden_states = self.model(input_ids)
    logits = self.lm_head(hidden_states)          # [B, S, V]

    if labels is not None:
        shift_logits = logits[..., :-1, :].contiguous()       # [B, S-1, V]
        shift_labels = labels[..., 1:].contiguous()           # [B, S-1]
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
```

### 为什么平移？

- `logits[t]` 预测的是 `labels[t+1]`（第 t 个位置的输出预测第 t+1 个位置的 token）
- 不平移的话，模型学到的是恒等映射：`logits[t] ≈ labels[t]`
- 训练时看似 loss 很低，但生成时完全无法产生新内容

> 面试点：平移的本质是什么？→ 将语言模型建模为条件概率 P(xₜ|x₁,...,xₜ₋₁)，第 t 个位置的输出对应第 t+1 个 token 的概率分布

### 辅助损失：MoE aux_loss

```python
loss = res.loss + res.aux_loss
```

当 `use_moe=1` 时，MoE 层还会产生一个 auxiliary load-balancing loss，鼓励专家负载均衡。非 MoE 模式下 `aux_loss = 0`。

---

## Q3. 训练超参如何影响模型？

### 核心参数详解

| 参数 | 默认值 | 作用 | 调大影响 | 调小影响 |
|---|---|---|---|---|
| `hidden_size` | 512 | 每 token 的表示维度 | 容量↑，速度↓ | 容量↓，速度↑ |
| `num_hidden_layers` | 8 | Transformer 层数 | 深度↑，梯度传播难 | 深度↓，表达弱 |
| `batch_size` | 32 | 每步样本数 | 梯度稳，显存↑ | 梯度噪，显存↓ |
| `accumulation_steps` | 8 | 梯度累积步数 | 等效 batch↑，速度不变 | 等效 batch↓ |
| `max_seq_len` | 340 | 最大序列长度 | 上下文↑，显存↑ | 上下文↓ |
| `learning_rate` | 5e-4 | 初始学习率 | 收敛快，可能不稳 | 收敛慢，更稳 |
| `dtype` | bfloat16 | 混合精度类型 | 精度高，速度中 | 精度低，速度快 |

### 等效 Batch Size

```
等效 batch = batch_size × accumulation_steps
```

本仓库 pretrain 默认 `batch_size=32, accumulation_steps=8` → 等效 batch = 256。

### 梯度累积实现

```python
# src/trainers/lm/pretrain.py:36-47
loss = loss / args.accumulation_steps     # 归一化
scaler.scale(loss).backward()             # 累积梯度

if step % args.accumulation_steps == 0:   # 累积够 N 步后
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)                # 更新参数
    scaler.update()
    optimizer.zero_grad(set_to_none=True) # 清空梯度
```

> 面试点：为什么 loss 要除以 accumulation_steps？→ 因为梯度是链式累积的，不归一化的话等效学习率会放大 accumulation_steps 倍。除以步数后，每步梯度相当于独立 batch 梯度的平均，保持学习率语义不变

---

## Q4. 学习率调度策略？

### Cosine Decay

```python
# src/utils/training.py:82-83
def get_lr(current_step, total_steps, lr):
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))
```

### 调度曲线

```
lr
↑
│    lr * 0.55 ─────────── 余弦下降 ──→ lr * 0.1
│    (初始)                          (最终)
└─────────────────────────────────────→ step
```

- 初始 LR = `lr * 0.55`（cos 从 0 开始，`1 + cos(0) = 2` → `0.1 + 0.45*2 = 1.0`... 不对）
  仔细看：`current_step=0` 时 `cos(0)=1` → `0.1 + 0.45*2 = 1.0` → `lr * 1.0 = lr`
  所以在 step 0 时 LR 正好等于设置的 learning_rate。
- 最终 LR = `lr * 0.1`（cos(π) = -1 → `0.1 + 0.45*0 = 0.1`）
- 即 LR 从 `lr` 余弦衰减到 `0.1 * lr`

这种"不下到 0"的调度比标准 cosine 更好，保持模型在训练后期仍有适度更新能力。

---

## Q5. 混合精度训练怎么做？

### AMP (Automatic Mixed Precision)

```python
# src/trainers/lm/pretrain.py:121-123
dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
autocast_ctx = torch.cuda.amp.autocast(dtype=dtype)
```

### bf16 vs fp16

| | bf16 | fp16 |
|---|---|---|
| 指数位 | 8 位（同 fp32） | 5 位 |
| 尾数位 | 7 位 | 10 位 |
| 数值范围 | 同 fp32（~3.4e38） | 有限（~6.5e4） |
| 是否需要 GradScaler | ❌ 不需要 | ✅ 需要 |
| 精度 | 低精度（7bit 尾数） | 高精度（10bit 尾数） |
| 硬件要求 | Ampere+（3090/A100 等） | 几乎所有 GPU |

### 为什么 bf16 不需要 GradScaler？

bf16 的指数范围和 fp32 一样，不会发生梯度下溢。fp16 的指数范围只有 5 位，小梯度会直接变 0，需要用 GradScaler 放大梯度再缩小。

### 精度保留技巧

```python
# src/core/norm.py:7-8
x = x.float()                              # RMSNorm 内部转 fp32
x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
return (self.weight * x).type_as(self.weight)  # 转回 bf16/fp16
```

即使训练是 bf16，归一化层内部转 fp32 计算再转回，避免归一化精度损失。

---

## Q6. 训练时显存都花在哪了？

### 显存分布（以 30M 参数模型为例）

```
模型参数 (fp32):      120 MB
梯度 (fp32):          120 MB
Adam 状态 (fp32×2):   240 MB
─────────────────────────────
模型状态总计:          480 MB

激活值 (bf16, 8层):   ~3-5 GB  ← 大头
输入数据:              <10 MB
CUDA 上下文:           ~200 MB
─────────────────────────────
总计:                  ~4-6 GB（取决于 batch_size 和 seq_len）
```

### 为什么激活值占这么多？

- 反向传播需要存储每层的中间激活值
- 每层存储 Q/K/V、注意力输出、MLP 中间结果等
- 数量级：`O(batch_size × seq_len × hidden_size × num_layers × k)`，k ≈ 15-34

### 没有激活检查点

本仓库**没有**使用 `torch.utils.checkpoint`（梯度检查点）。代价是激活值全存，好处是不需要重计算，训练速度更快。

> 面试点：激活检查点 trade-off 是什么？→ 节省显存（存部分激活，反向时重算），增加约 15-20% 计算时间

---

## Q7. 分布式训练是怎么做的？

### DDP (DistributedDataParallel)

```python
# src/utils/distributed.py
def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0                              # 单卡模式
    dist.init_process_group(backend="nccl")   # 多卡模式
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank
```

### 启动方式

```bash
# 单卡
python -m trainers.lm.pretrain --config configs/lm/lm_pretrain.yaml

# 多卡（torchrun）
torchrun --nproc_per_node=4 -m trainers.lm.pretrain --config configs/lm/lm_pretrain.yaml
```

### DDP 原理

- 每个 GPU 一张完整的模型副本
- 前向/反向独立计算
- 反向传播后通过 `allreduce` 同步梯度
- 每个 GPU 独立执行 optimizer.step()

### 本仓库没有使用

- ❌ ZeRO（DeepSpeed）
- ❌ FSDP
- ❌ 张量/序列并行
- ❌ torch.compile（默认关闭，`use_compile: 0`）

> 面试点：为什么小模型不用 ZeRO/FSDP？→ 模型仅 30M 参数，单卡就能装下，DDP 的梯度同步开销也很小。ZeRO/FSDP 的通信量更大，对小模型反而可能更慢

---

## Q8. Checkpoint 如何保存和恢复？

### 保存（`train_epoch` 内）

```python
# src/trainers/lm/pretrain.py:59-67
if (step % args.save_interval == 0 or step == iters) and is_main_process():
    ckp = f'{args.save_dir}/{args.save_weight}_{hidden_size}.pth'
    torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
    lm_checkpoint(lm_config, weight=args.save_weight, model=model,
                  optimizer=optimizer, epoch=epoch, step=step, ...)
```

### 保存两个文件

| 文件 | 内容 | 用途 |
|---|---|---|
| `pretrain_512.pth` | 模型权重 (fp16) | 推理/下游微调 |
| `pretrain_512_resume.pth` | 权重 + 优化器 + epoch + step | 断点续训 |

### 恢复训练 `from_resume=1`

```python
ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints')
if ckp_data:
    model.load_state_dict(ckp_data['model'])
    optimizer.load_state_dict(ckp_data['optimizer'])
    start_epoch = ckp_data['epoch']
    start_step = ckp_data.get('step', 0)
```

### 权重初始化 `from_weight`

```yaml
from_weight: none      # 从头训练（随机初始化）
from_weight: pretrain  # 加载 pretrain_512.pth 继续训练
from_weight: full_sft  # 加载 full_sft_512.pth 继续训练
```

`init_model` 会根据 `from_weight` 在 `save_dir` 下查找对应文件：

```python
weight_path = f'{save_dir}/{from_weight}_{hidden_size}.pth'
weights = torch.load(weight_path, map_location=device)
model.load_state_dict(weights, strict=False)
```

> 面试点：`strict=False` 意味着什么？→ 允许加载的权重和模型结构部分不匹配（如只加载 encoder 不加载 lm_head），LoRA 等场景常用

---

## Q9. SkipBatchSampler 如何实现断点续训？

```python
# src/utils/training.py:177-200
class SkipBatchSampler(Sampler):
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler                  # 原始索引（或 DistributedSampler）
        self.batch_size = batch_size
        self.skip_batches = skip_batches        # 跳过的步数

    def __iter__(self):
        batch = []
        skipped = 0
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:  # 跳过前 N 个 batch
                    skipped += 1
                    batch = []
                    continue
                yield batch
                batch = []
```

### 为什么需要跳过？

- 断点续训时已经从 checkpoint 恢复了优化器状态
- 但 DataLoader 从头开始迭代的话，会重复处理之前的数据
- `SkipBatchSampler` 通过 `skip_batches` 跳过已处理的 batch

### 使用场景

```python
skip = start_step if (epoch == start_epoch and start_step > 0) else 0
batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
```

---

## Q10. 随机种子和数据打乱

```python
# src/trainers/lm/pretrain.py:160
setup_seed(42 + epoch)
indices = torch.randperm(len(train_ds)).tolist()
```

### 每 epoch 重新打乱

- 每个 epoch 用不同的 `indices` 顺序
- 种子 = `42 + epoch`，保证可复现
- `DistributedSampler` 模式下，每个 GPU 拿到不同但确定性的分片

### 为什么 seed 要 + epoch？

- 每个 epoch 的数据顺序不同
- 同一 epoch 在不同运行间可复现
- 分布式下每个 rank 拿到不同的子集

---

## Q11. YAML 配置是如何生效的？

### 配置优先级

```
CLI 参数 > YAML 默认值 > Python argparse 默认值
```

### 实现机制

```python
# src/utils/training.py:25-38
def apply_config(parser, default_config=None):
    pre, _ = parser.parse_known_args()
    config_path = getattr(pre, 'config', None) or default_config
    if config_path and os.path.exists(config_path):
        defaults = _load_yaml_config(config_path)
        parser.set_defaults(**defaults)     # YAML 值设置为 argparse 默认值
    return parser.parse_args()              # CLI 显式传参仍可覆盖
```

### YAML 映射

```yaml
model:
  hidden_size: 512    → args.hidden_size
train:
  batch_size: 32      → args.batch_size
paths:
  save_dir: ...       → args.save_dir
```

YAML 的三个 section（model/train/paths）被扁平化为 argparse 参数后注入，之后任何 CLI 传参都可以覆盖。

---

## Q12. 为什么选 SwiGLU 作为激活函数？

### 公式对比

| 激活函数 | 公式 | 参数量 |
|---|---|---|
| ReLU FFN | `ReLU(xW₁)W₂` | 2×d×d_ff |
| SwiGLU | `SiLU(xW_g) * (xW_u) * W_d` | 3×d×d_ff |

SwiGLU 用三个投影（gate/up/down）替代两个，参数量增加 50%，但同等参数下效果更好。

### 仓库实现

```python
# src/core/mlp.py:16-17
def forward(self, x):
    return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

### intermediate_size 的奇怪计算

```python
# src/models/lm/config.py:22
self.intermediate_size = math.ceil(hidden_size * math.pi / 64) * 64
```

为什么用 π？这是一个取整技巧：`hidden_size × π / 64` 取整再 ×64，保证 intermediate_size 是 64 的倍数，有利于 GPU 内存对齐和 Tensor Core 加速。

对于 `hidden_size=512`：`intermediate_size = ceil(512 × π / 64) × 64 = ceil(25.13) × 64 = 1664`

---

## Q13. 训练过程中 loss 的变化规律

典型 pretrain loss 曲线：

```
loss
↑
8.0 │  █
7.0 │    ██
6.0 │      ███
5.0 │        ████
4.0 │          █████
3.0 │            ██████
    └────────────────────→ step
```

### 特征

1. **快速下降期**（前 5-10% 步数）：loss 从 ~8.5 降到 ~5.0，模型学到基础词法/语法模式
2. **平稳下降期**：loss 稳步下降，学习更复杂的语义/知识
3. **接近收敛**：loss 下降变缓，接近理论下界

### 预估最终 loss

对于 vocab_size=6400 的随机初始化模型：
- 初始 loss ≈ log(6400) ≈ 8.76（均匀分布的交叉熵）
- 训练后 loss ≈ 2.5-3.5（取决于模型大小和数据量）
- 理论下限 ≈ 0（完美拟合数据分布，但实际上达不到）

---

## 面试高频题汇总

### 基础

1. **预训练和 SFT 的区别？** → 预训练从零学习语言分布（全 token 监督），SFT 学习指令跟随（只监督回复）
2. **为什么用 bf16 而不是 fp16？** → bf16 指数范围同 fp32，无需 GradScaler，训练更稳定
3. **梯度累积的作用？** → 显存不足时用计算换显存，等效增大 batch_size
4. **Cosine 学习率调度的优缺点？** → 平滑衰减，早期快速学习后期精细调优，但可能过早衰减

### 进阶

5. **如何估计训练时间？** → `总步数 = ceil(样本数 / batch_size) × epochs`，`总时间 = 总步数 × 每步时间`
6. **为什么 loss 除以 accumulation_steps？** → 保持梯度期望不变，等效于 "平均 N 个 mini-batch 的梯度"
7. **DDP 和 DP 的区别？** → DDP 每个 GPU 独立前反向 + allreduce 梯度，DP 是单进程多线程（GIL 限制性能）
8. **weight tying 的作用？** → 共享 embedding 和 lm_head 的权重矩阵，减少参数量（本项目默认开启）
9. **GQA 和 MHA 的区别？** → GQA 减少 KV head 数量，降低 KV cache 大小，推理时更省显存
10. **Flash Attention 为什么省显存？** → tiling 计算注意力矩阵，不显式存储 O(n²) 的 score 矩阵
