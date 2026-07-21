# 面试：Transformer 架构深度

> 本仓库 `src/core/` 的纯 Transformer 组件，对应代码：`norm.py`、`rope.py`、`attention.py`、`mlp.py`、`block.py`

## 0. 整体架构图

```
输入 token id
    │
    ▼
┌─────────────┐
│ TokenEmbed  │  (vocab_size × d_model)
└──────┬──────┘
       │  + 位置编码(RoPE 作用于 Q/K)
       ▼
┌──────────────────────────────┐
│  Transformer Block × N       │
│  ┌────────────────────────┐  │
│  │  RMSNorm                │  │
│  │  MultiHeadSelfAttention │  │  ← Q/K 先过 QK-Norm → RoPE
│  │    SDPA (causal mask)   │  │
│  └───────────┬────────────┘  │
│       残差相加 │              │
│  ┌────────────▼───────────┐  │
│  │  RMSNorm                │  │
│  │  SwiGLU FFN / MoE       │  │  ← config.use_moe 切换
│  └───────────┬────────────┘  │
│       残差相加 │              │
└──────────────┼───────────────┘
       │
       ▼
    RMSNorm → lm_head → logits (vocab_size)
```

> 本仓库 `LM`（`src/models/lm/model.py:11`）：TokenEmbed → N×Block → RMSNorm → lm_head

---

## Q1. RMSNorm 和 LayerNorm 区别？为什么选 RMSNorm？

### 公式对比

| | LayerNorm | RMSNorm |
|---|---|---|
| 公式 | `(x - mean) / std * γ + β` | `x / sqrt(mean(x²) + eps) * γ` |
| 减均值 | ✅ 是 | ❌ 否 |
| 可学习参数 | γ, β | γ |

### 本仓库实现（`src/core/norm.py:5-15`）

```python
class RMSNorm(nn.Module):
    def forward(self, x):
        x = x.float()                              # 先转 float32 保精度
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).type_as(self.weight)  # 再转回原 dtype
```

### 为什么 RMSNorm 更好？

1. **计算效率**：省去均值计算，减少一次归一化操作
2. **训练稳定性**：LLaMA/GPT-NeoX 系列标配，实践证明足够
3. **精度技巧**：先转 float32 计算，再转回原 dtype——混合精度下的标准做法

> 面试点：为什么先转 float32？→ 避免 float16/bf16 下的数值溢出

---

## Q2. RoPE 是什么？为什么好？（`src/core/rope.py`）

### 公式推导

把 d_model 维向量按相邻两维配对，对第 m 个位置、第 i 对 (2i, 2i+1) 施加旋转角 θ_i·m：

```
θ_i = base^(-2i/d)   # base 即 rope_theta，默认 1e6

[cos(θ_i·m)  -sin(θ_i·m)]   [x_{2i}  ]
[sin(θ_i·m)   cos(θ_i·m)] · [x_{2i+1}]
```

### 为什么编码的是相对位置

旋转矩阵是正交阵，关键是**可加性**：旋转角相加 = 位置差。

```
(R_m q)ᵀ (R_n k) = qᵀ R_{n-m} k
```

即 Q@K 内积只依赖 **(n-m)** 这个相对位置 → 平移不变、长度外推好。

### 本仓库实现（`src/core/rope.py:5-37`）

```python
def precompute_freqs_cis(dim, max_position_embeddings=32768, base=1e6, rope_scaling=None):
    # 基础频率计算
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    
    # YaRN 扩展（可选）
    if rope_scaling is not None:
        # beta_fast=32, beta_slow=1, factor=16
        # 中频段频率缩放 + 注意力因子修正
        pass
    
    t = torch.arange(max_position_embeddings)
    freqs = torch.outer(t, inv_freq)
    
    # 复制一份用于 Q/K
    freqs_cos = torch.cat([freqs, freqs], dim=-1)
    freqs_sin = torch.cat([freqs, freqs], dim=-1)
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin):
    # 半维翻转 + 旋转
    q_rot = rotate_half(q)
    k_rot = rotate_half(k)
    q = q * freqs_cos + q_rot * freqs_sin
    k = k * freqs_cos + k_rot * freqs_sin
    return q.type_as(q_orig), k.type_as(k_orig)
```

### 对比：RoPE vs 绝对位置 Embedding

| | RoPE | 绝对位置 Embedding |
|---|---|---|
| 内积是否含相对位置 | ✅ 自动 | ❌ 需学 |
| 长度外推 | 好（可调整 base） | 差（超出训练长度即 OOV） |
| 额外参数 | 无 | 有 (context_length×d) |
| 代表模型 | LLaMA, GPT-NeoX | GPT-2, BERT |

### YaRN：长上下文扩展

当 `rope_scaling` 不为 None 时启用，支持：
- `factor=16`：上下文长度扩展倍数
- `original_max_position_embeddings=2048`：原始训练长度
- 中频段频率做 `1/factor` 缩放 + 注意力因子修正（NTK-aware）

> 面试点：YaRN 为什么比直接外推 RoPE 更好？→ 直接外推会导致高频位置编码失效，YaRN 通过分频段处理解决这个问题

---

## Q3. GQA 解决了什么？（`src/core/attention.py:10-55`）

### 三种注意力模式

| | MHA | GQA | MQA |
|---|---|---|---|
| Q 头数 | n_heads | n_heads | n_heads |
| KV 头数 | n_heads | n_kv_heads | 1 |
| 显存 | 高 | 中 | 低 |
| 质量 | 高 | 中 | 低 |
| 代表 | GPT-3 | LLaMA-2 | PaLM |

### 本仓库实现（`src/core/attention.py:13-16`）

```python
class Attention(nn.Module):
    def __init__(self, config):
        self.n_local_heads = config.num_attention_heads      # 8
        self.n_local_kv_heads = config.num_key_value_heads  # 4
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 2 倍复制
```

### repeat_kv 实现（`src/core/rope.py:33-37`）

```python
def repeat_kv(x, n_rep):
    # (bs, slen, num_kv_heads, head_dim) -> (bs, slen, num_heads, head_dim)
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(bs, slen, num_kv_heads, n_rep, head_dim).reshape(bs, slen, num_heads, head_dim)
```

> 面试点：GQA 的 KV cache 节省多少？→ 假设 8 Q 头 / 4 KV 头，KV cache 减半，推理显存显著降低

---

## Q4. QK-Norm 为什么有用？（`src/core/attention.py:23-24, 36`）

### 问题：Attention Logit 爆炸

在深层 Transformer 中，Q/K 的 norm 可能会越来越大，导致：
- softmax 进入饱和区
- 梯度消失
- 训练不稳定

### 解决方案：在 RoPE 前做 RMSNorm

```python
# src/core/attention.py:36
q = self.q_norm(self.q_proj(x))  # 先做 QK-Norm
k = self.k_norm(self.k_proj(x))
q, k = apply_rotary_pos_emb(q, k, freqs_cos, freqs_sin)  # 再做 RoPE
```

### 为什么在 RoPE 前？

RoPE 是旋转操作，不改变向量的 norm。如果在 RoPE 后做 QK-Norm，会破坏旋转角度的相对性。

> 面试点：QK-Norm 和 Pre-Norm 的区别？→ Pre-Norm 是对整个残差分支的输入做 Norm，QK-Norm 只对 Q/K 做 Norm，两者解决不同层面的稳定性问题

---

## Q5. SwiGLU vs ReLU FFN（`src/core/mlp.py:7-17`）

### 公式对比

| | ReLU FFN | SwiGLU |
|---|---|---|
| 公式 | `max(0, xW1)W2` | `silu(xW1) ⊗ (xW2)` |
| 门控 | 无 | 有（W2 作为 gate） |
| 参数量 | 2·d·d_ff | 3·d·d_ff (实际约 2·d·d_ff) |
| 效果 | 一般 | 更好 |

### 本仓库实现（`src/core/mlp.py:7-17`）

```python
class FeedForward(nn.Module):
    def __init__(self, config):
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]  # silu
    
    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

### FFN 容量公式

设 d_model=d，d_ff=f：
- 标准 FFN 参数：`2df`
- SwiGLU 参数：`3·(d·(2f/3))` ≈ `2df`（LLaMA 取 d_ff = 8d/3 配 SwiGLU）

> 面试点：为什么 SwiGLU 比 ReLU FFN 效果好？→ 门控结构让网络学"哪些信息该通过"，表达力更强、收敛更好

---

## Q6. Pre-Norm vs Post-Norm

### 结构对比

```
Pre-Norm（本仓库）:           Post-Norm（原 Transformer）:
x = x + Attn(RMSNorm(x))      x = RMSNorm(x + Attn(x))
x = x + FFN(RMSNorm(x))       x = RMSNorm(x + FFN(x))
```

### 为什么 Pre-Norm 更稳？

- **Post-Norm**：梯度需要经过 Norm 层才能回传，深层梯度易消失，需 warmup
- **Pre-Norm**：残差通道始终直通，梯度能无损回传，训练更稳、可省/减 warmup

### 本仓库实现（`src/core/block.py:8-24`）

```python
class Block(nn.Module):
    def forward(self, x, start_pos, freqs_cos, freqs_sin, mask=None):
        # Pre-Norm 架构
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cos, freqs_sin, mask)
        out = h + self.mlp(self.ffn_norm(h))  # self.mlp 可能是 FeedForward 或 MOEFeedForward
        return out
```

---

## Q7. Flash-Attention 为什么快？

### 核心思想：IO 感知

传统注意力需要物化完整的 N×N 注意力矩阵，显存 O(N²)。Flash-Attention 通过分块计算，避免物化完整矩阵。

### 本仓库的 Flash-Attention 条件（`src/core/attention.py:28, 44`）

```python
# 仅在以下条件使用 Flash Attention
if (seq_len > 1 and 
    (not self.causal or past_key_value is None) and 
    attention_mask is None):
    # 使用 F.scaled_dot_product_attention（PyTorch 2.0+ 内置 Flash Attention）
```

### 为什么需要这些条件？

1. `seq_len > 1`：单 token 无需注意力
2. `not self.causal or past_key_value is None`：Flash Attention 对 causal mask 支持有限
3. `attention_mask is None`：Flash Attention 不支持自定义 mask

> 面试点：Flash Attention 和标准 Attention 的计算量一样吗？→ 一样，只是 IO 优化，减少 HBM 读写

---

## Q8. 参数量估算（面试手算）

以 LLaMA-7B 量级为例（d=4096, layers=32, heads=32, d_ff=11008, vocab=32000）：

```
embedding : V·d           ≈ 32000·4096      ≈ 131M
per block  : 2·(d² + d·d_ff)  ≈ 2·(16.8M + 45.1M) ≈ 124M
all blocks: 32 · 124M      ≈ 3.97B
lm_head   : V·d            ≈ 131M (若共享则不计)
```

→ ≈ 6.7B，与公开 7B 吻合。手算时常忽略 layernorm/bias 小头。

### 本仓库默认参数（`src/models/lm/config.py`）

| 参数 | 默认值 |
|------|--------|
| hidden_size | 768 |
| num_hidden_layers | 8 |
| num_attention_heads | 8 |
| num_key_value_heads | 4 |
| intermediate_size | ceil(768 * π / 64) * 64 ≈ 3840 |
| vocab_size | 6400 |
| max_position_embeddings | 32768 |
| rope_theta | 1e6 |

---

## Q9. 权重绑定（Weight Tying）

### 本仓库实现（`src/models/lm/model.py:52, 59-60`）

```python
class LMForCausalLM(PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]  # 与 embed_tokens 绑定
    
    def __init__(self, config):
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # lm_head.weight 与 self.model.embed_tokens.weight 共享
```

### 为什么共享权重？

1. **省参数**：词表参数减半（vocab_size × hidden_size）
2. **语义对齐**：embedding 空间和 logits 空间在同一空间，语义更一致
3. **小模型更稳**：减少参数量有助于正则化

---

## Q10. 推理时的 KV Cache

### 核心思想

缓存已算的 K/V，每步只算新 token 的 Q 与已有 K/V 做注意力，避免重算。

### 本仓库实现（`src/core/attention.py:39-42`）

```python
def forward(self, x, start_pos, freqs_cos, freqs_sin, mask=None):
    # KV Cache 简单拼接实现
    if past_key_value is not None:
        xk = torch.cat([past_key_value[0], xk], dim=2)
        xv = torch.cat([past_key_value[1], xv], dim=2)
    past_key_value = (xk, xv)
```

### KV Cache 的显存占用

假设：
- batch_size=B, seq_len=S, num_layers=L
- num_kv_heads=H, head_dim=D
- 精度=fp16（2 bytes）

KV Cache 显存 = `2 × B × S × L × H × D × 2 bytes`

> 面试点：GQA 如何减少 KV Cache？→ KV 头数从 n_heads 减到 n_kv_heads，KV Cache 减少 n_heads/n_kv_heads 倍

---

## Q11. Logits to Keep 优化（`src/models/lm/model.py:65-66`）

### 问题

在生成时，我们只需要最后一个 token 的 logits，但标准实现会计算所有 token 的 logits。

### 本仓库优化

```python
def forward(self, input_ids, ..., logits_to_keep=1):
    # 仅计算最后 N 个 token 的 logits
    hidden_states = hidden_states[:, -logits_to_keep:]
    logits = self.lm_head(hidden_states)
```

### 显存节省

假设 vocab_size=64000，seq_len=32768：
- 不优化：`32768 × 64000 × 2 bytes ≈ 4GB`
- 优化后：`1 × 64000 × 2 bytes ≈ 128KB`

---

## Q12. MoE 设计模式（`src/core/mlp.py:20-49`）

### 关键设计：死 Expert 梯度保持

```python
class MOEFeedForward(nn.Module):
    def forward(self, x):
        # ... top-k 选择 ...
        
        # 死 expert 梯度保持技巧
        y[0, 0] += 0 * sum(p.sum() for p in self.experts.parameters())
        return y
```

### 为什么需要这个技巧？

在 DDP 分布式训练中，如果某个 expert 完全没被选中，它的参数就不会有梯度，导致 DDP 梯度同步死锁。`0 * sum(p)` 让这些参数仍然出现在计算图中，保持 DDP 通信闭环。

---

## Q13. 延迟 Buffer 初始化（`src/models/lm/model.py:31-33`）

### 问题

`torch.compile` 会重新初始化 buffer，导致 RoPE 的 freqs_cos/freqs_sin 变成零。

### 本仓库解决方案

```python
class LM(nn.Module):
    def forward(self, x, start_pos=0, freqs_cos=None, freqs_sin=None, mask=None):
        # 延迟 RoPE 初始化
        if freqs_cos is not None and freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(...)
```

### 为什么有效？

检查 `freqs_cos[0, 0] == 0` 可以检测 buffer 是否被重置，如果是则重新计算。

---

## Q14. 原子写入 Checkpoint（`src/utils/checkpoint.py:7-10`）

### 问题

训练过程中保存 checkpoint 时，如果中途崩溃，可能导致 checkpoint 损坏。

### 本仓库解决方案

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

## Q15. SkipBatchSampler（`src/utils/training.py:177-200`）

### 问题

分布式训练中断后，需要跳过已完成的 batch 继续训练。

### 本仓库解决方案

```python
class SkipBatchSampler:
    def __init__(self, dataset, batch_size, step):
        self.dataset = dataset
        self.batch_size = batch_size
        self.step = step  # 已完成的 step 数
    
    def __iter__(self):
        # 跳过前 step 个 batch
        indices = list(range(len(self.dataset)))
        indices = indices[self.step * self.batch_size:]
        # 返回剩余 batch
        ...
```

### 为什么需要这个？

DDP 训练中，每个 rank 可能中断在不同的 step，需要精确跳过已完成的 batch，避免重复训练。
