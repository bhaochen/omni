# 面试：推理优化深度

> 本仓库 `src/models/lm/model.py` 的推理实现，覆盖 KV Cache、采样策略、流式生成

## 0. 推理流程概览

```
输入 prompt
    │
    ▼
Token Embedding
    │
    ▼
┌──────────────────────────────┐
│  Prefill 阶段                │  处理整个 prompt
│  (一次性计算所有 token)       │
└──────────────┬───────────────┘
       │
       ▼
┌──────────────────────────────┐
│  Decode 阶段                 │  逐 token 生成
│  (KV Cache 加速)             │
└──────────────┬───────────────┘
       │
       ▼
    输出序列
```

---

## Q1. KV Cache 是什么？为什么快？

### 核心思想

缓存已算的 K/V，每步只算新 token 的 Q 与已有 K/V 做注意力，避免重算。

### 本仓库实现（`src/core/attention.py:39-42`）

```python
def forward(self, x, start_pos, freqs_cos, freqs_sin, mask=None):
    # 计算 Q/K/V
    xq = self.q_norm(self.q_proj(x))
    xk = self.k_norm(self.k_proj(x))
    xv = self.v_proj(x)
    
    # 应用 RoPE
    xq, xk = apply_rotary_pos_emb(xq, xk, freqs_cos, freqs_sin)
    
    # KV Cache 简单拼接实现
    if past_key_value is not None:
        xk = torch.cat([past_key_value[0], xk], dim=2)
        xv = torch.cat([past_key_value[1], xv], dim=2)
    past_key_value = (xk, xv)
    
    # 计算注意力
    attn_output = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=mask)
    return attn_output, past_key_value
```

### 显存节省

假设：
- batch_size=B, seq_len=S, num_layers=L
- num_kv_heads=H, head_dim=D
- 精度=fp16（2 bytes）

**无 KV Cache**：
- 每步计算量 = B × S² × L × H × D
- 显存 = B × S × L × H × D × 2 bytes

**有 KV Cache**：
- 每步计算量 = B × S × L × H × D
- 显存 = B × S × L × H × D × 2 bytes（但只需算一次）

> 面试点：KV Cache 为什么能加速？→ 避免重复计算 K/V，每步只需算新 token 的 Q

---

## Q2. Prefill vs Decode 阶段

### Prefill 阶段

- 处理整个 prompt
- 一次性计算所有 token 的 K/V
- 计算量大，但只需做一次

### Decode 阶段

- 逐 token 生成
- 每步只算新 token 的 Q
- 计算量小，但需要很多步

### 本仓库实现（`src/models/lm/model.py:73-113`）

```python
def generate(self, input_ids, max_new_tokens=200, temperature=0.6, top_k=5, top_p=0.8):
    for _ in range(max_new_tokens):
        # Prefill 阶段：处理整个 prompt
        if idx_cond.shape[1] > 1:
            logits, _ = self(idx_cond)
        # Decode 阶段：只处理最后一个 token
        else:
            logits, _ = self(idx_cond, start_pos=start_pos)
        
        # 采样
        logits = logits[:, -1, :] / temperature
        idx_next = self.sample(logits, top_k=top_k, top_p=top_p)
        
        # 更新序列
        idx_cond = torch.cat([idx_cond, idx_next], dim=1)
        start_pos += 1
```

> 面试点：为什么 Prefill 和 Decode 要分开处理？→ Prefill 可以并行处理所有 token，Decode 只能逐 token 处理

---

## Q3. Logits to Keep 优化（`src/models/lm/model.py:65-66`）

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

> 面试点：为什么可以只算最后 N 个？→ 生成时只需要最后一个 token 的 logits 来采样下一个 token

---

## Q4. 采样策略

### Top-k 采样

```python
def top_k_logits(logits, k):
    # 只保留概率最高的 k 个 token
    values, indices = torch.topk(logits, k)
    # 其他 token 设为 -inf
    logits[logits < values[:, -1:]] = float('-inf')
    return logits
```

### Top-p 采样（Nucleus Sampling）

```python
def top_p_logits(logits, p):
    # 按概率排序
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    
    # 移除累积概率超过 p 的 token
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    
    # 恢复原始顺序
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    logits[indices_to_remove] = float('-inf')
    return logits
```

### Temperature 采样

```python
def temperature_scale(logits, temperature):
    # temperature < 1: 分布更尖锐，更确定
    # temperature > 1: 分布更平滑，更随机
    return logits / temperature
```

> 面试点：Top-k 和 Top-p 的区别？→ Top-k 保留固定数量的 token，Top-p 保留累积概率达到 p 的 token

---

## Q5. Repetition Penalty

### 问题

生成时可能重复之前的 token，导致输出质量下降。

### 本仓库实现（`src/models/lm/model.py:73-113`）

```python
def generate(self, input_ids, ..., repetition_penalty=1.2):
    # 对已生成的 token 施加惩罚
    if input_ids.shape[1] > 1:
        # 计算每个 token 的出现次数
        for i in range(input_ids.shape[1]):
            token_id = input_ids[0, i].item()
            logits[0, token_id] /= repetition_penalty
```

### 为什么用除法？

- 除以惩罚系数，降低已生成 token 的概率
- 保持概率分布的相对顺序

> 面试点：repetition_penalty=1.0 表示什么？→ 无惩罚，等于没有使用 repetition penalty

---

## Q6. 流式生成（Streaming）

### 本仓库实现（`src/models/lm/model.py:73-113`）

```python
def generate(self, input_ids, ..., stream_callback=None):
    for _ in range(max_new_tokens):
        # ... 前向传播 ...
        
        # 流式回调
        if stream_callback:
            stream_callback(idx_next)
        
        # 更新序列
        idx_cond = torch.cat([idx_cond, idx_next], dim=1)
```

### 为什么需要流式生成？

1. **用户体验**：实时看到生成结果
2. **早停**：用户可以在生成完成前停止
3. **调试**：实时观察生成过程

> 面试点：流式生成如何实现？→ 通过回调函数，在每步生成后返回当前 token

---

## Q7. GQA 对推理的影响

### KV Cache 节省

假设：
- num_attention_heads = 8
- num_key_value_heads = 4
- head_dim = 64

**MHA（Multi-Head Attention）**：
- KV Cache = 2 × B × S × L × 8 × 64 × 2 bytes

**GQA（Grouped-Query Attention）**：
- KV Cache = 2 × B × S × L × 4 × 64 × 2 bytes

**节省**：50%

### 本仓库实现（`src/core/attention.py:13-16`）

```python
class Attention(nn.Module):
    def __init__(self, config):
        self.n_local_heads = config.num_attention_heads      # 8
        self.n_local_kv_heads = config.num_key_value_heads  # 4
        self.n_rep = self.n_local_heads // self.n_local_kv_heads  # 2 倍复制
```

> 面试点：GQA 如何减少 KV Cache？→ KV 头数从 n_heads 减到 n_kv_heads，KV Cache 减少 n_heads/n_kv_heads 倍

---

## Q8. Flash-Attention 对推理的影响

### 核心思想

IO 感知的注意力计算，避免物化完整的 N×N 注意力矩阵。

### 本仓库的 Flash-Attention 条件（`src/core/attention.py:28, 44`）

```python
if (seq_len > 1 and 
    (not self.causal or past_key_value is None) and 
    attention_mask is None):
    # 使用 Flash Attention
```

### 为什么有条件限制？

1. `seq_len > 1`：单 token 无需注意力
2. `not self.causal or past_key_value is None`：Flash Attention 对 causal mask 支持有限
3. `attention_mask is None`：Flash Attention 不支持自定义 mask

> 面试点：Flash-Attention 对推理有什么好处？→ 减少 HBM 读写，降低延迟

---

## Q9. 批量推理优化

### 问题

逐条推理效率低，需要批量处理。

### 解决方案

1. **Padding**：将不同长度的序列 padding 到统一长度
2. **Dynamic Batching**：根据序列长度动态调整 batch size
3. **Continuous Batching**：不等待整个 batch 完成，动态添加新请求

### 本仓库的批量推理（`src/models/lm/model.py:73-113`）

```python
def generate(self, input_ids, ...):
    # 支持 batch_size > 1
    for _ in range(max_new_tokens):
        logits, _ = self(idx_cond)
        # ... 采样 ...
```

> 面试点：Padding 的缺点是什么？→ 浪费计算资源，短序列需要 padding 到长序列长度

---

## Q10. 量化推理

### 问题

FP16 精度显存占用高，推理速度慢。

### 解决方案

1. **INT8 量化**：将权重从 FP16 量化到 INT8
2. **INT4 量化**：将权重从 FP16 量化到 INT4
3. **GPTQ**：基于二阶信息的量化方法
4. **AWQ**：激活感知的量化方法

### 本仓库的量化支持

```python
# 通过 config.dtype 控制精度
config.dtype = 'float16'  # FP16
config.dtype = 'bfloat16'  # BF16
```

> 面试点：量化会损失多少精度？→ 取决于量化方法和位数，INT8 通常损失很小，INT4 可能有明显损失

---

## Q11. 推理显存估算

### KV Cache 显存

假设：
- batch_size=B, seq_len=S, num_layers=L
- num_kv_heads=H, head_dim=D
- 精度=fp16（2 bytes）

KV Cache 显存 = `2 × B × S × L × H × D × 2 bytes`

### 模型参数显存

假设：
- hidden_size=d, num_layers=L
- 精度=fp16（2 bytes）

模型参数显存 = `12 × L × d × d × 2 bytes`（Q/K/V/O 四个投影矩阵）

### 总显存

总显存 ≈ 模型参数显存 + KV Cache 显存

> 面试点：如何估算推理显存？→ 模型参数显存 + KV Cache 显存

---

## Q12. 推理延迟估算

### Prefill 延迟

假设：
- batch_size=B, seq_len=S, hidden_size=d
- FLOPS = 2 × B × S² × d

### Decode 延迟

假设：
- batch_size=B, hidden_size=d
- FLOPS = 2 × B × S × d

### 总延迟

总延迟 ≈ Prefill 延迟 + Decode 延迟 × 生成长度

> 面试点：如何优化推理延迟？→ 使用 Flash-Attention、KV Cache、量化等方法

---

## Q13. 推理服务设计

### 问题

如何设计高并发的推理服务？

### 解决方案

1. **动态批处理**：根据请求到达时间动态组 batch
2. **请求排队**：使用消息队列管理请求
3. **负载均衡**：将请求分发到多个 GPU
4. **模型并行**：将模型拆分到多个 GPU

### 本仓库的推理服务（`src/serve/`）

```python
class RealtimeSession:
    def __init__(self, model):
        self.model = model
        self.vad = SileroVAD()  # 语音活动检测
    
    def process(self, audio):
        # 1. VAD 检测
        if not self.vad.detect(audio):
            return None
        
        # 2. 推理
        output = self.model.generate(audio)
        
        return output
```

> 面试点：如何提高推理吞吐量？→ 使用动态批处理、模型并行、量化等方法

---

## Q14. 推理与训练的区别

### 训练

- 需要反向传播
- 需要梯度存储
- 需要优化器状态
- 显存占用高

### 推理

- 只需要前向传播
- 不需要梯度存储
- 不需要优化器状态
- 显存占用低

### 本仓库的切换

```python
# 训练时
model.train()
loss = model(input_ids, labels=labels)

# 推理时
model.eval()
with torch.no_grad():
    logits = model(input_ids)
```

> 面试点：为什么推理时要 `torch.no_grad()`？→ 节省显存，避免存储梯度

---

## Q15. 推理优化的未来方向

### 当前瓶颈

1. **内存墙**：显存带宽限制推理速度
2. **计算墙**：GPU 计算能力限制吞吐量
3. **延迟墙**：逐 token 生成限制响应速度

### 未来方向

1. **投机采样**：用小模型预测，大模型验证
2. **模型并行**：将模型拆分到多个 GPU
3. **硬件优化**：使用专用推理芯片
4. **算法优化**：设计更高效的注意力机制

> 面试点：投机采样是什么？→ 用小模型快速生成候选，大模型验证并选择，提高生成速度
