# 面试：MoE 深度

> 本仓库 `src/core/mlp.py` 的 MoE 实现，覆盖路由机制、负载均衡、训练技巧

## 0. MoE 架构图

```
输入 x
    │
    ▼
┌─────────────┐
│   Router    │  softmax gate → top-k 选择
└──────┬──────┘
       │
       ▼
┌──────────────────────────────┐
│  Expert 0  │  Expert 1  │ ... │  每个 Expert 是独立的 FFN
└──────────────────────────────┘
       │
       ▼
    加权求和 → 输出
```

---

## Q1. MoE 的稀疏指什么？

### 核心思想

每 token 只过 top-k 个专家（默认 k=1），参数量大但单步计算量小。

### 本仓库实现（`src/core/mlp.py:20-49`）

```python
class MOEFeedForward(nn.Module):
    def __init__(self, config):
        self.num_experts = config.num_experts  # 默认 4
        self.num_experts_per_tok = config.num_experts_per_tok  # 默认 1
        
        # 每个 expert 是独立的 FeedForward
        self.experts = nn.ModuleList([
            FeedForward(config) for _ in range(self.num_experts)
        ])
        
        # Router: 将 hidden_size 映射到 num_experts
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
    
    def forward(self, x):
        # 1. 计算路由概率
        router_logits = self.gate(x)  # (batch, seq_len, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # 2. Top-k 选择
        topk_probs, topk_indices = torch.topk(router_probs, self.num_experts_per_tok, dim=-1)
        
        # 3. 加权求和
        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (topk_indices == i).any(dim=-1)
            if mask.any():
                expert_output = expert(x[mask])
                output[mask] += topk_probs[mask] * expert_output
        
        return output
```

> 面试点：为什么每 token 只过 1 个专家？→ 计算量小，但参数量大，适合大规模模型

---

## Q2. 为什么需要 aux_loss？

### 问题：Expert 坍缩

如果不加约束，router 可能只选少数专家，导致：
- 大部分专家闲置
- 参数利用率低
- 训练不稳定

### 解决方案：辅助损失（`src/core/mlp.py:44-46`）

```python
def aux_loss(router_probs, topk_indices):
    # 1. 计算每个 expert 的负载
    load = (topk_indices == torch.arange(num_experts).view(1, 1, -1)).float().sum(dim=[0, 1])
    
    # 2. 计算每个 expert 的平均路由概率
    scores = router_probs.mean(dim=[0, 1])
    
    # 3. 辅助损失：鼓励负载均衡
    aux_loss = (load * scores).sum() * num_experts * coef  # coef=5e-4
    
    return aux_loss
```

### 为什么用 `load * scores`？

- **load**：实际被选中的次数
- **scores**：平均路由概率
- **乘积**：鼓励两者一致，即负载高的 expert 路由概率也高

> 面试点：如果不用 aux_loss 会怎样？→ Expert 坍缩，大部分专家闲置，参数利用率低

---

## Q3. 死 Expert 梯度保持（`src/core/mlp.py:42-43`）

### 问题

在 DDP 分布式训练中，如果某个 expert 完全没被选中，它的参数就不会有梯度，导致 DDP 梯度同步死锁。

### 解决方案

```python
class MOEFeedForward(nn.Module):
    def forward(self, x):
        # ... top-k 选择 ...
        
        # 死 expert 梯度保持技巧
        y[0, 0] += 0 * sum(p.sum() for p in self.experts.parameters())
        return y
```

### 为什么用 `0 * sum(p)`？

- 让未被选中的 expert 仍然出现在计算图中
- 保持 DDP 通信闭环
- 实际值为 0，不影响 loss

> 面试点：为什么不用 `.requires_grad = True`？→ 那只是让参数可训练，但不会出现在计算图中，`0 * sum(p)` 才能保证计算图连通

---

## Q4. 本仓库 MoE 怎么切换？

### Block 中的切换（`src/core/block.py:14`）

```python
class Block(nn.Module):
    def __init__(self, config):
        # 根据 config.use_moe 切换 FeedForward / MOEFeedForward
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)
```

### aux_loss 累加（`src/core/mlp.py:44-46`）

```python
def forward(self, x):
    # ... 前向传播 ...
    
    # aux_loss 在 MLP.forward 内累加
    if self.training and self.config.use_moe:
        aux_loss = aux_loss(router_probs, topk_indices)
    else:
        aux_loss = 0
    
    return output, aux_loss
```

### 主模型汇总（`src/models/lm/model.py:31-33`）

```python
class LM(nn.Module):
    def forward(self, x, ...):
        aux_loss = 0
        for layer in self.layers:
            x, layer_aux_loss = layer(x, ...)
            aux_loss += layer_aux_loss
        
        return x, aux_loss
```

> 面试点：aux_loss 为什么在 MLP.forward 内累加？→ 每层独立计算，避免跨层依赖

---

## Q5. MoE vs Dense 的参数量对比

### 参数量计算

假设：
- hidden_size = 768
- intermediate_size = 3840
- num_experts = 4

**Dense FFN**：
- 参数量 = 2 × 768 × 3840 = 5.9M

**MoE FFN**：
- 参数量 = 4 × 2 × 768 × 3840 = 23.6M
- 每 token 计算量 = 2 × 768 × 3840 = 5.9M（只过 1 个 expert）

### 对比

| | Dense | MoE |
|---|---|---|
| 参数量 | 5.9M | 23.6M |
| 每 token 计算量 | 5.9M | 5.9M |
| 显存占用 | 低 | 高 |
| 表达能力 | 低 | 高 |

> 面试点：MoE 为什么适合大规模模型？→ 参数量大但计算量小，可以提升模型容量而不增加计算成本

---

## Q6. Router 的设计选择

### 本仓库的 Router（`src/core/mlp.py:31-32`）

```python
class MOEFeedForward(nn.Module):
    def __init__(self, config):
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
    
    def forward(self, x):
        router_logits = self.gate(x)  # 线性投影
        router_probs = F.softmax(router_logits, dim=-1)  # softmax 归一化
```

### 为什么用线性投影？

1. **简单高效**：只增加 hidden_size × num_experts 参数
2. **可训练**：通过反向传播学习路由策略
3. **无 bias**：减少参数量，避免过拟合

### 为什么用 softmax？

1. **概率分布**：确保所有 expert 的权重和为 1
2. **可微分**：支持反向传播
3. **top-k 选择**：方便选择概率最高的 expert

> 面试点：如果用其他归一化方法会怎样？→ softmax 是最常用的选择，其他方法（如 L1 归一化）可能训练不稳定

---

## Q7. MoE 的训练稳定性

### 问题

MoE 训练比 Dense 更难，因为：
- Router 容易坍缩
- Expert 负载不均衡
- 梯度不稳定

### 本仓库的解决方案

1. **辅助损失**：鼓励负载均衡
2. **死 expert 梯度保持**：保持 DDP 通信闭环
3. **Expert 初始化**：用 Dense FFN 的权重初始化

### Expert 初始化（`src/core/mlp.py:20-25`）

```python
class MOEFeedForward(nn.Module):
    def __init__(self, config):
        # 每个 expert 用相同的初始化
        self.experts = nn.ModuleList([
            FeedForward(config) for _ in range(self.num_experts)
        ])
```

> 面试点：为什么用相同的初始化？→ 避免初始路由偏向某个 expert，让训练更稳定

---

## Q8. MoE 的推理优化

### 问题

推理时，每个 token 只过 1 个 expert，但需要加载所有 expert 的参数到显存。

### 解决方案

1. **Expert 并行**：不同 expert 放在不同 GPU 上
2. **Expert 卸载**：将不常用的 expert 卸载到 CPU
3. **Expert 量化**：降低 expert 的精度

### 本仓库的推理实现（`src/core/mlp.py:35-45`）

```python
def forward(self, x):
    # 推理时只过 1 个 expert
    if not self.training:
        topk_probs, topk_indices = torch.topk(router_probs, 1, dim=-1)
        
        output = torch.zeros_like(x)
        for i, expert in enumerate(self.experts):
            mask = (topk_indices == i).any(dim=-1)
            if mask.any():
                output[mask] = expert(x[mask])
        
        return output
```

> 面试点：为什么推理时不需要 aux_loss？→ 推理时不需要反向传播，aux_loss 只在训练时使用

---

## Q9. MoE 的显存占用

### 参数量

假设：
- hidden_size = 768
- intermediate_size = 3840
- num_experts = 4
- 精度 = fp16（2 bytes）

**Dense FFN**：
- 显存 = 5.9M × 2 bytes = 11.8MB

**MoE FFN**：
- 显存 = 23.6M × 2 bytes = 47.2MB

### KV Cache

MoE 不影响 KV Cache，因为 KV Cache 只与注意力层相关。

> 面试点：MoE 为什么显存占用高？→ 参数量大，需要加载所有 expert 的参数

---

## Q10. MoE 的适用场景

### 适合 MoE 的场景

1. **大规模模型**：参数量大但计算量小
2. **多任务学习**：不同 expert 可以学习不同任务
3. **稀疏激活**：每 token 只过部分 expert

### 不适合 MoE 的场景

1. **小规模模型**：参数量增加但计算量不变，性价比低
2. **密集计算**：每 token 需要所有参数
3. **低延迟推理**：需要加载所有 expert，延迟高

> 面试点：什么时候应该用 MoE？→ 模型规模大（>10B），且需要提升容量而不增加计算成本

---

## Q11. MoE vs 其他高效方法

### 对比

| | MoE | LoRA | 蒸馏 |
|---|---|---|---|
| 原理 | 稀疏激活 | 低秩分解 | 知识迁移 |
| 参数量 | 大 | 小 | 小 |
| 计算量 | 小 | 小 | 小 |
| 适用场景 | 大规模模型 | 微调 | 压缩模型 |

### 本仓库的组合使用

- **MoE**：提升模型容量
- **LoRA**：微调时减少参数量
- **蒸馏**：压缩模型大小

> 面试点：MoE 和 LoRA 可以一起用吗？→ 可以，MoE 提升容量，LoRA 减少微调参数，两者互补

---

## Q12. MoE 的负载均衡指标

### 指标定义

1. **Expert 负载**：每个 expert 被选中的比例
2. **负载方差**：expert 负载的方差，越小越均衡
3. **最大负载**：被选中次数最多的 expert

### 本仓库的辅助损失（`src/core/mlp.py:44-46`）

```python
def aux_loss(router_probs, topk_indices):
    load = (topk_indices == torch.arange(num_experts).view(1, 1, -1)).float().sum(dim=[0, 1])
    scores = router_probs.mean(dim=[0, 1])
    aux_loss = (load * scores).sum() * num_experts * coef
    return aux_loss
```

> 面试点：如何衡量负载均衡？→ 用辅助损失的值，越小越均衡

---

## Q13. MoE 的 Router 训练技巧

### 问题

Router 容易过拟合，导致：
- 路由策略不稳定
- Expert 负载不均衡

### 解决方案

1. **Dropout**：在 router 输出上加 dropout
2. **Label Smoothing**：对 router logits 做 label smoothing
3. **Warmup**：训练初期逐渐增加 router 的学习率

### 本仓库的实现（`src/core/mlp.py:31-32`）

```python
class MOEFeedForward(nn.Module):
    def __init__(self, config):
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
    
    def forward(self, x):
        router_logits = self.gate(x)
        router_probs = F.softmax(router_logits, dim=-1)
        # 没有使用额外的技巧，依赖辅助损失
```

> 面试点：Router 为什么容易过拟合？→ Router 参数少，容易记住训练数据的路由模式

---

## Q14. MoE 的 Expert 选择策略

### Top-k 选择

```python
topk_probs, topk_indices = torch.topk(router_probs, self.num_experts_per_tok, dim=-1)
```

### 为什么用 Top-k？

1. **简单高效**：只选择概率最高的 k 个 expert
2. **可微分**：支持反向传播
3. **可控稀疏度**：通过 k 控制稀疏程度

### 其他选择策略

1. **Random**：随机选择 expert
2. **Threshold**：选择概率超过阈值的 expert
3. **Gumbel-Softmax**：用 Gumbel-Softmax 采样

> 面试点：为什么不用 Random？→ Random 不可微分，无法反向传播

---

## Q15. MoE 的 Expert 初始化

### 问题

Expert 初始化不当会导致：
- 路由偏向某个 expert
- 训练不稳定

### 本仓库的解决方案（`src/core/mlp.py:20-25`）

```python
class MOEFeedForward(nn.Module):
    def __init__(self, config):
        # 每个 expert 用相同的初始化
        self.experts = nn.ModuleList([
            FeedForward(config) for _ in range(self.num_experts)
        ])
```

### 为什么用相同的初始化？

- 避免初始路由偏向某个 expert
- 让训练更稳定
- 所有 expert 从同一起点开始学习

> 面试点：如果用不同的初始化会怎样？→ 路由可能偏向某个 expert，导致其他 expert 闲置
