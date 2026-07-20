# core/mlp.py · FFN 与 MoE

## SwiGLU FFN（`FeedForward`）

```python
self.gate_proj = nn.Linear(h, inter, bias=False)
self.up_proj   = nn.Linear(h, inter, bias=False)
self.down_proj = nn.Linear(inter, h, bias=False)
self.act_fn    = ACT2FN[config.hidden_act]     # 默认 silu

def forward(self, x):
    return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
```

公式（SwiGLU）：
$$\mathrm{FFN}(x)=\mathrm{down}\big(\mathrm{silu}(\mathrm{gate}(x))\odot \mathrm{up}(x)\big)$$

## MoE（`MOEFeedForward`）

```python
self.gate   = nn.Linear(h, num_experts, bias=False)
self.experts = ModuleList([FeedForward(config, moe_intermediate_size) for _ in range(num_experts)])

def forward(self, x):
    scores   = softmax(self.gate(x_flat), -1)
    topk_weight, topk_idx = topk(scores, k=num_experts_per_tok)
    if norm_topk_prob: topk_weight /= topk_weight.sum(-1, keepdim=True)   # 归一化
    y = zeros_like(x_flat)
    for i, expert in enumerate(self.experts):
        mask = (topk_idx == i)
        token_idx = mask.any(-1).nonzero().flatten()
        y.index_add_(0, token_idx, expert(x_flat[token_idx]) * weight)
    # 训练时累计 router 辅助损失
    load = one_hot(topk_idx, num_experts).float().mean(0)
    self.aux_loss = (load * scores.mean(0)).sum() * num_experts * router_aux_loss_coef
    return y
```

要点：

- **路由**：每个 token 经 `gate` 投影到专家 logits → softmax → 取 top-`k` 个专家（默认 k=1）。
- **`norm_topk_prob`**：把选中专家的权重归一化，保证输出尺度稳定。
- **负载均衡 (`aux_loss`)**：`Σ load_e · mean(scores)_e`，鼓励专家被均匀选中；系数 `router_aux_loss_coef`（默认 5e-4）只在训练时累加，推理为 0。
- **高效实现**：用 `index_add_` 把 token 分发给对应专家，避免显式循环展开大矩阵。
- **切换**：`Block` 中 `self.mlp = FeedForward(...) if not use_moe else MOEFeedForward(...)`，MoE 可插拔。

## 要点（面试）

- SwiGLU 相比原始 FFN（ReLU 两塔）多了 `up` 分支做门控，效果更优。
- MoE 的「稀疏」体现在：每个 token 只过 top-k 个专家，参数量大但单步计算量小。
- 为什么需要 aux_loss？防止 router 退化为只选少数专家（坍缩），保证负载均衡。
- `index_add_` 实现注意 `expert(x[token_idx]) * weight` 的广播与 dtype 对齐。
