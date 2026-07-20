# core/attention.py · Attention

## 代码（关键片段）

```python
self.q_proj = nn.Linear(h, n_heads*head_dim, bias=False)
self.k_proj = nn.Linear(h, n_kv_heads*head_dim, bias=False)
self.v_proj = nn.Linear(h, n_kv_heads*head_dim, bias=False)
self.o_proj = nn.Linear(n_heads*head_dim, h, bias=False)
self.q_norm = RMSNorm(head_dim)           # QK-Norm
self.k_norm = RMSNorm(head_dim)

def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
    xq,xk,xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
    xq,xk = self.q_norm(xq), self.k_norm(xk)
    cos,sin = position_embeddings
    xq,xk = apply_rotary_pos_emb(xq, xk, cos, sin)
    # concat past_kv (KV-cache)
    xq,xk,xv = (xq.T(1,2), repeat_kv(xk,n_rep).T(1,2), repeat_kv(xv,n_rep).T(1,2))
    if self.flash and ...:
        out = F.scaled_dot_product_attention(xq,xk,xv, is_causal=self.is_causal)
    else:
        scores = xq@xk.T / sqrt(head_dim)
        scores[:,:,:,-seq_len:] += causal_mask   # 上三角 -inf
        scores += (1-attn_mask)*-1e9
        out = softmax(scores)@xv
    return self.o_proj(out), past_kv
```

## 关键设计

1. **GQA（Grouped Query Attention）**
   - `n_rep = n_heads // n_kv_heads`；k/v 经 `repeat_kv` 沿 head 维复制。
   - 减少 KV 显存与计算，是 LLaMA-2/3 的标配。
2. **QK-Norm**
   - 对 q/k 每个 head（`head_dim` 维）做 RMSNorm，抑制注意力 logits 过大，提升稳定性（Qwen / MiniMind 采用）。
3. **Flash-Attention 分支**
   - 满足 `flash=True 且 seq_len>1 且 无 past 且 mask 全 1` 时走 `scaled_dot_product_attention`（IO 感知、省显存）。
   - 否则走手算 softmax 路径，手动加因果掩码与 padding 掩码。
4. **因果掩码**：`scores[:,:,:,-seq_len:].triu(1) += -inf`，只遮当前位置之后的 token。
5. **KV-Cache**：`past_key_value` 拼接历史 k/v，`start_pos` 用于 RoPE 频率切片。

## 要点（面试）

- 为什么 GQA？相比 MHA 省 KV 缓存、推理更快；相比 MQA 质量更好。
- QK-Norm 放在 RoPE **之前**还是**之后**？本实现在投影后、RoPE 前对 q/k 归一化。
- Flash-Attn 为什么快？分块计算、不物化完整 `N×N` 注意力矩阵，减少 HBM 读写。
- 因果mask 只在当前窗口 `[-seq_len:]` 加，配合增量解码的 past_kv。
