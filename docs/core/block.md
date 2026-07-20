# core/block.py · Block

## 代码

```python
class Block(nn.Module):
    def __init__(self, layer_id, config: LMConfig):
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(hidden_size, eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps)
        self.mlp = FeedForward(config) if not config.use_moe else MOEFeedForward(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        residual = hidden_states
        h, present = self.self_attn(self.input_layernorm(hidden_states), position_embeddings, past_key_value, use_cache, attention_mask)
        h = h + residual
        h = h + self.mlp(self.post_attention_layernorm(h))
        return h, present
```

## 结构

Pre-Norm Transformer 块（LLaMA 风格）：

```
x ─┬─► RMSNorm ─► Attention ─────────────┐
  │                                       + ─► x
  └───────────────────────────────────────┘
x ─┬─► RMSNorm ─► MLP ───────────────────┐
  │                                       + ─► 输出
  └───────────────────────────────────────┘
```

- **残差直接相加**，归一化在子层**之前**（Pre-Norm），深层训练更稳定。
- `mlp` 依据 `config.use_moe` 在稠密 / 稀疏间切换，对上层透明。
- `present` 返回当前层 KV（用于 cache）。

## 要点（面试）

- Pre-Norm vs Post-Norm：Pre-Norm 残差路径无归一化，梯度更易回传，现代 LLM 主流。
- 为什么是 `input_layernorm` / `post_attention_layernorm` 两个？分别归一化 attention 与 mlp 的输入。
- MoE 在 Block 层切换，意味着**同一层要么全稠密、要么全 MoE**，没有混合层。
