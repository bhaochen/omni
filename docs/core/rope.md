# core/rope.py · 旋转位置编码 (RoPE)

## 原理

对 query/key 的每个二维子空间 $(x_{2i},x_{2i+1})$ 旋转角度 $\theta_i m$：

$$
\begin{pmatrix}q_{2i}\\ q_{2i+1}\end{pmatrix}
=
\begin{pmatrix}\cos m\theta_i & -\sin m\theta_i\\ \sin m\theta_i & \cos m\theta_i\end{pmatrix}
\begin{pmatrix}x_{2i}\\ x_{2i+1}\end{pmatrix}
$$

频率 $\theta_i=\text{base}^{-2i/d}$，$m$ 为位置下标。好处：相对位置 $m-n$ 只出现在旋转角度差里，天然带相对位置信息，且可外推。

## 代码

```python
def precompute_freqs_cis(dim, end, rope_base=1e6, rope_scaling=None):
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[:dim//2].float() / dim))
    if rope_scaling is not None:           # YaRN
        ...  # 按 ramp 对中段频率做 1/factor 缩放
    t = torch.arange(end)
    freqs = torch.outer(t, freqs).float()
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], -1) * attn_factor
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], -1) * attn_factor
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), -1)
    q_embed = (q*cos + rotate_half(q)*sin).to(q.dtype)
    k_embed = (k*cos + rotate_half(k)*sin).to(k.dtype)
    return q_embed, k_embed
```

- `freqs_cos/freqs_sin` 在 `LM.__init__` 里 `register_buffer(..., persistent=False)` 缓存，按 `start_pos:start_pos+seq_len` 切片取用（支持 KV-cache）。
- `cos/sin` 在最后一维 **拼接**（不是交替），所以 `rotate_half` 直接前后半交换即可。

## YaRN（长上下文扩展）

当 `rope_scaling` 为 YaRN 配置且 `end/orig_max>1` 时：

- 用 `inv_dim(b)=dim·ln(orig_max/(2πb))/(2ln base)` 找 `[β_fast,β_slow]` 频段；
- 对中间频段做 `freqs *= (1-ramp + ramp/factor)`（高频不动、低频压缩）；
- `attn_factor` 修正注意力缩放，保证 NTK 一致性。

## 要点（面试）

- RoPE 是**乘法旋转**，不增加参数量；Q/K 用**同一套** cos/sin。
- 与绝对位置编码相比，RoPE 的注意力分数只依赖相对距离 → 外推友好。
- 本实现用 YaRN 做长上下文（base=1e6，factor=16），属于「NTK-aware + 注意力缩放」组合。
