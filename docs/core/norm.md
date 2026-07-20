# core/norm.py · RMSNorm

## 代码

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)
```

## 公式

$$
\mathrm{RMS}(x)=\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2+\varepsilon},\qquad
y=\frac{x}{\mathrm{RMS}(x)}\odot \gamma
$$

## 要点（面试）

- 相比 LayerNorm **省去均值中心化**，只做缩放，参数更少、更稳定。
- 在 **float32** 里算归一化再 `.type_as(x)` 还原，避免低精度下数值爆炸。
- `gamma`（即 `weight`）初始化为 1，等价于先不做缩放。
- LLaMA / MiniMind 全程用 RMSNorm，不使用 bias。
