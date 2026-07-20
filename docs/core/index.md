# core/ · 基础组件

`core/` 是与模态无关的纯 Transformer 组件，全部继承自 `nn.Module`，可被 `models/` 任意拼装。

| 文件 | 类 / 函数 | 作用 |
| --- | --- | --- |
| `norm.py` | `RMSNorm` | 均方根层归一化 |
| `rope.py` | `precompute_freqs_cis`, `apply_rotary_pos_emb`, `repeat_kv` | 旋转位置编码（含 YaRN） |
| `attention.py` | `Attention` | 带 QK-Norm + GQA 的注意力 |
| `mlp.py` | `FeedForward`, `MOEFeedForward` | SwiGLU FFN 与 MoE |
| `block.py` | `Block` | Pre-Norm Transformer 块 |

各组件细节见：

- [norm.md](norm.md) — RMSNorm 实现与数值技巧
- [rope.md](rope.md) — RoPE 原理、代码、YaRN 长上下文扩展
- [attention.md](attention.md) — GQA、QK-Norm、Flash-Attention 分支
- [mlp.md](mlp.md) — SwiGLU 与 MoE 路由、aux_loss
- [block.md](block.md) — Block 的残差结构与 MoE 切换
