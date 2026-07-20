# models/ · 模型拼装

按模态能力分三个子包，每个含 `config.py` + `model.py`：

| 子包 | 主干类 | 配置类 | 父类 |
| --- | --- | --- | --- |
| `models/lm` | `LM` + `LMForCausalLM` | `LMConfig` | `PreTrainedModel` |
| `models/vlm` | `VLM` | `VLMConfig(LMConfig)` | `LMForCausalLM` |
| `models/vam` | `VAM` | `VAMConfig(LMConfig)` | `LMForCausalLM` |

继承链：`VLM` / `VAM` → `LMForCausalLM` → `PreTrainedModel` + `GenerationMixin`。
因此三者**共享 `LM` 主干、`lm_head`、`generate()`**，多模态只在 `forward` 里额外注入视觉/音频特征。

细节：

- [lm.md](lm.md) — `LM`（纯组件堆叠）与 `LMForCausalLM`（加 head、损失、生成）
- [vlm.md](vlm.md) — 视觉特征如何拼接到 LLM
- [vam.md](vam.md) — 全模态：视觉 + 音频 + Talker 语音生成
