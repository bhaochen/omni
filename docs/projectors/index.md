# projectors/ · 模态桥接层

把 encoder 输出的模态特征投影到 LLM 隐藏维度（`hidden_size`），随后替换 `input_ids` 中的占位 token 位置。

| 文件 | 类 | 作用 |
| --- | --- | --- |
| `projectors/vision.py` | `MMVisionProjector` | 视觉特征 → LLM `hidden_size` |
| `projectors/audio.py` | `MMAudioProjector` | 音频特征 → LLM `hidden_size` |

## 注入范式

```
原始输入(图/音)
   └─ encoder      → 模态特征
        └─ projector → 投影到 hidden_size
             └─ 替换 input_ids 中的占位 token 位置的 embedding
                  └─ 进入共享 LM 主干
```

- 文本侧用特殊占位 token（`<|image_pad|>` / `<|audio_pad|>`）预留位置；
- `forward` 把投影特征写到这些位置，LLM 对文本/视觉/音频 token 一视同仁。
