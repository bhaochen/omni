# models/lm · LM 与 LMForCausalLM

## 类图

```
LM(nn.Module)                      # 纯主干
  ├─ embed_tokens
  ├─ layers: ModuleList[Block]
  ├─ norm: RMSNorm
  └─ freqs_cos / freqs_sin (buffer)

LMForCausalLM(PreTrainedModel, GenerationMixin)   # 可训练 / 可生成
  ├─ model: LM
  └─ lm_head: Linear(hidden, vocab)
```

## LM（主干）

```python
class LM(nn.Module):
    def __init__(self, config):
        self.embed_tokens = nn.Embedding(vocab, hidden)
        self.layers = nn.ModuleList([Block(l, config) for l in range(num_layers)])
        self.norm = RMSNorm(hidden)
        freqs_cos, freqs_sin = precompute_freqs_cis(...)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, ...):
        h = self.dropout(self.embed_tokens(input_ids))
        position_embeddings = (freqs_cos[start:], freqs_sin[start:])   # 按 past 长度切片
        presents = []
        for layer, past in zip(self.layers, past_key_values):
            h, present = layer(h, position_embeddings, past_key_value=past, ...)
            presents.append(present)
        h = self.norm(h)
        aux_loss = sum(l.mlp.aux_loss for l in self.layers if MoE)
        return h, presents, aux_loss
```

- `freqs_cos/sin` 用 `persistent=False` buffer 缓存，不进 state_dict；首次前向若 buffer 为 0 会重新计算并搬到设备。
- `aux_loss` 仅在 MoE 时非 0，供 `LMForCausalLM` 加到总损失。

## LMForCausalLM（带 head + 损失 + 生成）

```python
class LMForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = LMConfig
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        self.model = LM(config)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        if config.tie_word_embeddings:
            self.model.embed_tokens.weight = self.lm_head.weight   # 权重绑定

    def forward(self, input_ids, labels=None, logits_to_keep=0, ...):
        h, past, aux_loss = self.model(...)
        logits = self.lm_head(h[:, slice(-logits_to_keep, None)])
        loss = CE(logits[:, :-1], labels[:, 1:], ignore_index=-100) if labels else None
        return MoeCausalLMOutputWithPast(loss, aux_loss, logits, ...)

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens=8192, temperature=0.85,
                 top_p=0.85, top_k=50, repetition_penalty=1.0, ...):
        # 自回归循环：每次只喂新 token，拼接 past_key_values
        # 支持 top_k / top_p 截断、repetition_penalty、eos 提前停止、streamer
```

## 要点（面试）

1. **权重绑定 (weight tying)**：`lm_head` 与 `embed_tokens` 共享权重，省一半词表参数；`_tied_weights_keys` 让 `save_pretrained` 只存一份。
2. **`logits_to_keep`**：生成时只算最后若干 token 的 logits，省算力。
3. **损失平移**：用 `logits[:, :-1]` 与 `labels[:, 1:]` 对齐，标准next-token预测。
4. **`generate` 自回归**：增量解码 + KV-cache，每次只 forward 新 token；`repetition_penalty` 对历史 token 降权防重复。
5. 继承 `PreTrainedModel` → 可直接用 `from_pretrained` / `save_pretrained` / `generate`（HF 生态）。
