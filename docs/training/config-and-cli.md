# utils/config-and-cli.md · 配置与命令行

## 机制

`utils/training.apply_config(parser, default_config=None)` 让 **YAML 成为 argparse 默认值，CLI 显式参数覆盖 YAML**：

```python
def apply_config(parser, default_config=None):
    pre, _ = parser.parse_known_args()          # 先读出 --config
    config_path = getattr(pre, 'config', None) or default_config
    if config_path and os.path.exists(config_path):
        defaults = _load_yaml_config(config_path)
        parser.set_defaults(**defaults)          # 注入默认值
    return parser.parse_args()                   # 最终解析（CLI 覆盖）
```

`_load_yaml_config` 把 YAML 的 `model / train / paths` 三段**扁平化**为顶层键：
`model.hidden_size → hidden_size`，`train.epochs → epochs`，`paths.data_path → data_path` ……
这些键名与 trainer 的 `--hidden_size` / `--epochs` / `--data_path` 等 argparse 参数一一对应。

## 配置文件位置

```
configs/
├── lm/         # lm_pretrain.yaml / lm_full_sft.yaml / lm_pretrain_mini.yaml / lm_full_sft_mini.yaml / lm_pretrain_moe.yaml / lm_full_sft_moe.yaml
├── vlm/        # vlm.yaml / vlm_moe.yaml
└── vam/        # vam.yaml / vam_moe.yaml
checkpoint/
└── tokenizer/  # tokenizer.json / tokenizer_config.json
```

每个 YAML 例：

```yaml
model:
  hidden_size: 768
  num_hidden_layers: 8
  use_moe: 0
  vocab_size: 6400
train:
  epochs: 2
  batch_size: 16
  learning_rate: 1.0e-5
  from_weight: pretrain
paths:
  save_dir: checkpoint/lm
  data_path: dataset/sft.jsonl
```

## 配置如何真正生效

trainer 用 `LMConfig(**vars(args))`（或 `VLMConfig` / `VAMConfig`）构造模型配置，
因此 YAML 里**所有** `model` 字段（含 `vocab_size`、head 数、MoE 专家数、talker 层数等）
都会驱动模型结构，而非仅少数硬编码字段。

## 启动

```bash
python -m trainers.lm.full_sft --config configs/lm/lm_full_sft.yaml
python -m trainers.vlm.full_sft --config configs/vlm/vlm_moe.yaml
python -m trainers.vam.full_sft --config configs/vam/vam.yaml --epochs 10
python -m trainers.lm.train_tokenizer --data_path dataset/sft_t2t_mini.jsonl \
                                      --vocab_size 6400 --no_eval
```

## Tokenizer 训练

`trainers/lm/train_tokenizer.py` 仅供学习参考（MiniMind 已自带 tokenizer，重复训练会导致词表不统一）。
训练得到的 tokenizer 直接保存到 **`checkpoint/tokenizer/`**（与模型权重同目录），包含：

- `tokenizer.json` / `vocab.json` / `merges.txt`：BPE 词表
- `tokenizer_config.json`：special token、chat template 等配置

常用参数：

```bash
python -m trainers.lm.train_tokenizer --data_path dataset/sft_t2t_mini.jsonl \
                                      --vocab_size 6400 \
                                      --checkpoint_dir ../checkpoint \
                                      --no_eval
```

## 要点（面试）

- 设计亮点：**单一事实源（YAML）+ CLI 覆盖**，实验可复现、参数可微调。
- `set_defaults` 注入后，argparse 仍允许命令行覆盖 → 优先级 `CLI > YAML > 代码默认`。
- 扁平化约定要求 YAML 键名与 `--arg` 名一致，故 `trainers` 的参数命名需与 YAML 段对齐。
