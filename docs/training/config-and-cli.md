# training/config-and-cli.md · 配置与命令行

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
├── model/      # lm / lm_moe / vlm / vlm_moe / vam / vam_moe .yaml
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
bash runs/lm.sh                                   # 默认 configs/model/lm.yaml
bash runs/vlm.sh --config configs/model/vlm_moe.yaml
bash runs/vam.sh --epochs 10                       # 覆盖单字段
```

## 要点（面试）

- 设计亮点：**单一事实源（YAML）+ CLI 覆盖**，实验可复现、参数可微调。
- `set_defaults` 注入后，argparse 仍允许命令行覆盖 → 优先级 `CLI > YAML > 代码默认`。
- 扁平化约定要求 YAML 键名与 `--arg` 名一致，故 `trainers` 的参数命名需与 YAML 段对齐。
