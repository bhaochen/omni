# Omni

Omni 是一个以 **多模态 (omni)** 为目标的 LLM 训练 / 推理框架，已完整集成
[MiniMind](https://github.com/jingyaogong/minimind)（纯文本）、
[miniMind-V](https://github.com/jingyaogong/minimind-v)（视觉多模态）与
[miniMind-O](https://github.com/jingyaogong/minimind-o)（语音 / 全模态）三套代码。

项目采用标准 `src/` 布局（`pip install -e .` 即可安装），按
**core（组件）/ models（拼装）/ encoders（模态编码器）/ projectors（桥接层）** 分层。

## 设计分层

- `core/`：可复用模型**纯底层组件**，按层级细分为独立模块——
  `norm.py`（`RMSNorm`）、`rope.py`（`precompute_freqs_cis` / `apply_rotary_pos_emb` / `repeat_kv`）、
  `attention.py`（`Attention`）、`mlp.py`（`FeedForward` / `MOEFeedForward`）、
  `block.py`（`Block`）。
- `models/`：把 `core` 组件**拼装**成成品模型，按模态能力分为三个子包（每个含 `config.py` 配置 + `model.py` 建模）：
  - `models/lm/`：纯文本——`LMConfig` + `LMForCausalLM`（主干 `LM`）
  - `models/vlm/`：文本 + 视觉——`VLMConfig` + `VLM`
  - `models/vam/`：文本 + 语音/全模态——`VAMConfig` + `VAM`（含 `TalkerModule`）
- `encoders/`：外部模态编码器，按模态分目录——`vision/`（SigLIP）、`audio/`（SenseVoice）。
- `projectors/`：把 encoder 输出**桥接**到 LLM 隐藏维度的拼接层（`MMVisionProjector`、`MMAudioProjector`）。
- `serve/`：实时语音会话工程层（`SileroVAD`、`RealtimeSession`）。
- `trainers/`：训练脚本，按模态分 `lm/` `vlm/` `vam/` 子包。
- `dataset/`：数据集（Pretrain / SFT / DPO / RLAIF / Agent / VLM / VAM），每类一个文件。
- `utils/`：训练与多模态工具。

## 目录结构

```
src/
├── core/               # 模型底层组件（按层级拆分）
│   ├── norm.py         #   RMSNorm
│   ├── rope.py         #   precompute_freqs_cis / apply_rotary_pos_emb / repeat_kv
│   ├── attention.py    #   Attention
│   ├── mlp.py          #   FeedForward / MOEFeedForward
│   └── block.py        #   Block
├── models/             # 模型拼装（按模态能力分子包）
│   ├── lm/             #   纯文本
│   │   ├── config.py   #     LMConfig
│   │   ├── model.py    #     LMForCausalLM + LM 主干
│   │   └── lora.py     #     LoRA 注入 / 保存 / 合并（作用于 LM 主干）
│   ├── vlm/            #   文本 + 视觉
│   │   ├── config.py   #     VLMConfig
│   │   └── model.py    #     VLM
│   └── vam/            #   文本 + 语音/全模态
│       ├── config.py   #     VAMConfig
│       └── model.py    #     VAM + TalkerModule
├── encoders/           # 多模态编码器（按模态分目录）
│   ├── vision/        #   SiglipVisionEncoder
│   └── audio/         #   SenseVoiceAudioEncoder
├── projectors/         # 多模态桥接层
│   ├── vision.py      #   MMVisionProjector
│   └── audio.py       #   MMAudioProjector
├── trainers/           # 训练脚本（按模态分 lm / vlm / vam）
│   ├── lm/            #   pretrain / full_sft / lora / dpo / distillation / ppo / grpo / agent
│   │                 #   + rollout_engine / train_tokenizer
│   ├── vlm/          #   pretrain / full_sft
│   └── vam/          #   full_sft
├── dataset/           # 数据集（每类一个文件）
│   ├── pretrain.py / sft.py / dpo.py / rlaif.py / agent_rl.py / vlm.py / vam.py
│   └── common.py     #   共享辅助函数
├── utils/            # 工具
│   ├── training.py    # get_lr / init_model / lm_checkpoint / SkipBatchSampler / apply_config
│   ├── multimodal.py  # init_vlm_model / vlm_checkpoint / init_omni_model / omni_checkpoint
│   ├── distributed.py # 分布式初始化
│   └── checkpoint.py  # checkpoint 读写辅助
├── serve/            # 实时语音会话（SileroVAD / RealtimeSession）
configs/
├── model/            # lm / lm_moe / vlm / vlm_moe / vam / vam_moe 训练配置
└── tokenizer/        # tokenizer.json / tokenizer_config.json
trainer/              # 根目录可直接运行的训练入口（默认加载对应 configs/model/*.yaml）
├── lm.py            # python trainer/lm.py  -> configs/model/lm.yaml
├── vlm.py           # python trainer/vlm.py -> configs/model/vlm.yaml
└── vam.py           # python trainer/vam.py -> configs/model/vam.yaml
scripts/              # 推理 / 服务 / 转换脚本
├── eval_llm.py      # 命令行推理与对话
├── eval_vlm.py      # 视觉多模态推理
├── eval_vam.py      # 全模态推理
├── serve_openai_api.py # OpenAI 兼容 API 服务
├── omni_web_demo.py # 网页演示（含实时语音）
├── eval_toolcall.py # 工具调用评测
└── convert_model.py # torch <-> transformers 权重互转
```

## 安装

```bash
pip install -e .
# 可选依赖：RL 训练 / API 服务 / 演示
pip install -e ".[rl,serve,demo]"
```

## 快速开始

### 训练（YAML 驱动）

根目录 `trainer/` 提供可直接运行的入口，默认加载 `configs/model/` 下对应的 YAML，
也可通过 `--config` 指定其它配置，任意 CLI 参数都能覆盖 YAML 中的默认值：

```bash
cd trainer
python lm.py                       # 使用 configs/model/lm.yaml
python vlm.py --config ../configs/model/vlm_moe.yaml
python vam.py --epochs 5           # 在 vam.yaml 基础上覆盖单字段
```

### 推理 / 对话

```bash
python scripts/eval_llm.py --load_from ../model --weight full_sft
```

## 配置说明

`configs/model/*.yaml` 分为 `model` / `train` / `paths` 三段，由 `utils.training.apply_config`
注入为 argparse 默认值；CLI 显式传参优先级更高。训练产出保存在 `checkpoint/` 目录。

## 与原 MiniMind 的差异

- 去除 `omni` 中间包，模块直接置于 `src/` 下（`core` / `models` / `trainers` / `dataset` / `utils` …），统一绝对导入；
- 类名统一：`MiniMindModel→LM`、`MiniMindVLM→VLM`、`MiniMindOmni→VAM`、`MiniMindConfig→LMConfig`、
  `MiniMindForCausalLM→LMForCausalLM`、`OmniConfig→VAMConfig`；
- `trainer_utils.py` 拆分为 `utils/training.py`、`utils/distributed.py`、`utils/checkpoint.py`；
- 训练脚本暴露 `main(default_config=None)`，既可由 `python -m trainers.<mod>` 调用，也可由根 `trainer/*.py` 调用；
- `dataset/` 按数据集类型拆分为独立文件，供多模态扩展。
