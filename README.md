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
├── lm/               # 纯文本 LM 配置（pretrain / full_sft / MoE / mini）
├── vlm/              # 视觉多模态 VLM 配置
└── vam/              # 全模态 VAM 配置
checkpoint/
└── tokenizer/        # tokenizer.json / tokenizer_config.json（由 train_tokenizer.py 生成）
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
uv sync --no-default-groups --no-install-project
pip install -e .
# 可选依赖：RL 训练 / API 服务 / 演示
pip install -e ".[rl,serve,demo]"
```

## 快速开始

### 训练（YAML 驱动）

训练入口统一为 `python -m trainers.<包>.<脚本>`，通过 `--config` 指定 YAML 配置，
任意 CLI 参数都能覆盖 YAML 中的默认值：

#### 纯文本 LM

```bash
# 预训练（从头训练语言模型）
python -m trainers.lm.pretrain --config configs/lm/lm_pretrain.yaml

# 全量 SFT（以预训练权重初始化，指令微调）
python -m trainers.lm.full_sft --config configs/lm/lm_full_sft.yaml

# 训练 tokenizer
python -m trainers.lm.train_tokenizer --data_path dataset/lm/sft_t2t_mini.jsonl \
                                      --vocab_size 6400 \
                                      --checkpoint_dir ./checkpoint \
                                      --no_eval

# LoRA 微调
python -m trainers.lm.lora_sft --config configs/lm/lm_full_sft.yaml

# DPO / PPO / GRPO 偏好对齐
python -m trainers.lm.dpo   --config configs/lm/lm_full_sft.yaml
python -m trainers.lm.ppo   --config configs/lm/lm_full_sft.yaml
python -m trainers.lm.grpo  --config configs/lm/lm_full_sft.yaml

# 知识蒸馏
python -m trainers.lm.distill --teacher <teacher_path> --config configs/lm/lm_full_sft.yaml

# MoE 变体
python -m trainers.lm.full_sft --config configs/lm/lm_full_sft_moe.yaml
python -m trainers.lm.pretrain --config configs/lm/lm_pretrain_moe.yaml

# Mini 变体（快速验证用，h=128, L=4, ~14min pretrain）
python -m trainers.lm.pretrain --config configs/lm/lm_pretrain_mini.yaml
python -m trainers.lm.full_sft --config configs/lm/lm_full_sft_mini.yaml
```

#### 视觉多模态 VLM

```bash
# 预训练（视觉模态对齐）
python -m trainers.vlm.pretrain --config configs/vlm/vlm_pretrain.yaml
python -m trainers.vlm.pretrain --config configs/vlm/vlm_pretrain_moe.yaml   # MoE 变体

# 全量 SFT
python -m trainers.vlm.full_sft --config configs/vlm/vlm_sft.yaml
python -m trainers.vlm.full_sft --config configs/vlm/vlm_sft_moe.yaml   # MoE 变体

# Mini 变体（快速验证用，h=768, L=8, ~1h on 4060）
python -m trainers.vlm.full_sft --config configs/vlm/vlm_sft_mini.yaml
python -m trainers.vlm.full_sft --config configs/vlm/vlm_sft_mini_resume.yaml   # 续训
```

#### 全模态 VAM（文本 + 视觉 + 语音）

```bash
# 全量 SFT
python -m trainers.vam.full_sft --config configs/vam/vam.yaml
python -m trainers.vam.full_sft --config configs/vam/vam_moe.yaml   # MoE 变体
```

#### 常用覆盖参数

```bash
# 覆盖任意 YAML 字段
python -m trainers.lm.full_sft --config configs/lm/lm_full_sft.yaml \
                               --epochs 3 --batch_size 8 --learning_rate 5e-6

python -m trainers.vam.full_sft --config configs/vam/vam.yaml --epochs 5

# 指定 checkpoint 目录
python -m trainers.lm.full_sft --config configs/lm/lm_full_sft.yaml \
                               --save_dir checkpoint/my_exp

# 多卡 DDP 训练（torchrun）
torchrun --nproc_per_node=4 -m trainers.lm.pretrain --config configs/lm/lm_pretrain.yaml
torchrun --nproc_per_node=4 -m trainers.lm.full_sft --config configs/lm/lm_full_sft.yaml
```

### 推理 / 对话

```bash
# 原生 torch 格式（.pth）
python scripts/eval_llm.py --native --save_dir checkpoint/lm_full_sft_mini \
                           --weight full_sft --hidden_size 128

python scripts/eval_llm.py --native --save_dir checkpoint/lm_pretrain_mini \
                           --weight pretrain --hidden_size 128

# HuggingFace 格式（config.json + model.safetensors）
python scripts/eval_llm.py --load_from checkpoint/omni/native_hf \
                           --tokenizer_path checkpoint/omni/native_hf

python scripts/eval_llm.py --load_from checkpoint/lm_full_sft_mini/hf \
                           --tokenizer_path checkpoint/lm_full_sft_mini/hf

# 多模态（VLM / VAM）
# 原生 torch 格式（.pth）
python scripts/eval_vlm.py --native --save_dir checkpoint/vlm_sft_mini \
                           --weight sft_vlm --hidden_size 768 \
                           --image_dir dataset/eval_images

# VLM HF 格式（需先 convert；注：转换不含 vision encoder，为纯文本 LM）
python scripts/eval_vlm.py --load_from checkpoint/vlm_sft_mini/hf \
                           --tokenizer_path checkpoint/omni/native_hf \
                           --image_dir dataset/eval_images

python scripts/eval_vam.py --save_dir checkpoint/vam --weight full_sft
```

### 格式转换

```bash
# 原生 torch → HuggingFace 格式（omni 原生）
python scripts/convert_model.py checkpoint/lm_full_sft_mini/full_sft_128.pth \
                               checkpoint/lm_full_sft_mini/hf \
                               --tokenizer_path checkpoint/tokenizer

# VLM SFT（clean checkpoint 不含 vision encoder，转为纯文本 LM HF 格式）
python scripts/convert_model.py checkpoint/vlm_sft_mini/sft_vlm_768.pth \
                               checkpoint/vlm_sft_mini/hf \
                               --tokenizer_path checkpoint/omni/native_hf

# 从 .pth 转换到指定目录
python scripts/convert_model.py checkpoint/omni/omni.pth checkpoint/omni/native_hf \
                               --tokenizer_path <训练所用的 tokenizer 目录>

# 也可输出 Qwen3 兼容格式（发布到 HF Hub）
python scripts/convert_model.py checkpoint/omni/omni.pth output_dir --mode qwen \
                               --tokenizer_path <tokenizer 目录>

# 自动推断 hidden_size / num_hidden_layers、自定义精度
python scripts/convert_model.py checkpoint/omni/omni.pth output_dir --dtype bfloat16 \
                               --hidden_size 768 --num_hidden_layers 8
```

> **注意**：`--tokenizer_path` 必须传入**训练时使用的同一个 tokenizer**，否则模型加载后输出乱码（vocab 映射错位）。

## 配置说明

`configs/*.yaml` 分为 `model` / `train` / `paths` 三段，由 `utils.training.apply_config`
注入为 argparse 默认值；CLI 显式传参优先级更高。训练产出保存在 `checkpoint/` 目录。

## 与原 MiniMind 的差异

- 去除 `omni` 中间包，模块直接置于 `src/` 下（`core` / `models` / `trainers` / `dataset` / `utils` …），统一绝对导入；
- 类名统一：`MiniMindModel→LM`、`MiniMindVLM→VLM`、`MiniMindOmni→VAM`、`MiniMindConfig→LMConfig`、
  `MiniMindForCausalLM→LMForCausalLM`、`OmniConfig→VAMConfig`；
- `model_type` 重命名：`"minimind"`→`"omni"`、`"minimind-v"`→`"omni-v"`、`"minimind-o"`→`"omni-o"`；
- `trainer_utils.py` 拆分为 `utils/training.py`、`utils/distributed.py`、`utils/checkpoint.py`；
- 训练脚本暴露 `main(default_config=None)`，既可由 `python -m trainers.<mod>` 调用，也可由根 `trainer/*.py` 调用；
- `dataset/` 按数据集类型拆分为独立文件，供多模态扩展；
- 配置按模态分目录：`configs/lm/`、`configs/vlm/`、`configs/vam/`。
