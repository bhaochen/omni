# Omni

Omni 是一个以 **多模态 (omni)** 为目标的 LLM 训练 / 推理框架，已完整集成
[MiniMind](https://github.com/jingyaogong/minimind)（纯文本）、
[miniMind-V](https://github.com/jingyaogong/minimind-v)（视觉多模态）与
[miniMind-O](https://github.com/jingyaogong/minimind-o)（语音 / 全模态）三套代码。

项目采用标准 `src/` 布局（`pip install -e .` 即可安装为 `omni` 包），
按 **core（组件）/ models（拼装）/ encoders（模态编码器）/ projectors（桥接层）** 分层。

## 设计分层

- `core/`：可复用模型**纯底层组件**，按层级细分为独立模块——
  `norm.py`（`RMSNorm`）、`rope.py`（`precompute_freqs_cis` / `apply_rotary_pos_emb` / `repeat_kv`）、
  `attention.py`（`Attention`）、`mlp.py`（`FeedForward` / `MOEFeedForward`）、
  `block.py`（`MiniMindBlock`）。（Transformer 主干 `MiniMindModel` 已归入 `models/lm/model.py`）
- `models/`：把 `core` 组件**拼装**成成品模型，按模态能力分为三个子包（每个含 `config.py` 配置 + `model.py` 建模）：
  - `models/lm/`：纯文本——`MiniMindConfig` + `MiniMindForCausalLM`
  - `models/vlm/`：文本 + 视觉——`VLMConfig` + `MiniMindVLM`
  - `models/vam/`：文本 + 语音/全模态——`OmniConfig` + `MiniMindOmni`（含 `TalkerModule`）
- `encoders/`：外部模态编码器，按模态分目录——`vision/`（SigLIP）、`audio/`（SenseVoice）。
- `projectors/`：把 encoder 输出**桥接**到 LLM 隐藏维度的拼接层（`MMVisionProjector`、`MMAudioProjector`）。
- `serve/`：实时语音会话工程层（`SileroVAD`、`RealtimeSession`）。

## 目录结构

```
src/omni/
├── core/               # 模型底层组件（按层级拆分）
│   ├── norm.py         #   RMSNorm
│   ├── rope.py         #   precompute_freqs_cis / apply_rotary_pos_emb / repeat_kv
│   ├── attention.py    #   Attention
│   ├── mlp.py          #   FeedForward / MOEFeedForward
│   └── block.py        #   MiniMindBlock
├── models/             # 模型拼装（按模态能力分子包）
│   ├── lm/             #   纯文本
│   │   ├── config.py   #     MiniMindConfig
│   │   ├── model.py    #     MiniMindForCausalLM + MiniMindModel 主干
│   │   └── lora.py     #     LoRA 注入 / 保存 / 合并（作用于 MiniMind 系主干）
│   ├── vlm/            #   文本 + 视觉
│   │   ├── config.py   #     VLMConfig
│   │   └── model.py    #     MiniMindVLM
│   └── vam/            #   文本 + 语音/全模态
│       ├── config.py   #     OmniConfig
│       └── model.py    #     MiniMindOmni + TalkerModule
├── encoders/           # 多模态编码器（按模态分目录）
│   ├── vision/        #   SiglipVisionEncoder
│   └── audio/         #   SenseVoiceAudioEncoder
├── projectors/         # 多模态桥接层
│   ├── vision.py      #   MMVisionProjector
│   └── audio.py       #   MMAudioProjector
├── trainers/           # 训练脚本（可直接 python -m 运行）
│   ├── pretrain.py     # 文本预训练
│   ├── full_sft.py     # 文本全量 SFT
│   ├── lora.py / dpo.py / distillation.py / ppo.py / grpo.py / agent.py
│   ├── rollout_engine.py  # torch / sglang 推理引擎
│   ├── train_tokenizer.py # tokenizer 训练（学习用）
│   ├── pretrain_vlm.py # 视觉预训练
│   ├── full_sft_vlm.py # 视觉 SFT
│   └── full_sft_omni.py  # 全模态 SFT
├── datasets/           # 数据集（Pretrain/SFT/DPO/RLAIF/Agent/VLM/Omni）
│   └── lm_dataset.py
├── utils/             # 工具
│   ├── training.py     # get_lr / init_model / lm_checkpoint / SkipBatchSampler / LMForRewardModel
│   ├── multimodal.py   # init_vlm_model / vlm_checkpoint / init_omni_model / omni_checkpoint
│   ├── distributed.py  # 分布式初始化
│   └── checkpoint.py   # checkpoint 读写辅助
├── serve/             # 实时语音会话（SileroVAD / RealtimeSession）
└── __init__.py
examples/               # 推理 / 服务 / 转换脚本
├── eval_llm.py        # 命令行推理与对话
├── eval_vlm.py        # 视觉多模态推理
├── eval_omni.py       # 全模态推理
├── serve_openai_api.py # OpenAI 兼容 API 服务
├── omni_web_demo.py   # 网页演示（含实时语音）
├── eval_toolcall.py   # 工具调用评测
└── convert_model.py   # torch <-> transformers 权重互转
weights/MiniMind2/      # tokenizer 与模型配置（从 MiniMind 迁移）
configs/                # 训练配置（按需补充）
```

## 安装

```bash
pip install -e .
# 可选依赖：RL 训练 / API 服务 / 演示
pip install -e ".[rl,serve,demo]"
```

## 快速开始

### 推理 / 对话

```bash
python -m examples.eval_llm --load_from weights/MiniMind2 --weight full_sft
```

### 训练

每个训练脚本都是 `omni.trainers` 下的一个模块，直接运行即可：

```bash
# 预训练
python -m omni.trainers.pretrain --data_path dataset/pretrain.jsonl
# 全量 SFT
python -m omni.trainers.full_sft --data_path dataset/sft.jsonl
# LoRA 微调
python -m omni.trainers.lora --data_path dataset/lora.jsonl
# DPO / 蒸馏 / PPO / GRPO / Agent RL
python -m omni.trainers.dpo    --data_path dataset/dpo.jsonl
python -m omni.trainers.distillation --data_path dataset/sft.jsonl
python -m omni.trainers.grpo    --data_path dataset/rlaif.jsonl
python -m omni.trainers.agent  --data_path dataset/agent_rl.jsonl
```

所有脚本参数与原 MiniMind 保持一致（hidden_size / num_hidden_layers / use_moe / data_path 等）。

## 与原 MiniMind 的差异

- 去除所有 `sys.path` 注入 hack，统一使用 `omni.*` 包导入；
- `trainer_utils.py` 拆分为 `utils/training.py`（训练工具）、`utils/distributed.py`、`utils/checkpoint.py`；
- 训练脚本从「`if __name__ == '__main__'` 内联」改为可被 `python -m omni.trainers.<name>` 调用的模块；
- 预留 `encoders/`（vision/audio）、`projectors/`、`core/` 供多模态扩展。

## 多模态扩展方向

在 `models/minimind.py` 的 `MiniMindModel` 之上接入：

1. `encoders/vision` / `encoders/audio` —— 各自的模态编码器；
2. `projectors` —— 将编码器输出投影到 LLM 隐藏维度；
3. 在 `MiniMindModel.forward` 中把投影特征拼接到 `embed_tokens` 之后。
