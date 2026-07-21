# Omni 开发文档

> Omni — 把 MiniMind / MiniMind-V / MiniMind-O 三套代码融合成一个按模态分层（文本 / 视觉 / 全模态）的训练 + 推理框架。面向**学习与面试**的模块化开发文档。

## 仓库结构

```
docs/                 ↔  src/
├── architecture/     #   概念总览（入口，非代码模块）
├── core/             ↔  src/core/        纯 Transformer 组件
├── models/           ↔  src/models/      lm / vlm / vam 模型拼装
├── multimodal/       ↔  src/encoders/ + src/projectors/  多模态编码与投影
├── training/         ↔  src/dataset/ + src/trainers/ + src/utils/  训练相关
├── serve/            ↔  src/serve/       实时语音会话
└── interview/        #   面试速查（学习辅助）
```

## 文档导航

### 架构（概念入口）
| 文档 | 说明 |
| --- | --- |
| [架构总览](architecture/overview.md) | 能力分层设计、模型能力矩阵、前向数据流、训练入口、关键设计取舍 |

### core/（基础组件）
| 文档 | 说明 |
| --- | --- |
| [组件索引](core/index.md) | 与模态无关的纯 Transformer 组件总览 |
| [RMSNorm](core/norm.md) | RMSNorm 实现与数值技巧 |
| [RoPE](core/rope.md) | 旋转位置编码（含 YaRN）、`repeat_kv` |
| [Attention](core/attention.md) | 带 QK-Norm + GQA 的注意力 |
| [MLP / MoE](core/mlp.md) | SwiGLU FFN 与 MoE 前馈 |
| [Block](core/block.md) | Pre-Norm Transformer 块、MoE 可插拔 |

### models/（模型拼装）
| 文档 | 说明 |
| --- | --- |
| [模型索引](models/index.md) | `lm` / `vlm` / `vam` 三子包拼装、继承链、共享主干 |
| [LM](models/lm.md) | 纯文本主干 `LM` + `LMForCausalLM` |
| [VLM](models/vlm.md) | 文本 + 图像（SigLIP 编码器 + 视觉投影） |
| [VAM](models/vam.md) | 文本 + 图像 + 语音，`TalkerModule` 双 head |

### multimodal/（多模态）
| 文档 | 说明 |
| --- | --- |
| [多模态索引](multimodal/index.md) | 编码器 + 投影器、模态对齐范式 |

### training/（训练）
| 文档 | 说明 |
| --- | --- |
| [训练索引](training/index.md) | 数据集 + 训练脚本 + 工具 |
| [Trainers](training/trainers.md) | 各 trainer 模块概览与通用训练循环 |
| [配置与命令行](training/config-and-cli.md) | YAML 配置驱动训练、`apply_config` 机制 |

### serve/（实时语音会话）
| 文档 | 说明 |
| --- | --- |
| [服务索引](serve/index.md) | `SileroVAD` / `RealtimeSession` 端到端语音链路 |

### 面试速查
| 文档 | 说明 |
| --- | --- |
| [面试准备](interview/面试速查.md) | 高频问题 + 一句话答法，按主题组织 |

## 类名

| 类 | 说明 |
| --- | --- |
| `LM` | 纯文本主干 |
| `LMForCausalLM` | 文本模型（含 lm_head + 生成） |
| `LMConfig` | 文本模型配置 |
| `VLM` | 文本 + 视觉多模态 |
| `VAM` | 全模态（文本 + 视觉 + 语音） |
| `VAMConfig` | 全模态配置 |
| `model_type = "omni"` | LM 架构 ID |
| `model_type = "omni-v"` | VLM 架构 ID |
| `model_type = "omni-o"` | VAM 架构 ID |

包结构（已扁平化，无 `omni` 中间层）：

```
src/
├── core/        # 纯组件：norm / rope / attention / mlp / block
├── models/      # lm / vlm / vam 三个子包（config + model）
├── dataset/     # 每类一个文件的数据集
├── encoders/    # vision / audio 编码器
├── projectors/  # 视觉/音频桥接层
├── serve/       # 实时语音会话
├── trainers/    # lm / vlm / vam 训练脚本
└── utils/       # training / distributed / checkpoint / multimodal
```
