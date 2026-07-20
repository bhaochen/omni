# Omni 开发文档

> Omni — 把 MiniMind / MiniMind-V / MiniMind-O 三套代码融合成一个按模态分层（文本 / 视觉 / 全模态）的训练 + 推理框架。面向**学习与面试**的模块化开发文档。

文档目录与 `src/` 代码模块一一对应：

```
docs/                 ↔  src/
├── architecture/     #   概念总览（入口，非代码模块）
├── core/             ↔  src/core/        纯 Transformer 组件
├── models/           ↔  src/models/      lm / vlm / vam 模型拼装
├── dataset/          ↔  src/dataset/     数据集
├── encoders/         ↔  src/encoders/    视觉/音频编码器
├── projectors/       ↔  src/projectors/  模态桥接层
├── serve/            ↔  src/serve/       实时语音会话
├── trainers/         ↔  src/trainers/    训练脚本
├── utils/            ↔  src/utils/       训练/分布式/checkpoint 工具
└── interview/        #   面试速查（学习辅助）
```

## 目录

### 架构（概念入口）
| 文档 | 说明 |
| --- | --- |
| [架构总览](architecture/overview.md) | 能力分层设计、模型能力矩阵、前向数据流、训练入口、关键设计取舍 |

### core/（基础组件）
| 文档 | 说明 |
| --- | --- |
| [组件索引](core/README.md) | 与模态无关的纯 Transformer 组件总览 |
| [RMSNorm](core/norm.md) | RMSNorm 实现与数值技巧 |
| [RoPE](core/rope.md) | 旋转位置编码（含 YaRN）、`repeat_kv` |
| [Attention](core/attention.md) | 带 QK-Norm + GQA 的注意力 |
| [MLP / MoE](core/mlp.md) | SwiGLU FFN 与 MoE 前馈 |
| [Block](core/block.md) | Pre-Norm Transformer 块、MoE 可插拔 |

### models/（模型拼装）
| 文档 | 说明 |
| --- | --- |
| [模型索引](models/README.md) | `lm` / `vlm` / `vam` 三子包拼装、继承链、共享主干 |
| [LM](models/lm.md) | 纯文本主干 `LM` + `LMForCausalLM` |
| [VLM](models/vlm.md) | 文本 + 图像（SigLIP 编码器 + 视觉投影） |
| [VAM](models/vam.md) | 文本 + 图像 + 语音，`TalkerModule` 双 head |

### dataset/（数据集）
| 文档 | 说明 |
| --- | --- |
| [数据集索引](dataset/README.md) | `dataset/` 每类一个文件的数据集、loss mask、批次拼接 |

### encoders/（多模态编码器）
| 文档 | 说明 |
| --- | --- |
| [编码器索引](encoders/README.md) | `SiglipVisionEncoder` / `SenseVoiceAudioEncoder` |

### projectors/（模态桥接层）
| 文档 | 说明 |
| --- | --- |
| [桥接层索引](projectors/README.md) | `MMVisionProjector` / `MMAudioProjector`、注入范式 |

### serve/（实时语音会话）
| 文档 | 说明 |
| --- | --- |
| [服务索引](serve/README.md) | `SileroVAD` / `RealtimeSession` 端到端语音链路 |

### trainers/（训练脚本）
| 文档 | 说明 |
| --- | --- |
| [训练索引](trainers/README.md) | 按模态组织的 trainer 模块分布 |
| [Trainers](trainers/trainers.md) | 各 trainer 模块概览与通用训练循环 |

### utils/（训练工具）
| 文档 | 说明 |
| --- | --- |
| [工具索引](utils/README.md) | `training` / `checkpoint` / `distributed` / `multimodal` 工具分布 |
| [配置与命令行](utils/config-and-cli.md) | YAML 配置驱动训练、`apply_config` 机制、tokenizer 训练、启动示例 |

### 面试速查
| 文档 | 说明 |
| --- | --- |
| [面试准备](interview/README.md) | 高频问题 + 一句话答法，按主题组织 |

## 命名约定（本仓库）

类名已在融合过程中统一重命名，阅读代码/面试时对照：

| 旧名（上游） | 本仓库 |
| --- | --- |
| `MiniMindModel` | `LM` |
| `MiniMindForCausalLM` | `LMForCausalLM` |
| `MiniMindConfig` | `LMConfig` |
| `MiniMindVLM` | `VLM` |
| `MiniMindOmni` | `VAM` |
| `OmniConfig` | `VAMConfig` |

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
