# Omni 开发文档

面向**学习与面试**的模块化开发文档。本仓库把 MiniMind / MiniMind-V / MiniMind-O 三套代码融合成一个按模态分层的训练 / 推理框架。

> 阅读顺序建议：先看 `01-architecture-overview.md` 建立全局视图，再按子系统深入 `core/` `models/` `training/` 等。

## 文档地图

| 主题 | 入口 |
| --- | --- |
| 全局架构、目录、数据流 | [01-architecture-overview.md](01-architecture-overview.md) |
| 基础组件（norm / rope / attention / mlp / block） | [core/README.md](core/README.md) |
| 模型（LM / VLM / VAM） | [models/README.md](models/README.md) |
| 数据集 | [data/README.md](data/README.md) |
| 训练（配置、trainer） | [training/README.md](training/README.md) |
| 多模态（encoder / projector / serve） | [multimodal/README.md](multimodal/README.md) |
| 面试速查 | [interview/README.md](interview/README.md) |

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
├── encoders/    # vision / audio 编码器
├── projectors/  # 视觉/音频桥接层
├── serve/       # 实时语音会话
├── trainers/    # lm / vlm / vam 训练脚本
├── dataset/     # 每类一个文件的数据集
└── utils/       # training / distributed / checkpoint / multimodal
```
