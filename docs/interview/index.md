# interview/ · 面试速查

按主题组织的「高频问题 + 一句话答法」，配合各模块详细文档食用。

## 深度文档导航

| 文档 | 内容 |
| --- | --- |
| [Transformer 架构](transformer-arch.md) | RMSNorm、RoPE、GQA、QK-Norm、SwiGLU、Pre-Norm、Flash-Attention、参数量估算、KV Cache |
| [训练系统](training-systems.md) | Loss 计算、DPO、GRPO、PPO、蒸馏、Rollout Engine、混合精度、梯度累积、Checkpoint |
| [多模态](multimodal.md) | VLM/VAM 注入范式、SigLIP/SenseVoice、TalkerModule、流式生成、冻结策略 |
| [MoE](moe.md) | 路由机制、负载均衡、死 Expert 梯度、Expert 初始化、推理优化 |
| [推理优化](inference.md) | KV Cache、Prefill/Decode、采样策略、Repetition Penalty、量化、显存估算 |

---

## 架构与设计

**Q: 这个框架怎么组织多模态模型？**
A: `core` 放模态无关纯组件；`models` 按 `lm/vlm/vam` 拼装，`VLM`/`VAM` 继承 `LMForCausalLM` 复用主干；外部模态经 `encoders`+`projectors` 投影到 LLM 隐藏维后替换占位 token。

**Q: 为什么 VLM/VAM 继承同一个 LMForCausalLM？**
A: 共享 `LM` 主干、`lm_head`、`generate()`；多模态只在 `forward` 注入视觉/音频特征，新增模态零改动主干。

**Q: 权重绑定（weight tying）是什么？**
A: `lm_head` 与 `embed_tokens` 共享权重，省一半词表参数；用 `_tied_weights_keys` 让 HF 只存一份。

## 位置编码 / 注意力

**Q: RoPE 相比绝对位置编码好在哪？**
A: 注意力分数只依赖相对距离（旋转角度差），天然外推友好，且不增参。

**Q: 什么是 YaRN？**
A: 长上下文 RoPE 扩展：对中频段频率做 `1/factor` 缩放 + 注意力因子修正（NTK-aware），在有限微调下把上下文从 2k 扩到 32k+。

**Q: GQA 解决了什么？**
A: 减少 KV 头数（k/v 经 `repeat_kv` 复制），降低推理显存与延迟，质量介于 MHA 与 MQA 之间。

**Q: QK-Norm 为什么有用？**
A: 在 RoPE 前对 q/k 每 head 做 RMSNorm，抑制注意力 logits 过大，提升训练稳定性。

**Q: Flash-Attention 为什么快？**
A: 分块计算、不物化完整 N×N 注意力矩阵，减少 HBM 读写（IO 感知）。

## MoE

**Q: MoE 的稀疏指什么？**
A: 每 token 只过 top-k 个专家（默认 k=1），参数量大但单步计算量小。

**Q: 为什么需要 aux_loss？**
A: 防止 router 只选少数专家（坍缩），用 `Σ load_e·mean(scores)_e` 鼓励专家负载均衡。

**Q: 本仓库 MoE 怎么切换？**
A: `Block` 根据 `config.use_moe` 在 `FeedForward` / `MOEFeedForward` 间切换；`aux_loss` 在 `MLP.forward` 内累加，主模型汇总到总损失。

## 训练

**Q: 损失里 label 为什么要平移？**
A: `logits[:, :-1]` 对齐 `labels[:, 1:]`，标准 next-token 预测。

**Q: SFT 为什么只标 assistant 段？**
A: loss mask 把 prompt/system 段 label 置 -100，模型只学生成回复，不拟合用户输入。

**Q: 配置系统优先级？**
A: `CLI 参数 > YAML 默认值 > 代码默认`；`apply_config` 把 YAML 扁平化注入 argparse 默认值。

**Q: 续训怎么保证精确？**
A: `SkipBatchSampler` 在分布式下跳过已训 step；`from_resume` 从 `checkpoint/` 恢复 optimizer/scheduler 状态。

## 多模态

**Q: 图像/音频怎么进 LLM？**
A: 占位特殊 token 预留位置 → encoder 提特征 → projector 投影到 hidden → 替换占位处 embedding。

**Q: 为什么常冻结 encoder 只训 projector？**
A: 避免预训练主干灾难性遗忘，轻量投影层即可完成模态对齐。

**Q: VAM 的语音生成怎么做？**
A: `TalkerModule` 在主干某层后把文本隐状态解码为 8 层音频 code（Mimi 风格），与文本 head 并行训练。

## 推理

**Q: KV Cache 为什么快？**
A: 缓存已算的 K/V，每步只算新 token 的 Q 与已有 K/V 做注意力，避免重算。

**Q: Logits to Keep 优化是什么？**
A: 仅计算最后 N 个 token 的 logits，显存从 `seq_len × vocab_size` 降到 `1 × vocab_size`。

**Q: Top-k 和 Top-p 采样的区别？**
A: Top-k 保留固定数量的 token，Top-p 保留累积概率达到 p 的 token（动态数量）。
