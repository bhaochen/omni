# 面试：多模态深度

> 本仓库 `src/models/vlm/` 和 `src/models/vam/` 的多模态实现，覆盖视觉语言模型和全模态模型

## 0. 多模态数据流

```
原始输入(图/音)
   └─ encoder      → 模态特征
        └─ projector → 投影到 hidden_size
             └─ 替换 input_ids 中的占位 token 位置的 embedding
                  └─ 进入共享 LM 主干
```

---

## Q1. 图像/音频怎么进 LLM？

### VLM 注入流程（`src/models/vlm/model.py:56-77`）

```python
def count_vision_proj(self, input_ids, vision_proj_output):
    # 1. 找到图像占位 token 位置
    image_ids = self.config.image_ids  # [12]
    image_positions = (input_ids == image_ids[0]).nonzero()
    
    # 2. 将视觉特征注入 hidden_states 对应位置
    hidden_states = self.model.embed_tokens(input_ids)
    for i, pos in enumerate(image_positions):
        hidden_states[pos] = vision_proj_output[i]
    
    return hidden_states
```

### 注入范式

```
文本侧：特殊占位 token（<|vision_start|>` / ` Müslü`）预留位置
视觉侧：encoder → projector → 替换占位处 embedding
音频侧：encoder → projector → 替换占位处 embedding
```

> 面试点：为什么用占位 token 而不是拼接？→ 占位 token 可以精确定位注入位置，拼接会打乱文本顺序

---

## Q2. 为什么常冻结 encoder 只训 projector？

### 原因

1. **避免灾难性遗忘**：预训练主干参数量大，微调容易遗忘
2. **轻量对齐**：投影层参数少，足以完成模态对齐
3. **计算效率**：冻结 encoder 可以省显存和计算

### 本仓库实现（`src/utils/multimodal.py:29-53`）

```python
def init_vlm_model(model, config):
    # 1. 先冻结全部参数
    for param in model.parameters():
        param.requires_grad = False
    
    # 2. 只开 vision_proj
    for param in model.vision_proj.parameters():
        param.requires_grad = True
    
    # 3. 按需解冻 LLM
    if config.unfreeze_llm:
        for param in model.model.parameters():
            param.requires_grad = True
```

> 面试点：如果全量微调会怎样？→ 显存不足，且容易过拟合，投影层已经足够完成模态对齐

---

## Q3. VAM 的语音生成怎么做？

### Thinker + Talker 双塔结构

```
输入文本
    │
    ▼
┌─────────────┐
│   Thinker   │  (共享 LM 主干)
└──────┬──────┘
       │
       ├──────────────────────┐
       │                      │
       ▼                      ▼
┌─────────────┐        ┌─────────────┐
│  Text Head  │        │ TalkerModule │
└──────┬──────┘        └──────┬──────┘
       │                      │
       ▼                      ▼
    文本输出              8 层音频 code
```

### TalkerModule（`src/models/vam/model.py:52-79`）

```python
class TalkerModule(nn.Module):
    def __init__(self, config):
        # 独立的 Transformer 层
        self.layers = nn.ModuleList([
            Block(config) for _ in range(config.num_talker_hidden_layers)
        ])
        
        # bridge 机制：将 thinker 的 hidden_states 投影到 talker 维度
        self.embed_proj = nn.Linear(config.hidden_size, config.talker_hidden_size)
        
        # 双尺度融合（可学习参数）
        self.text_scale = nn.Parameter(torch.ones(1) * 3.0)
        self.audio_scale = nn.Parameter(torch.ones(1) * 1.0)
    
    def forward(self, thinker_hidden, audio_embeddings):
        # 1. 投影 thinker hidden
        text_features = self.embed_proj(thinker_hidden)
        
        # 2. 双尺度融合
        fused = self.text_scale * text_features + self.audio_scale * audio_embeddings
        
        # 3. 通过独立 Transformer 层
        for layer in self.layers:
            fused = layer(fused)
        
        return fused
```

### TalkerHead（`src/models/vam/model.py:22-34`）

```python
class TalkerHead(nn.Module):
    def __init__(self, config):
        self.base = nn.Linear(config.talker_hidden_size, config.audio_vocab_size)
        # 8 个 adapter，每个 adapter: Linear -> GELU -> Linear
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.talker_hidden_size, config.talker_hidden_size),
                nn.GELU(),
                nn.Linear(config.talker_hidden_size, config.audio_vocab_size)
            ) for _ in range(8)
        ])
    
    def forward(self, x):
        # 8 个并行输出头
        return [self.base(x) + adapter(x) for adapter in self.adapters]
```

> 面试点：为什么用 8 个 adapter？→ 音频 code 有 8 层，每层独立预测，8 个 adapter 并行处理

---

## Q4. VAM 的流式生成（`src/models/vam/model.py:309-388`）

### 核心思想

文本先完成，然后 8 个音频 code 交错流式输出。

### 实现细节

```python
def stream_generate(self, input_ids, ...):
    # 1. 先生成文本
    text_tokens = self.thinker.generate(input_ids, ...)
    
    # 2. 找到 think 结束标记
    think_end_ids = [26, 234, 234]  # 思考结束标记
    
    # 3. 8 个音频 code 交错生成
    audio_codes = [[] for _ in range(8)]
    for step in range(max_audio_length):
        # 每个 step 生成 8 个 code
        for i in range(8):
            logits = self.talker_head(talker_hidden)
            code = sample(logits[i])
            
            # 音频 code 范围: 0-2047 为正常 code，>=2048 为 stop token
            if code >= 2048:
                break
            
            audio_codes[i].append(code)
        
        # 更新 talker hidden
        audio_embeddings = self.audio_embedding(torch.tensor(audio_codes))
        talker_hidden = self.talker(text_hidden, audio_embeddings)
    
    return text_tokens, audio_codes
```

> 面试点：为什么文本和音频分开生成？→ 文本需要自回归生成，音频 code 可以并行预测，分开生成更高效

---

## Q5. SigLIP 视觉编码器

### 本仓库使用（`src/encoders/vision/siglip.py`）

- **输入**：图像 (batch, 3, 224, 224)
- **输出**：patch 特征 (batch, num_patches, hidden_size)

### 特点

1. **冻结参数**：预训练权重不参与训练
2. **Patch 特征**：将图像分割为 patch，每个 patch 作为一个 token
3. **输出维度**：通过 projector 投影到 LLM 的 hidden_size

> 面试点：SigLIP 和 CLIP 的区别？→ SigLIP 使用 sigmoid loss 替代 softmax loss，训练更稳定

---

## Q6. SenseVoice 音频编码器

### 本仓库使用（`src/encoders/audio/sensevoice.py`）

- **输入**：音频 (batch, audio_length)
- **输出**：语义/声学特征 (batch, seq_len, hidden_size)

### 特点

1. **双特征输出**：语义特征（理解内容）+ 声学特征（理解语气）
2. **冻结参数**：预训练权重不参与训练
3. **输出维度**：通过 projector 投影到 LLM 的 hidden_size

> 面试点：为什么需要双特征？→ 语义特征用于理解内容，声学特征用于理解语气和情感，两者结合才能完整理解语音

---

## Q7. VLM 的多图处理（`src/models/vlm/model.py:79-157`）

### 支持的输入格式

```python
# pixel_values 可以是：
1. tensor (batch, channels, height, width)  # 单图
2. dict {'pixel_values': tensor, 'image_ids': list}  # 多图
3. 4D/5D/6D 形状  # 不同分辨率
```

### 实现细节

```python
def forward(self, input_ids, pixel_values=None, ...):
    # 1. 文本 embedding
    hidden_states = self.model.embed_tokens(input_ids)
    
    # 2. 处理视觉输入
    if pixel_values is not None:
        if isinstance(pixel_values, dict):
            # 多图处理
            vision_features = self.vision_encoder(pixel_values['pixel_values'])
            vision_features = self.vision_proj(vision_features)
            
            # 注入到对应位置
            for i, pos in enumerate(pixel_values['image_ids']):
                hidden_states[:, pos:pos+self.config.image_token_len] = vision_features[i]
        else:
            # 单图处理
            vision_features = self.vision_encoder(pixel_values)
            vision_features = self.vision_proj(vision_features)
            hidden_states = self.count_vision_proj(input_ids, vision_features)
    
    # 3. 通过 LM 主干
    outputs = self.model(hidden_states, ...)
    return outputs
```

> 面试点：多图处理时如何区分不同图像？→ 通过 image_ids 标记每张图像的位置，分别注入对应的视觉特征

---

## Q8. aux_loss 中的技巧（`src/models/vlm/model.py:145`）

### 问题

在分布式训练中，如果 vision_proj 的参数没有出现在 loss 计算图中，DDP 梯度同步会出错。

### 解决方案

```python
def forward(self, input_ids, pixel_values=None, ...):
    # ... 前向传播 ...
    
    # 确保 vision_proj 参数出现在计算图中
    aux_loss = aux_loss + sum(p.sum() for p in self.vision_proj.parameters()) * 0
    
    return logits, aux_loss
```

### 为什么用 `* 0`？

- 保持计算图连通，让 DDP 可以同步梯度
- 实际值为 0，不影响 loss

> 面试点：为什么不用 `.requires_grad = True`？→ 那只是让参数可训练，但不会出现在计算图中，`* 0` 才能保证计算图连通

---

## Q9. object.__setattr__ 技巧（`src/models/vam/model.py:88-99`）

### 问题

VAM 模型需要将 thinker 的某些属性直接赋值给 VAM，但 `nn.Module.__setattr__` 会自动注册为子模块。

### 解决方案

```python
class VAM(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        # 绕过 nn.Module.__setattr__ 直接设置属性
        object.__setattr__(self, 'thinker', self.model)
        object.__setattr__(self, 'model', None)  # 避免重复注册
```

### 为什么需要这个技巧？

- `nn.Module.__setattr__` 会自动将 `nn.Module` 注册为子模块
- 但 thinker 已经是子模块了，重复注册会导致 DDP 问题
- `object.__setattr__` 绕过这个机制

> 面试点：如果不绕过会怎样？→ thinker 会被重复注册为子模块，导致 DDP 梯度同步出错

---

## Q10. Talker 层复制初始化（`src/utils/multimodal.py:156-164`）

### 问题

Talker 模块需要从头训练，初始化不当会导致训练不稳定。

### 解决方案

```python
def init_talker_from_thinker(talker, thinker, num_layers):
    # 用 thinker 的后 N 层初始化 talker
    for i in range(num_layers):
        talker.layers[i].load_state_dict(thinker.layers[-(num_layers - i)].state_dict())
```

### 为什么用后 N 层？

- 后 N 层更接近输出，语义更丰富
- 前 N 层更关注低级特征，对语音生成帮助不大

> 面试点：为什么不随机初始化？→ 随机初始化会导致训练初期不稳定，用 thinker 的权重可以加速收敛

---

## Q11. 音频 code 范围设计（`src/models/vam/config.py`）

### 关键常量

```python
audio_vocab_size = 2112      # 音频词汇表大小
audio_pad_token = 2049       # 填充 token
audio_stop_token = 2050      # 停止 token
audio_spk_token = 2051       # 说话人 token
```

### 设计思路

- **0-2047**：正常音频 code（2048 个）
- **2048-2051**：特殊 token（pad, stop, spk）

### 为什么这样设计？

1. **正常 code 足够**：2048 个 code 足以表示音频特征
2. **特殊 token 分离**：避免与正常 code 混淆
3. **stop token 重要**：用于判断音频生成何时结束

---

## Q12. bridge_layer 选择（`src/models/vam/config.py:24`）

### 实现

```python
bridge_layer = num_hidden_layers // 2 - 1
```

### 为什么选中间层？

- **前层**：关注低级特征，语义不够丰富
- **后层**：关注高级特征，但可能过拟合
- **中间层**：语义丰富度适中，泛化能力最好

> 面试点：如果选最后一层会怎样？→ 可能过拟合，且梯度消失风险更高

---

## Q13. VLM 的 aux_loss 设计（`src/models/vlm/model.py:145`）

### 问题

在分布式训练中，如果 vision_proj 的参数没有出现在 loss 计算图中，DDP 梯度同步会出错。

### 解决方案

```python
def forward(self, input_ids, pixel_values=None, ...):
    # ... 前向传播 ...
    
    # 确保 vision_proj 参数出现在计算图中
    aux_loss = aux_loss + sum(p.sum() for p in self.vision_proj.parameters()) * 0
    
    return logits, aux_loss
```

### 为什么用 `* 0`？

- 保持计算图连通，让 DDP 可以同步梯度
- 实际值为 0，不影响 loss

> 面试点：为什么不用 `.requires_grad = True`？→ 那只是让参数可训练，但不会出现在计算图中，`* 0` 才能保证计算图连通

---

## Q14. 多模态训练的冻结策略

### VLM 冻结策略（`src/utils/multimodal.py:29-53`）

```python
# 阶段 1：只训练 vision_proj
for param in model.parameters():
    param.requires_grad = False
for param in model.vision_proj.parameters():
    param.requires_grad = True

# 阶段 2：解冻 LLM
if config.unfreeze_llm:
    for param in model.model.parameters():
        param.requires_grad = True
```

### VAM 冻结策略（`src/utils/multimodal.py:139-175`）

```python
# freeze_backbone 支持 'all' 和 'last1' 模式
if config.freeze_backbone == 'all':
    # 冻结所有 thinker 参数
    for param in model.thinker.parameters():
        param.requires_grad = False
elif config.freeze_backbone == 'last1':
    # 只解冻最后一层
    for param in model.thinker.layers[-1].parameters():
        param.requires_grad = True
```

> 面试点：为什么要分阶段冻结？→ 先对齐模态，再微调主干，避免灾难性遗忘

---

## Q15. 多模态数据处理

### VLM 数据集（`src/dataset/vlm.py`）

```python
class VLMDataset(Dataset):
    def __getitem__(self, idx):
        # 1. 加载图像
        image = Image.open(self.image_paths[idx])
        
        # 2. 加载文本
        text = self.texts[idx]
        
        # 3. Tokenize
        input_ids = self.tokenizer(text, return_tensors='pt')
        
        # 4. 图像预处理
        pixel_values = self.image_processor(image, return_tensors='pt')
        
        return input_ids, pixel_values
```

### VAM 数据集（`src/dataset/vam.py`）

```python
class VAMDataset(Dataset):
    def __getitem__(self, idx):
        # 1. 加载图像
        image = Image.open(self.image_paths[idx])
        
        # 2. 加载音频
        audio = torchaudio.load(self.audio_paths[idx])
        
        # 3. 加载文本
        text = self.texts[idx]
        
        # 4. Tokenize
        input_ids = self.tokenizer(text, return_tensors='pt')
        
        # 5. 音频特征提取
        audio_features = self.audio_encoder(audio)
        
        return input_ids, pixel_values, audio_features
```

> 面试点：多模态数据如何 batch？→ 使用 collate_fn 处理不同长度的序列，padding 到统一长度
