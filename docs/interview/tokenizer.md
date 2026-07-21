# Tokenizer 深度讲解

> 本仓库 `src/trainers/lm/train_tokenizer.py` 的 tokenizer 训练实现，覆盖 BPE 原理、特殊 token 体系、配置文件结构

## 0. Tokenizer 在 LLM 中的位置

```
原始文本
    |
    v
+-------------+
|  Tokenizer  |  文本 -> token ids
+------+------+
       |
       v
+-------------+
| Embedding   |  token ids -> 向量
+------+------+
       |
       v
    Transformer
```

Tokenizer 是 LLM 的"词典"，决定了模型如何理解文本。

---

## Q1. 为什么用 BPE？

### 分词方案对比

| 方案 | 优点 | 缺点 |
|------|------|------|
| 字符级 | 词表小，无 OOV | 序列长，语义弱 |
| 词级 | 语义强 | 词表大，有 OOV |
| BPE | 平衡词表大小和语义 | 需要训练 |

### BPE 核心思想

```

初始：所有字符独立
"low" -> ['l', 'o', 'w']

第1步：统计最频繁的相邻对
"lo" 出现 100 次，"ow" 出现 80 次

第2步：合并最频繁的对
"lo w" -> ['lo', 'w']

重复直到达到词表大小
```

### BPE 训练三步

训练 BPE tokenizer 只需要三步：

```python
# 1. 创建模型
tokenizer = Tokenizer(models.BPE())

# 2. 配置预分词 + 训练器
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
trainer = trainers.BpeTrainer(vocab_size=6400, special_tokens=[...])

# 3. 训练
tokenizer.train_from_iterator(texts, trainer=trainer)
```

> 面试点：BPE 和 WordPiece 的区别？BPE 基于频率合并，WordPiece 基于似然合并

---

## Q2. 字节级 BPE（ByteLevel）为什么好？

### 传统 BPE 的问题

- 基础字母表是 Unicode 字符
- 不同语言需要不同处理
- 可能出现 OOV

### 字节级 BPE 的优势

```python
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
trainer = trainers.BpeTrainer(
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # 256 个字节值
)
```

**核心改进**：

1. 基础字母表是 256 个字节值，覆盖所有可能的输入
2. 任何文本都能无损表示，永远不会 OOV
3. 中文/英文/emoji/代码都能统一处理
4. 加前缀空格是 GPT-2 风格，本仓库设为 False

> 面试点：为什么字节级 BPE 比字符级好？封闭 256 词表、无 OOV、多语言无损

---

## Q3. 本仓库的特殊 Token 体系

### 三层结构（共 36 个）

#### 第一层：核心特殊 token（21 个，special=True）

| 类别 | Token | 用途 |
|------|-------|------|
| 对话结构 | `<|im_start|>` / `<|im_end|>` | ChatML 角色标记边界 |
| 视觉 | `<|vision_start|>` / `<|vision_end|>` | 视觉内容边界 |
| 视觉 | `<|image_pad|>` | 图像占位 token |
| 视觉 | `<|vision_pad|>` | 视觉填充 |
| 音频 | `<|audio_start|>` / `<|audio_end|>` | 音频内容边界 |
| 音频 | `<|audio_pad|>` | 音频占位 token |
| TTS | `<tts_pad>` | TTS 填充 |
| TTS | `<tts_text_bos>` / `<tts_text_eod>` | TTS 文本开始/结束 |
| TTS | `<tts_text_bos_single>` | TTS 单句开始 |
| 对象引用 | `<|object_ref_start|>` / `<|object_ref_end|>` | 对象引用边界 |
| 坐标 | `<|box_start|>` / `<|box_end|>` | 边界框坐标 |
| 坐标 | `<|quad_start|>` / `<|quad_end|>` | 四边形坐标 |
| 视频 | `<|video_pad|>` | 视频占位 token |
| 结束 | `<|endoftext|>` | 文本结束 / pad / unk |

#### 第二层：额外特殊 token（6 个，special=True）

| Token | 用途 |
|-------|------|
| `<tool_call>` / `</tool_call>` | 工具调用边界 |
| `<tool_response>` / `</tool_response>` | 工具响应边界 |
| `<think>` / `</think>` | 思考链边界 |

#### 第三层：预留 buffer token（9 个，special=False）

```
<|buffer1|> ~ <|buffer9|>
用途：为未来扩展预留位置
```

### 为什么这样设计？

1. **ChatML 格式**：业界标准，兼容性好
2. **多模态支持**：视觉/音频 token 让模型区分不同模态
3. **工具调用**：支持 function calling
4. **思考能力**：`<think>` / `</think>` 支持 chain-of-thought
5. **预留扩展**：buffer token 为未来功能留空间

---

## Q4. 训练数据加载

```python
def get_texts(data_path):
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 10000:
                break  # 只取 10000 行
            data = json.loads(line)
            # 提取所有对话中的 content 字段
            contents = [
                item['content']
                for item in data['conversations']
                if item.get('content')
            ]
            yield '\n'.join(contents)
```

数据来源：sft_t2t_mini.jsonl（SFT 对话数据）
格式：每行是 JSON，包含 conversations 数组
提取：将所有 content 字段拼成一段文本

---

## Q5. BPE 训练内部过程

### 步骤 1：初始化词表

```
基础词表 = 256 个字节值 + 36 个特殊 token = 292
目标词表 = 6400
还需合并次数 = 6400 - 292 = 6108
```

### 步骤 2：统计字节对频率

```
输入: "the cat sat on the mat"
统计相邻对频率:
  "t", "h" -> 3 次
  "h", "e" -> 2 次
  "a", "t" -> 3 次
  ...
```

### 步骤 3：合并最频繁对

```
合并 "th":
  "the" -> ["th", "e"]
继续合并 "he":
  "the" -> ["the"]
继续...直到词表达到 6400
```

### 最后一次合并

```
第 6108 次合并后，词表包含：
- 256 个字节 token
- 36 个特殊 token
- 6108 个合并后的子词 token
= 6400 个 token
```

---

## Q6. tokenizer.json 结构详解

### 整体结构

```json
{
  "version": "1.0",
  "model": {
    "type": "BPE",
    "vocab": {
      "!": 0,
      "\"": 1,
      ...
      "the": 256,
      "ing": 257,
      ...
    },
    "merges": [
      "t h",
      "h e",
      "t h e",
      ...
    ]
  },
  "added_tokens": [
    {
      "id": 6400,
      "content": "<|im_start|>",
      "single_word": false,
      "lstrip": false,
      "rstrip": false,
      "normalized": false,
      "special": true
    }
  ]
}
```

| 字段 | 说明 |
|------|------|
| vocab | token -> id 映射（6400 个条目） |
| merges | BPE 合并规则（按合并顺序排列） |
| added_tokens | 额外添加的 token（36 个特殊 token） |

---

## Q7. tokenizer_config.json 关键配置

```json
{
  "bos_token": "<|im_start|>",
  "eos_token": "<|im_end|>",
  "pad_token": "<|endoftext|>",
  "unk_token": "<|endoftext|>",
  "additional_special_tokens": [
    "<|object_ref_start|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|audio_start|>",
    "<|audio_end|>",
    "<|image_pad|>",
    "<|audio_pad|>",
    ...
  ],
  "chat_template": "...",
  "image_token": "<|image_pad|>",
  "audio_token": "<|audio_pad|>",
  "video_token": "<|video_pad|>",
  "vision_bos_token": "<|vision_start|>",
  "vision_eos_token": "<|vision_end|>",
  "audio_bos_token": "<|audio_start|>",
  "audio_eos_token": "<|audio_end|>",
  "tokenizer_class": "PreTrainedTokenizerFast"
}
```

### 关键字段说明

| 字段 | 值 | 说明 |
|------|-----|------|
| bos_token | `<|im_start|>` | 开始 token |
| eos_token | `<|im_end|>` | 结束 token |
| pad_token | `<|endoftext|>` | 填充 token |
| unk_token | `<|endoftext|>` | 未知 token |
| model_max_length | 131072 | 最大序列长度 |

---

## Q8. Chat Template 的工作原理

### 模板示例

```
<|im_start|>system
你是一个助手<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！有什么可以帮助你的？<|im_end|>
```

### 代码调用

```python
messages = [
    {"role": "system", "content": "你是一个助手"},
    {"role": "user", "content": "你好"},
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False)
# -> "<|im_start|>system\n你是一个助手<|im_end|>\n<|im_start|>user\n你好<|im_end|>\n<|im_start|>assistant\n"
```

### Chat Template 格式

```
{% for message in messages %}
  <|im_start|>{{ message.role }}
  {{ message.content }}<|im_end|>
{% endfor %}
<|im_start|>assistant
```

---

## Q9. 编码解码机制

### 编码过程（text -> token ids）

```
输入: "你好世界"

分词过程:
1. ByteLevel 预分词: 将文本转成字节序列
2. BPE 合并: 按 merges 规则合并字节对
3. 输出 token ids: [123, 456, 789]

底层原理:
"你好" 的 UTF-8 字节 -> [0xe4, 0xbd, 0xa0, 0xe5, 0xa5, 0xbd]
BPE 逐步合并字节对 -> 最终形成子词 token
```

### 解码过程（token ids -> text）

```
输入: [123, 456, 789]

解码过程:
1. 查 vocab 得到 token 字符串
2. ByteLevel 解码: 字节序列 -> UTF-8 文本
3. 输出: "你好世界"

注意: decode 时 skip_special_tokens 参数
- True: 跳过特殊 token（默认）
- False: 保留特殊 token
```

### 编解码一致性测试

```python
text = "你好世界"
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)
assert decoded == text  # 字节级 BPE 保证无损
```

---

## Q10. 压缩率评估

### 本仓库的评测方法

```python
def eval_compression(tokenizer, texts):
    for text in texts:
        encoded = tokenizer.encode(text)
        char_count = len(text)
        token_count = len(encoded)
        ratio = char_count / token_count
        print(f"Chars: {char_count}, Tokens: {token_count}, 压缩率: {ratio:.2f}")
```

### 典型结果

| 语言 | 样本 | 压缩率 |
|------|------|--------|
| 中文 | 200 字 | 3.5 chars/token |
| 英文 | 200 词 | 4.0 chars/token |
| 混合 | 中英混合 | 3.8 chars/token |

> 中文压缩率偏低，因为中文字符多，字节表示更密集

---

## Q11. 流式解码

### 问题

逐 token 解码时，字节可能跨 token 分布，导致出现 Unicode 替换字符（\ufffd）。

### 本仓库的流式解码

```python
token_cache = []
for tid in input_ids:
    token_cache.append(tid)
    current_decode = tokenizer.decode(token_cache)
    # 只有当没有替换字符时才输出
    if current_decode and '\ufffd' not in current_decode:
        print(current_decode, end='')
        token_cache = []
```

### 为什么用缓存？

- UTF-8 编码中，一个中文字符可能被拆成多个 token
- 需要等待完整字节序列再解码
- `\ufffd` 检测确保输出完整的字符

---

## Q12. 为什么不建议重复训练 tokenizer？

```python
# 脚本开头的注释：
# Note: It is not recommended to re-train the tokenizer.
# MiniMind already includes one.
```

### 原因

1. **社区兼容性**：不同 tokenizer 导致模型不兼容
2. **数据敏感**：训练数据影响分词质量
3. **词表依赖**：下游任务依赖词表一致性
4. **已足够好**：6400 词表对小模型足够

### 什么时候需要重训？

- 新领域：代码、法律、医疗等专业领域
- 新语言：当前词表不支持的语言
- 优化需求：压缩率不满足要求

---

## Q13. tokenizer 与模型的交互

### 在模型前向传播中的位置

```python
# src/models/lm/model.py
class LMForCausalLM(PreTrainedModel):
    def forward(self, input_ids, ...):
        # input_ids = tokenizer.encode(text)
        hidden_states = self.model.embed_tokens(input_ids)
        # ... transformer ...
        return logits

    def generate(self, input_ids, ...):
        # 1. 编码: tokenizer.encode(prompt)
        # 2. 逐 token 生成
        for _ in range(max_new_tokens):
            logits = self(input_ids)
            next_token = sample(logits)
            input_ids = torch.cat([input_ids, next_token])
        # 3. 解码: tokenizer.decode(output_ids)
```

### 交互流程

```
用户输入文本
    |
    v
tokenizer.encode(text) -> [token ids]
    |
    v
model.forward(input_ids) -> logits
    |
    v
sample(logits) -> next token id
    |
    v
tokenizer.decode([next token]) -> 文本
```

---

## Q14. Tokenizer 与多模态

### 占位 token 机制

```python
# 视觉占位 token
"<|image_pad|>" -> id 6412

# 在输入序列中的位置
input_ids: [bos, ..., image_pad, image_pad, ..., eos]
                  |          |
                  1 个 token 对应 64 个视觉特征
                  image_token_len = 64
```

### VLM 中的使用

```python
# src/models/vlm/model.py
def count_vision_proj(self, input_ids, vision_proj_output):
    # 找到视觉占位 token 位置
    image_positions = (input_ids == self.config.image_ids[0]).nonzero()
    # 替换为视觉特征
    for i, pos in enumerate(image_positions):
        hidden_states[pos] = vision_proj_output[i]
```

### 多模态特殊 token 的作用

| Token | id | 在多模态中的角色 |
|-------|-----|-----------------|
| `<|vision_start|>` | 6408 | 视觉输入开始标记 |
| `<|vision_end|>` | 6409 | 视觉输入结束标记 |
| `<|image_pad|>` | 6412 | 图像特征占位符 |
| `<|audio_start|>` | 6414 | 音频输入开始标记 |
| `<|audio_pad|>` | 6416 | 音频特征占位符 |

---

## Q15. 常见面试问题

### Q: 词表大小怎么选？

A: 本仓库用 6400。词表太小 -> 序列长；词表太大 -> Embedding 参数多。通常根据模型规模选：小模型（<1B）用 8k-16k，大模型用 32k-128k。

### Q: ByteLevel 和 WordLevel 的区别？

A: ByteLevel 以字节为基本单元，WordLevel 以单词为基本单元。ByteLevel 更通用、无 OOV。

### Q: data_path 中的 jsonl 文件是什么格式？

A: 每行一个 JSON 对象，包含 conversations 数组，每个 conversation 包含 role 和 content 字段。

### Q: 为什么需要 special_tokens_num 参数？

A: 预留位置给特殊 token。特殊 token 在 BPE 训练时直接加入词表，不会被进一步拆分。

### Q: 怎样评估 tokenizer 的质量？

A: 看压缩率（chars/token）、编解码一致性、流式解码是否出现乱码。
