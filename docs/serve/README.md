# serve/ · 实时语音会话

端到端语音对话链路（VAD 检测 → ASR → LLM → TTS/Talker）。

| 文件 | 内容 | 作用 |
| --- | --- | --- |
| `serve/realtime.py` | `SileroVAD` / `RealtimeSession` | 实时语音会话（VAD + 全双工） |

## 要点（面试）

- `serve/realtime.py` 提供端到端语音对话链路：VAD 检测语音段 → ASR 转文本 → LLM 生成 → TTS / `VAM.TalkerModule` 合成语音。
- 与理解侧（`encoders` + `projectors`）共用 LLM 主干，是「全模态」能力的推理入口。
