# 系统设计（design.md）

## 1. 总体架构

```
┌─────────────────────────────── Browser (Chrome / Edge) ───────────────────────────────┐
│  static/index.html + style.css + app.js                                               │
│                                                                                       │
│  麦克风: getUserMedia → AudioContext(16kHz) → AudioWorklet(ScriptProcessor 降级)        │
│          → PCM 累计缓冲 → 前端编码 WAV(16k/mono/16bit)                                  │
│          → 每 3s 整段重识别(实时预览) → 停止后最终识别                                    │
│  上传:   文件预检(扩展名/空/10MB) → FormData                                            │
│        两条路径共用同一 fetch('/api/transcribe', multipart)                              │
└──────────────────────────────────────┬────────────────────────────────────────────────┘
                                       │ HTTP (multipart/form-data)
┌──────────────────────────────────────▼────────────────────────────────────────────────┐
│  FastAPI (app/main.py)                                                                │
│   GET /            → static/index.html                                                │
│   GET /health      → {"status":"ok","model":<状态>}                                    │
│   POST /api/transcribe:                                                               │
│     ① language 校验 (zh/en/auto, 大小写/空白容错)          → 400 非法                   │
│     ② 流式读取文件, ≤10MB                                  → 413 超限 / 400 空         │
│     ③ PyAV 解码 → 16kHz mono float32 (app/audio.py)       → 400 损坏/无音频流           │
│     ④ ASRService.transcribe(pcm, lang)  [线程池]                                       │
│     ⑤ JSON: text/language/duration_seconds/processing_seconds                          │
│   异常: AppError(状态码,中文信息) / 兜底 500                                             │
└──────────────┬────────────────────────────────────────────────────────────────────────┘
               │ app/asr/whisper.py (WhisperASRService, 单例, 后台线程懒加载)
┌──────────────▼────────────────────────────────────────────────────────────────────────┐
│  faster-whisper (CTranslate2, CPU int8, small) + Silero VAD(onnxruntime)               │
│  首次: HuggingFace 下载权重 → ~/.cache/huggingface  → 之后完全离线                       │
└───────────────────────────────────────────────────────────────────────────────────────┘
```

## 2. 核心模块

| 模块 | 职责 | 关键决策 |
|---|---|---|
| `app/main.py` | 路由、参数校验、错误映射 | `create_app(asr_service)` 工厂支持依赖注入，Mock 测试不碰真实模型；解码与推理经 `run_in_threadpool` 执行，避免阻塞事件循环（识别期间 `/health` 仍可响应） |
| `app/audio.py` | 任意音频 → 16kHz mono float32 | 用 PyAV（wheel 自带 FFmpeg 库）统一解码 WAV/WebM/MP3，无需系统 ffmpeg；时长按解码采样点计算（对无时长头的容器也准确） |
| `app/asr/base.py` | `ASRService` 抽象 + `Transcription` 数据类 | 接口隔离：Web 层不依赖具体模型；`inference_seconds` 单独返回，使 `processing_seconds` 可按题目口径排除模型加载时间 |
| `app/asr/whisper.py` | faster-whisper 封装 | 状态机 idle→loading→ready/failed；启动时后台预热（lifespan），请求到达时未就绪则等待（带超时）；加载失败返回 503 可读信息 |
| `static/app.js` | 录音/上传/状态机/实时预览 | 详见下文 |

## 3. 前端设计要点

1. **录音即 WAV**：`AudioContext({sampleRate:16000})` 让浏览器完成重采样，AudioWorklet `recorder-worklet.js` 逐帧回传 Float32 PCM，主线程累计后用 `encodeWav()` 拼装 16-bit WAV Blob。好处：时长精确（按采样点）、60 秒上限可在采样层面精确判定、实时预览可随时取「当前累计音频」整段识别、与上传共用同一接口。
2. **保活技巧**：Chrome 只「拉取」连接到 destination 的节点，因此 Worklet → 增益 0 节点 → destination，既保持数据流又不回放麦克风（避免啸叫）。
3. **状态机**：`idle / recording / recognizing / done / error`，徽章 + 按钮禁用 + 计时器三重反馈；识别失败保留「重试」按钮（重发上次音频，不要求用户重新录音/选择文件）。
4. **实时预览（加分）**：录音中每 3s 整段重识别一次（串行，busy 则跳过）；预览结果标注「实时预览」，停止后最终识别覆盖之。
5. **错误文案**：按 `err.name` 映射麦克风错误（NotAllowed/NotFound/NotReadable/Overconstrained），后端 `detail` 直接展示。

## 4. 音频处理流水线

```
浏览器:  mic(48k) ─AudioContext─▶ 16k mono Float32 ─累计─▶ WAV Blob(≤60s, ~1.9MB)
上传:    file(≤10MB) ──────────────────────────────────▶ multipart
后端:    bytes ─PyAV(重采样 s16/mono/16k)─▶ float32 PCM ─▶ VAD 过滤 ─▶ Whisper ─▶ 文本
         时长 = 采样点数 / 16000；processing = 解码耗时 + 推理耗时（不含模型加载）
```

## 5. ASR 模型选型及理由

**选定：faster-whisper（small / int8 / CPU）**

| 候选 | 结论 | 原因 |
|---|---|---|
| **faster-whisper** ✅ | 采用 | Whisper 的 CTranslate2 实现：中英文同模型、CPU int8 快（small 在普通笔记本上 5–10s 音频一般 1–5s 完成）、pip wheel 全平台（含 Win11 x64、无编译）、PyAV 解码多格式、缓存后离线 |
| openai-whisper | 弃 | 官方实现依赖 PyTorch（~2GB）、CPU 慢 2–4 倍 |
| Vosk | 弃 | 轻量流式，但中文准确率明显低于 Whisper，标点/大小写弱 |
| FunASR (Paraformer) | 弃 | 中文顶级但英文弱、依赖链复杂（modelscope 生态），对 Win11 免编译不够友好 |
| sherpa-onnx | 弃 | 流式生态好，但离线批量场景准确率/文档不及 Whisper 系，集成复杂度更高 |

- 规格默认 `small`（准确率/速度/内存的平衡点，~460MB）；低配机器可 `ASR_MODEL=base`（~145MB）。
- `vad_filter=True`（Silero VAD）：跳过静音段，既提速又抑制 Whisper 对静音的「幻觉文本」，满足「无语音返回空结果」的要求。
- `beam_size=5`：默认束宽，短音频质量更好；速度敏感时可降为 1。
- 自动检测：`language=None` 时由 Whisper 检测（返回 `info.language`，日志记录）。

## 6. 错误处理与 HTTP 映射

| 层 | 异常 | HTTP | 前端表现 |
|---|---|---|---|
| 参数 | language 非法 | 400 | 展示 detail |
| 参数 | 缺 file/language | 422 | 展示校验信息 |
| 文件 | 空、损坏、无音频流、解码失败 | 400 | 展示 detail + 重试 |
| 文件 | >10MB | 413 | 展示 detail |
| 模型 | 未就绪超时 / 加载失败 | 503 | 提示稍后重试/失败原因 |
| 模型 | 推理异常 | 500 | 「服务器内部错误」+ 重试 |
| 结果 | 无语音内容 | 200 空文本 | 「未识别到语音内容」 |

## 7. 测试架构（与 design 联动）

- **依赖注入**：`create_app(asr_service=FakeASRService())` → 常规套件零模型依赖（题目硬性要求）；
- **realmodel 标记**：`pytest.ini` 默认 `-m "not realmodel"`，真实模型套件显式 `-m realmodel` 运行；
- **音频 fixture**：测试内用 `wave` 生成正弦/静音 WAV，用 PyAV 编码 WebM/Opus（覆盖浏览器录音格式族）；
- **前端冒烟**：断言 `/`、静态资源可访问，且 app.js 含关键行为（60s、权限错误映射、目标接口）。

## 8. 已知取舍

1. 截断但可部分解码的音频：返回可识别部分而非报错（FFmpeg 容错语义），完全损坏才 400；
2. 自动检测对**段落级**中英切换不保证逐句切换语言（Whisper 整段检测），已在 UI 标注「实验性」；
3. 实时预览采用整段重识别（O(n) 请求体增长，60s 上限下最大 1.9MB，可接受），换取无切词错误与实现简单；如需严格低延迟可换 WebSocket + 流式模型（sherpa-onnx）。
