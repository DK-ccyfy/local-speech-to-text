# Local AI Speech-to-Text Web Application

基于 B/S 架构的**本地**中英文语音识别 Web 系统：用户在浏览器里通过麦克风录音（或上传音频文件），系统使用部署在本地的开源 ASR 模型 **faster-whisper（Whisper / CTranslate2，CPU int8）** 将中文/英文语音转换为文字并展示在页面上。模型准备完成后可**完全离线**运行，不调用任何云端语音 API。

## 功能一览

| 功能 | 说明 | 状态 |
|---|---|---|
| 麦克风录音识别 | 开始/停止录音，状态与时长提示，单次最长 60 秒（自动结束） | ✅ 必做 |
| 音频文件上传 | WAV / WebM / MP3，≤10 MB，与录音共用同一识别接口 | ✅ 必做 |
| 中英文识别 | 页面选择 中文 / English / 自动检测 | ✅ 必做 |
| 结果展示 | 识别文本、语言、音频时长、处理耗时、一键复制 | ✅ 必做 |
| 异常处理 | 权限拒绝、无麦克风、空/损坏文件、超限、模型失败等 | ✅ 必做 |
| 实时语音识别 | 录音期间每 3 秒刷新一次识别预览（整段重识别方案） | ✅ 加分 |
| 自动语言检测 | `language=auto`，支持中英混合的尽力而为识别 | ✅ 加分 |

## 环境要求

- Python **3.12**（Windows / macOS / Linux 均可开发，验收环境为 Windows 11 x64 普通CPU）
- Chrome / Edge 最新稳定版
- 无需 Docker、Node.js、数据库、GPU/CUDA；所有 Python 依赖均为 pip wheel，**无需编译**

## 安装与启动

```bash
# 1. 安装依赖（建议先创建虚拟环境）
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt

# 2. 启动（考核统一命令）
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

浏览器访问 <http://127.0.0.1:8000>（localhost 属于安全上下文，可直接使用麦克风）。

### 模型初始化（首次运行）

- 服务启动后会在后台自动下载并加载 faster-whisper `small` 模型（约 **460 MB**，来自 HuggingFace，缓存于用户目录，不进仓库）；
- **已缓存的模型秒级离线加载**（代码优先 `local_files_only`，缓存命中时不访问网络；断网重启服务即可识别）；
- 下载期间 `/health` 返回 `{"status":"ok","model":"loading"}`，页面状态栏显示「模型加载中」；首次识别请求会等待模型就绪（带超时保护）；
- 如需提前准备模型（例如要在无外网环境演示），先在有网机器上执行一次：

```bash
python -m scripts.download_model
```

- **模型缓存完成后即可断网识别**（首次下载是唯一需要联网的步骤）；
- 网络较慢时可使用 HF 镜像：`export HF_ENDPOINT=https://hf-mirror.com`（Windows 用 `set`）。

### 配置项（环境变量，均有默认值，无需设置）

| 变量 | 默认 | 说明 |
|---|---|---|
| `ASR_MODEL` | `small` | 模型规格：`base`（145MB，更快）/ `small`（460MB，更准）等 |
| `ASR_COMPUTE_TYPE` | `int8` | CPU 量化推理 |
| `ASR_CPU_THREADS` | 自动（≤8） | CPU 推理线程数 |
| `MAX_UPLOAD_MB` | `10` | 上传文件大小上限 |
| `ASR_LOAD_TIMEOUT` | `900` | 请求等待模型就绪的超时秒数 |

## 接口

| Method | Endpoint | 功能 |
|---|---|---|
| GET | `/health` | `{"status":"ok","model":"ready\|loading\|failed"}` |
| POST | `/api/transcribe` | multipart：`file` + `language`（`zh`/`en`/`auto`） |
| GET | `/` | Web 页面 |

`POST /api/transcribe` 成功返回：

```json
{
  "text": "你好，欢迎使用语音识别系统。",
  "language": "zh",
  "duration_seconds": 5.2,
  "processing_seconds": 1.8
}
```

错误返回 `{"detail": "可读的中文错误信息"}`：参数/文件问题为 4xx（400/413/422），模型不可用或推理失败为 5xx（500/503）。交互式文档见 `/docs`。

## 测试

```bash
# 常规自动化测试（不下载模型、不依赖麦克风）
python -m pytest -q

# 真实模型验证（首次会触发模型下载，之后离线可跑）
python -m pytest -q -m realmodel
```

- 常规套件使用注入的 Mock ASR 服务，覆盖 HTTP API、音频处理（含 WebM）、语言参数、错误处理、前端资源等 **41 个用例**；
- 真实模型套件使用 `samples/chinese.wav`、`samples/english.wav` 验证中英文识别、自动语言检测与完整 HTTP 链路；
- 样本音频由 `scripts/make_samples.sh`（macOS TTS）自主生成，可自由分发，参考文本见 `samples/reference.txt`。

## 项目结构

```
speech/
├── app/
│   ├── main.py           # FastAPI 路由与错误处理
│   ├── config.py         # 环境变量配置
│   ├── audio.py          # PyAV 解码（任意格式 → 16kHz mono float32）
│   ├── errors.py         # AppError → HTTP 映射
│   └── asr/
│       ├── base.py       # ASRService 抽象（测试可注入 Mock）
│       └── whisper.py    # faster-whisper 实现（懒加载/后台预热）
├── static/               # 原生 HTML/CSS/JS 前端 + AudioWorklet
├── tests/                # pytest（Mock 套件 + realmodel 套件）
├── samples/              # 中英文样本 + 参考文本
├── scripts/              # 模型预下载、样本生成
├── docs/                 # 需求/设计/测试报告/AI开发记录
├── requirements.txt
└── pytest.ini
```

## 技术选型说明（摘要）

选择 **faster-whisper**：开源（MIT 系）、本地推理、原生支持中英文、CTranslate2 CPU int8 在普通开发机上对 5–15 秒音频通常数秒内完成；纯 pip wheel 安装（含 PyAV 自带 FFmpeg 解码能力，无需系统 ffmpeg）；模型缓存后完全离线。备选对比（openai-whisper/Vosk/FunASR/sherpa-onnx）见 `docs/design.md`。

详细文档：[需求理解](docs/requirements.md) · [系统设计](docs/design.md) · [测试报告](docs/test-report.md) · [AI 辅助开发记录](docs/ai-development.md)
