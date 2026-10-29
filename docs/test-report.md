# 测试报告（test-report.md）

## 1. 测试环境

| 项目 | 值 |
|---|---|
| 开发机 | macOS (Apple Silicon, darwin 25.5.0, arm64) |
| Python | 3.12.14（venv） |
| 关键依赖 | fastapi 0.162 / uvicorn / faster-whisper 1.2.1 / ctranslate2 4.8.4 / av 19.0.2 / pytest 9.2 |
| 模型 | faster-whisper `small`，CPU int8（权重已缓存于 `~/.cache/huggingface`，473 MB） |
| 浏览器 | ZCode 内置 Chromium（Chromium 内核，等价 Chrome 行为） |

> 验收目标环境为 Windows 11 x64：所有依赖（ctranslate2/onnxruntime/av/fastapi）在 PyPI 均提供 Win x64 wheel，无需编译；统一启动命令见 README。

## 2. 自动化测试（Mock，无需模型/麦克风）

命令与结果（2026-10-28 实测）：

```
$ python -m pytest -q
41 passed, 5 deselected in 0.24s
```

（5 deselected 为真实模型用例，见第 3 节；默认套件不触发模型下载。）

覆盖范围：

| 范围 | 用例（文件） | 结果 |
|---|---|---|
| HTTP API | `/health` 正常与模型加载中两种状态；`/api/transcribe` 正常 zh/en/auto、字段类型与时长精度 | ✅ 全过 |
| 语言参数 | 非法值（fr/chinese/e/zh-CN）400；空串 4xx；大小写与空白容错（" ZH "→200）；缺参 422 | ✅ 全过 |
| 音频处理 | WAV 解码时长/格式；44.1kHz 立体声重采样；WebM/Opus 解码；空字节/垃圾字节/无数据帧 WAV 均 400 | ✅ 全过 |
| 文件限制 | 10MB+1 → 413；空文件 → 400 | ✅ 全过 |
| 模型集成（桩） | 语言/VAD/简体提示词参数透传；auto 检测中文后二次识别；推理异常→500「语音识别失败」；加载失败/未就绪→503；**缓存优先加载（先 local_files_only，未命中才联网）**；联网回退；加载失败置 failed | ✅ 全过 |
| 错误处理 | Mock 抛 RuntimeError → 兜底 500「服务器内部错误」；AppError 503 透传 | ✅ 全过 |
| 前后端流程 | `GET /` 页面与 3 个静态资源可达；app.js 含 /api/transcribe、60s 上限、getUserMedia、权限错误映射 | ✅ 全过 |

## 3. 真实模型测试（`-m realmodel`）

```
$ python -m pytest -q -m realmodel
5 passed, 41 deselected in 85.85s
```

实测输出（含中文简体引导后的最终效果）：

| 用例 | 音频 | 识别结果 | 推理耗时 |
|---|---|---|---|
| 中文 | samples/chinese.wav（5.043s） | `大家好,今天天气很好,我们一起去公园散步吧。` | 1.2–2.0s |
| 英文 | samples/english.wav（6.277s） | `Hello, this is a local speech recognition system running on my computer. It can turn speech into text.`（与参考文本完全一致） | 1.1s |
| 自动检测 | chinese.wav, language=None | 非空中文文本（检测为 zh，并按简体重写输出） | ~2×单次 |
| HTTP 全链路 | multipart 上传 chinese.wav | 200，四字段齐全，关键词重合 ≥2/4 | 1.1–1.2s |
| 耗时口径 | — | processing_seconds 排除模型加载时间 | ✅ |

参考文本（samples/reference.txt）：中文 4 关键词命中 4/4（大家好/天气/公园/散步，简体引导后），英文关键词 4/4。

## 4. 离线运行验证（模拟断网）

以 `HF_HUB_OFFLINE=1` 强制离线启动（端口 8001），模型从本地缓存秒级加载：

```
$ curl -s http://127.0.0.1:8001/health
{"status":"ok","model":"ready"}
$ curl -s -X POST .../api/transcribe -F file=@samples/chinese.wav -F language=zh
{"text":"大家好,今天天气很好,我们一起去公园散步吧。","language":"zh","duration_seconds":5.043,"processing_seconds":1.143}
$ curl -s -X POST .../api/transcribe -F file=@samples/english.wav -F language=en
{"text":"Hello, ... turn speech into text.","language":"en","duration_seconds":6.277,"processing_seconds":1.09}
```

代码层面另有三重保障：加载时优先 `local_files_only=True`（缓存命中即完全联网零依赖，单测覆盖）；`/health` 不依赖网络；识别过程无任何外部调用。

## 5. 浏览器手动验证（实测记录）

环境：服务 `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`，Chromium 内核浏览器访问 `http://127.0.0.1:8000`。页面截图见 `docs/screenshot-browser.png`。

| # | 场景 | 结果 |
|---|---|---|
| 1 | 页面加载 | 标题/语言选择/录音/上传/结果区齐全，模型状态「模型已就绪」 |
| 2 | 上传中文 WAV（页面操作） | ✅ 识别完成：`大家好,今天天气很好,我们一起去公园散步吧。`，时长 5.043s，耗时 1.194s，复制按钮激活 |
| 3 | 上传英文 WAV（切 English 模式） | ✅ 文本与参考完全一致，时长 6.277s，耗时 1.134s |
| 4 | 上传 .txt 文件 | ✅ 红色错误框「不支持的文件格式 .txt，请上传 WAV / WebM / MP3 音频文件。」，页面不崩溃 |
| 5 | 浏览器麦克风录音 | ✅ 授权后开始录音：状态「录音中」红点脉冲、计时器 `00:41 / 00:60` 走字、开始钮禁用/停止钮激活 |
| 6 | 实时预览（加分） | ✅ 录音 42 秒期间后端收到 14 次整段预览请求（约每 3 秒 1 次，共 16 次 POST 含 2 次上传） |
| 7 | 停止录音（无人声环境） | ✅ 最终识别返回空文本 → 页面显示「（未识别到语音内容）」，41.904s 音频处理 0.071s（VAD 全滤），状态复位可再录 |
| 8 | 复制文本 | ✅ 点击后提示「已复制到剪贴板」（clipboard API） |
| 9 | 权限被拒场景 | IAB 环境麦克风自动授予，无法真实触发拒绝弹窗；该路径由前端 `describeMicError`（NotAllowedError→「请在浏览器地址栏的站点权限设置中允许麦克风访问」）实现并经代码走查，错误提示已改为顶部悬浮弹框；建议在 Chrome 中手动复核一次（点击地址栏麦克风图标选「阻止」→ 点开始录音） |
| 10 | Windows 11 复现 | 依赖全为 PyPI wheel（ctranslate2/onnxruntime/av 均有 win_amd64 版本），统一命令启动；未在本机实测 Windows（本报告环境为 macOS 开发机），属已知限制。README 已提供 Windows 全新机器完整步骤（含 Python 安装、venv 激活、国内 HF 镜像配置） |

**2026-10-29 用户本机（Chrome）人工复测确认：**

| 场景 | 结果 |
|---|---|
| 静默录音约 10 秒 → 手动点「停止录音」 | ✅ 正常结束，显示「（未识别到语音内容）」，无报错，可立即再录 |
| 60 秒上限自动停止 | ✅ 到 60 秒系统自动结束录音并给出提示，随后完成识别 |

（复测过程中发现并修复了三个前端问题：错误框因 CSS 优先级常驻显示 → 改为顶部悬浮弹框；双击「开始录音」的并发竞态 → 同步防重入守卫；`audioCtx.close()` 可能阻塞停止流程 → 改为非阻塞清理。过程记录见 `docs/ai-development.md` 问题表 #7/#8。）

## 6. 已知问题与限制

1. **截断但可部分解码的音频**（如传输中断的 WAV）：PyAV 容错解码有效部分并返回其可识别内容，而非报错；完全无法解码才 400。工程取舍：能听清的部分就返回。
2. **自动检测/中英混读**：Whisper 按整段检测语言，段落级中英切换不保证逐句切换；UI 已标注「实验性」。检测为中文时会自动做一次简体引导重识别。
3. **实时预览延迟**：约 3 秒一拍且随音频变长推理耗时增长（60s 上限内可接受）；非流式方案，符合题目「不要求毫秒级延迟」的约定。
4. **繁/简体**：Whisper 中文有繁体倾向，已用 initial_prompt 引导简体（zh 模式实测全简体）；极端口音下仍可能出现个别繁体字。
5. **Windows 实测**：开发在 macOS 完成；依赖均为 Windows wheel、无编译，预期按 README 可直接运行，但未在真实 Win11 机器上验证。
6. **麦克风权限拒绝弹窗**：自动化环境无法触发真实拒绝，见第 5 节第 9 行说明。
