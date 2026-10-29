/* 本地语音识别系统 - 前端逻辑
 *
 * 录音：getUserMedia + Web Audio API（AudioWorklet，降级 ScriptProcessor），
 *       直接在浏览器内采集 16kHz 单声道 PCM 并编码为 WAV，
 *       因此“麦克风录音”和“文件上传”走同一条 /api/transcribe 接口。
 * 实时预览（加分）：录音期间每 3 秒把当前累计音频整段识别一次，持续刷新文本。
 */

"use strict";

const SAMPLE_RATE = 16000;      // 与后端 faster-whisper 期望一致
const MAX_SECONDS = 60;         // 单次录音上限（题目要求）
const REALTIME_INTERVAL_MS = 3000;
const MAX_UPLOAD_BYTES = 10 * 1024 * 1024;
const ACCEPTED_EXTENSIONS = ["wav", "webm", "mp3"];
const MIN_RECORD_SECONDS = 0.2; // 过短的录音直接在前端拦截

const $ = (id) => document.getElementById(id);
const els = {
  lang: $("langSelect"),
  btnStart: $("btnStart"), btnStop: $("btnStop"),
  micStatus: $("micStatus"), recordTimer: $("recordTimer"),
  uploadFile: $("uploadFile"), btnUpload: $("btnUpload"),
  stateBadge: $("stateBadge"), resultText: $("resultText"),
  metaLanguage: $("metaLanguage"), metaDuration: $("metaDuration"),
  metaProcessing: $("metaProcessing"),
  btnCopy: $("btnCopy"), copyHint: $("copyHint"),
  errorBox: $("errorBox"), errorMessage: $("errorMessage"), btnRetry: $("btnRetry"),
  modelStatus: $("modelStatus"),
};

/* ------------------------------------------------------------------ state */
const state = {
  recording: false,
  recognizing: false,     // 一次最终识别请求进行中
  audioCtx: null,
  mediaStream: null,
  workletNode: null,
  sourceNode: null,
  muteNode: null,
  usingWorklet: false,
  chunks: [],             // Float32Array chunks @16kHz mono
  totalSamples: 0,
  timerInterval: null,
  realtimeInterval: null,
  realtimeBusy: false,
  lastJob: null,          // { blob, filename } 用于失败重试
};

/* ------------------------------------------------------------- ui helpers */
function setBadge(name, label) {
  els.stateBadge.className = `badge ${name}`;
  els.stateBadge.textContent = label;
}

function setMicStatus(text, cls = "") {
  els.micStatus.textContent = text;
  els.micStatus.className = `status-text ${cls}`;
}

function setBusy(busy) {
  state.recognizing = busy;
  els.btnStart.disabled = busy || state.recording;
  els.btnStop.disabled = busy || !state.recording;
  els.btnUpload.disabled = busy;
}

function showError(message, canRetry = false) {
  els.errorMessage.textContent = message;
  els.errorBox.hidden = false;
  els.btnRetry.hidden = !canRetry;
}

function hideError() {
  els.errorBox.hidden = true;
  els.btnRetry.hidden = true;
}

function fmtClock(seconds) {
  const s = Math.floor(seconds), m = Math.floor(s / 60);
  return `${String(m).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
}

/* ------------------------------------------------------------ audio utils */
function concatChunks() {
  const out = new Float32Array(state.totalSamples);
  let offset = 0;
  for (const chunk of state.chunks) { out.set(chunk, offset); offset += chunk.length; }
  return out;
}

/** Float32 mono PCM -> 16-bit PCM WAV Blob */
function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeStr = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };
  writeStr(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeStr(8, "WAVE");
  writeStr(12, "fmt ");
  view.setUint32(16, 16, true);        // PCM chunk size
  view.setUint16(20, 1, true);         // PCM format
  view.setUint16(22, 1, true);         // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true);         // block align
  view.setUint16(34, 16, true);        // bits per sample
  writeStr(36, "data");
  view.setUint32(40, samples.length * 2, true);
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([view], { type: "audio/wav" });
}

/* -------------------------------------------------------------- recording */
function describeMicError(err) {
  const name = (err && err.name) || "";
  if (name === "NotAllowedError" || name === "PermissionDeniedError") {
    return "麦克风权限被拒绝：请在浏览器地址栏的站点权限设置中允许麦克风访问，然后重试。";
  }
  if (name === "NotFoundError" || name === "DevicesNotFoundError") {
    return "未检测到可用的麦克风设备，请检查麦克风是否已连接并启用。";
  }
  if (name === "NotReadableError" || name === "TrackStartError") {
    return "无法读取麦克风：设备可能被其他程序占用，请关闭后重试。";
  }
  if (name === "OverconstrainedError") {
    return "当前麦克风不满足录音要求，请更换设备后重试。";
  }
  return `无法访问麦克风：${(err && err.message) || name || "未知错误"}`;
}

function onPcmChunk(chunk) {
  state.chunks.push(chunk);
  state.totalSamples += chunk.length;
}

async function startRecording() {
  if (state.recording || state.recognizing) return;
  hideError();

  if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) {
    clientError("当前浏览器不支持麦克风采集，请使用最新版 Chrome 或 Edge 通过 http://127.0.0.1:8000 访问。");
    return;
  }

  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
  } catch (err) {
    showError(describeMicError(err));
    setBadge("error", "录音失败");
    return;
  }

  const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: SAMPLE_RATE });
  try {
    await ctx.resume();
  } catch (_) { /* already running */ }

  const source = ctx.createMediaStreamSource(stream);
  let node;
  let usingWorklet = false;
  try {
    await ctx.audioWorklet.addModule("/static/recorder-worklet.js");
    node = new AudioWorkletNode(ctx, "pcm-recorder", { numberOfOutputs: 1 });
    node.port.onmessage = (e) => onPcmChunk(e.data);
    usingWorklet = true;
  } catch (_) {
    // Fallback for browsers without AudioWorklet
    node = ctx.createScriptProcessor(4096, 1, 1);
    node.onaudioprocess = (e) => onPcmChunk(new Float32Array(e.inputBuffer.getChannelData(0)));
  }
  // Chrome only pulls nodes connected to the destination; route through a
  // zero-gain node to keep the graph alive without echoing the mic.
  const mute = ctx.createGain();
  mute.gain.value = 0;
  source.connect(node);
  node.connect(mute);
  mute.connect(ctx.destination);

  Object.assign(state, {
    recording: true,
    audioCtx: ctx,
    mediaStream: stream,
    workletNode: node,
    sourceNode: source,
    muteNode: mute,
    usingWorklet,
    chunks: [],
    totalSamples: 0,
  });

  setBadge("recording", "录音中");
  setMicStatus("录音中", "recording");
  els.btnStart.disabled = true;
  els.btnStop.disabled = false;
  els.btnUpload.disabled = true;

  state.timerInterval = setInterval(() => {
    const seconds = state.totalSamples / SAMPLE_RATE;
    els.recordTimer.textContent = `${fmtClock(seconds)} / 00:60`;
    if (seconds >= MAX_SECONDS) {
      stopRecording(true);
    }
  }, 100);

  state.realtimeInterval = setInterval(recognizePartial, REALTIME_INTERVAL_MS);
}

async function stopRecording(auto = false) {
  if (!state.recording) return;
  state.recording = false;

  clearInterval(state.timerInterval);
  clearInterval(state.realtimeInterval);
  state.timerInterval = state.realtimeInterval = null;

  // teardown audio graph
  try { state.sourceNode && state.sourceNode.disconnect(); } catch (_) {}
  try { state.workletNode && state.workletNode.disconnect(); } catch (_) {}
  try { state.muteNode && state.muteNode.disconnect(); } catch (_) {}
  if (state.mediaStream) state.mediaStream.getTracks().forEach((t) => t.stop());
  if (state.audioCtx && state.audioCtx.state !== "closed") {
    try { await state.audioCtx.close(); } catch (_) {}
  }

  const duration = state.totalSamples / SAMPLE_RATE;
  els.recordTimer.textContent = "";

  if (duration < MIN_RECORD_SECONDS) {
    setMicStatus("空闲");
    els.btnStart.disabled = false;
    els.btnStop.disabled = true;
    els.btnUpload.disabled = false;
    setBadge("error", "录音太短");
    showError(auto
      ? "已达到 60 秒上限自动结束，但未采集到有效音频，请重试。"
      : "录音时长太短，请长按说话后再停止。");
    return;
  }

  setMicStatus(auto ? `已达 60 秒上限，自动结束（${duration.toFixed(1)} 秒）` : `已停止（${duration.toFixed(1)} 秒）`);
  setBadge("working", "识别中");

  const blob = encodeWav(concatChunks(), SAMPLE_RATE);
  await submitAudio(blob, "recording.wav");
}

/* ------------------------------------------------------- realtime preview */
async function recognizePartial() {
  if (!state.recording || state.realtimeBusy || state.totalSamples < SAMPLE_RATE) return;
  state.realtimeBusy = true;
  try {
    const blob = encodeWav(concatChunks(), SAMPLE_RATE);
    const data = await postTranscribe(blob, "recording.wav");
    if (data && data.text) {
      els.resultText.value = data.text + "\n（实时预览，录音结束后给出最终结果）";
      setBadge("recording", "录音中·实时识别");
      els.metaLanguage.textContent = languageLabel(data.language);
      els.metaDuration.textContent = `${data.duration_seconds} 秒`;
      els.metaProcessing.textContent = `${data.processing_seconds} 秒`;
      els.btnCopy.disabled = true; // 等最终结果再允许复制
    }
  } catch (_) {
    /* 实时预览失败不打扰用户，最终识别仍会给出结果或错误 */
  } finally {
    state.realtimeBusy = false;
  }
}

/* --------------------------------------------------------------- requests */
function languageLabel(lang) {
  return { zh: "中文", en: "英文", auto: "自动检测" }[lang] || lang || "-";
}

/** Client-side validation failure: show error and mark the badge. */
function clientError(message) {
  setBadge("error", "操作无效");
  showError(message);
}

function errDetail(payload, status) {
  if (payload && typeof payload.detail === "string") return payload.detail;
  if (Array.isArray(payload && payload.detail)) {
    return payload.detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
  }
  return `请求失败（HTTP ${status}）`;
}

async function postTranscribe(blob, filename) {
  const form = new FormData();
  form.append("file", blob, filename);
  form.append("language", els.lang.value);
  const res = await fetch("/api/transcribe", { method: "POST", body: form });
  const payload = await res.json().catch(() => null);
  if (!res.ok) throw new Error(errDetail(payload, res.status));
  return payload;
}

async function submitAudio(blob, filename) {
  hideError();
  state.lastJob = { blob, filename };
  setBusy(true);
  setBadge("working", "识别中");
  try {
    const data = await postTranscribe(blob, filename);
    els.resultText.value = data.text || "（未识别到语音内容）";
    els.metaLanguage.textContent = languageLabel(data.language);
    els.metaDuration.textContent = `${data.duration_seconds} 秒`;
    els.metaProcessing.textContent = `${data.processing_seconds} 秒`;
    els.btnCopy.disabled = !data.text;
    els.copyHint.textContent = "";
    setBadge("done", "识别完成");
    setMicStatus("空闲");
  } catch (err) {
    setBadge("error", "识别失败");
    showError(`识别失败：${err.message}`, true);
  } finally {
    setBusy(false);
  }
}

/* ----------------------------------------------------------------- upload */
function uploadChosenFile() {
  if (state.recording || state.recognizing) return;
  hideError();
  const file = els.uploadFile.files && els.uploadFile.files[0];
  if (!file) {
    clientError("请先选择要上传的音频文件。");
    return;
  }
  const ext = (file.name.split(".").pop() || "").toLowerCase();
  if (!ACCEPTED_EXTENSIONS.includes(ext)) {
    clientError(`不支持的文件格式 .${ext || "?"}，请上传 WAV / WebM / MP3 音频文件。`);
    return;
  }
  if (file.size === 0) {
    clientError("上传的文件为空，请检查后重试。");
    return;
  }
  if (file.size > MAX_UPLOAD_BYTES) {
    clientError(`文件大小 ${(file.size / 1024 / 1024).toFixed(1)} MB 超过 10 MB 上限。`);
    return;
  }
  setBadge("working", "识别中");
  submitAudio(file, file.name);
}

/* ------------------------------------------------------------------- copy */
async function copyResult() {
  const text = els.resultText.value;
  if (!text) return;
  try {
    await navigator.clipboard.writeText(text);
    els.copyHint.textContent = "已复制到剪贴板";
  } catch (_) {
    // http fallback (e.g. non-secure origin)
    els.resultText.select();
    const ok = document.execCommand && document.execCommand("copy");
    els.copyHint.textContent = ok ? "已复制到剪贴板" : "复制失败，请手动选择文本复制";
  }
  setTimeout(() => (els.copyHint.textContent = ""), 2000);
}

/* ------------------------------------------------------------- model info */
async function refreshModelStatus() {
  try {
    const res = await fetch("/health");
    const data = await res.json();
    const labels = { ready: "模型已就绪", loading: "模型加载中…", idle: "模型未加载", failed: "模型加载失败" };
    els.modelStatus.textContent = labels[data.model] || data.model || "";
  } catch (_) {
    els.modelStatus.textContent = "";
  }
}

/* ------------------------------------------------------------------ wireup */
els.btnStart.addEventListener("click", startRecording);
els.btnStop.addEventListener("click", () => stopRecording(false));
els.btnUpload.addEventListener("click", uploadChosenFile);
els.btnCopy.addEventListener("click", copyResult);
els.btnRetry.addEventListener("click", () => {
  if (state.lastJob) submitAudio(state.lastJob.blob, state.lastJob.filename);
});

setBadge("idle", "空闲");
refreshModelStatus();
setInterval(refreshModelStatus, 10000);
