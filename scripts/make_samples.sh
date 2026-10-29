#!/usr/bin/env bash
# Generate the two sample audio files (self-made via macOS TTS => free to distribute).
#
#   samples/chinese.wav   Mandarin, ~8s
#   samples/english.wav   English,  ~7s
#
# Output: 16 kHz, mono, 16-bit PCM WAV — exactly what the ASR pipeline expects.
# Reference transcripts are written to samples/reference.txt.
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p samples

ZH_TEXT="大家好，今天天气很好，我们一起去公园散步吧。"
EN_TEXT="Hello, this is a local speech recognition system running on my computer. It can turn speech into text."

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

say -v Tingting -o "$tmp/chinese.aiff" "$ZH_TEXT"
say -v Samantha -o "$tmp/english.aiff" "$EN_TEXT"

afconvert -f WAVE -d LEI16@16000 -c 1 "$tmp/chinese.aiff" samples/chinese.wav
afconvert -f WAVE -d LEI16@16000 -c 1 "$tmp/english.aiff" samples/english.wav

cat > samples/reference.txt <<EOF
chinese.wav: ${ZH_TEXT}
english.wav: ${EN_TEXT}
EOF

echo "wrote samples/chinese.wav samples/english.wav samples/reference.txt"
afinfo samples/chinese.wav | grep -E 'duration|data format' || true
afinfo samples/english.wav | grep -E 'duration|data format' || true
