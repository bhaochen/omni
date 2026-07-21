#!/usr/bin/env bash
# VLM (文本 + 视觉) 训练启动脚本
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec uv run python -m trainers.vlm.full_sft --config "$ROOT/configs/vlm.yaml" "$@"
