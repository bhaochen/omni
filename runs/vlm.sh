#!/usr/bin/env bash
# VLM (文本 + 视觉) 训练启动脚本
# 用法: bash runs/vlm.sh [额外参数...]   例如  bash runs/vlm.sh --config configs/model/vlm_moe.yaml
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH}"
exec python -m trainers.vlm.full_sft --config "$ROOT/configs/model/vlm.yaml" "$@"
