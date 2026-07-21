#!/usr/bin/env bash
# LM (纯文本) 训练启动脚本
# 用法: bash runs/lm.sh [额外参数...]   例如  bash runs/lm.sh --epochs 5
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH}"
exec python -m trainers.lm.full_sft --config "$ROOT/configs/lm.yaml" "$@"
