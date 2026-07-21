#!/usr/bin/env bash
# LM (纯文本) 训练启动脚本
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec uv run python -m trainers.lm.full_sft --config "$ROOT/configs/lm.yaml" "$@"
