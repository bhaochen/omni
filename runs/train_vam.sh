#!/usr/bin/env bash
# VAM (文本 + 视觉 + 语音 全模态) 训练启动脚本
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec uv run python -m trainers.vam.full_sft --config "$ROOT/configs/vam.yaml" "$@"
