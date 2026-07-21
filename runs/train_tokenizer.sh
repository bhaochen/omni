#!/usr/bin/env bash
# Tokenizer 训练启动脚本（仅供学习和参考，不建议重复训练 tokenizer）
# 用法: bash runs/train_tokenizer.sh [额外参数...]
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:${PYTHONPATH}"
exec uv run python -m trainers.lm.train_tokenizer "$@"
