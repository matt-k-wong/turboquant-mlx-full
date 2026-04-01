#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# quantize_qwen.sh — offline TurboQuant weight quantisation
#
# Usage:
#   bash scripts/quantize_qwen.sh [model] [output] [bits] [n_skip] [group_size]
#
# Examples:
#   bash scripts/quantize_qwen.sh \
#       mlx-community/Qwen2.5-14B-Instruct-4bit \
#       ~/tq_models/Qwen2.5-14B-tq4bit 4
#
#   bash scripts/quantize_qwen.sh \
#       mlx-community/Qwen2.5-32B-Instruct-4bit \
#       ~/tq_models/Qwen2.5-32B-tq4bit 4
#
# MEMORY NOTE: Always use an already-quantised mlx-community model as input.
# Raw fp16 32B models require ~64 GB. Starting from 4-bit stays within 16 GB.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

MODEL="${1:-mlx-community/Qwen2.5-14B-Instruct-4bit}"
OUTPUT="${2:-${HOME}/tq_models/Qwen2.5-14B-tq4bit}"
BITS="${3:-4}"
N_SKIP="${4:-2}"
GROUP_SIZE="${5:-128}"

echo ""
echo "  TurboQuant Offline Weight Quantiser"
echo "  ────────────────────────────────────"
echo "  Model      : $MODEL"
echo "  Output     : $OUTPUT"
echo "  Bits       : $BITS"
echo "  Group size : $GROUP_SIZE"
echo "  Skip layers: $N_SKIP (first + last)"
echo ""

python3 -c "
import sys; sys.path.insert(0, '.')
from turboquant_mlx_full.quantize_weights import quantize_model
quantize_model(
    model_or_path='$MODEL', output_path='$OUTPUT',
    bits=$BITS, group_size=$GROUP_SIZE, n_skip_layers=$N_SKIP, verbose=True,
)
"

echo ""
echo "  Done! Model saved to: $OUTPUT"
echo ""
echo "  Chat with it:"
echo "    python examples/qwen_chat.py --model $OUTPUT --tq-weights --kv-bits 3"
