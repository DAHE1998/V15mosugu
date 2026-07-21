#!/usr/bin/env bash
# V15mosugu 完整 pipeline 一键运行（到 VLM 前停）
# 用法: ./run.sh <video.mp4> [base_output_dir]
# 输出: base_output_dir/video_name/STEP_NAME/...

set -eo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

VIDEO="$1"
BASE_DIR="${2:-$(dirname "$VIDEO")}"
VIDEO_NAME="$(basename "$VIDEO" | sed 's/\.[^.]*$//')"
WORK_DIR="$BASE_DIR/$VIDEO_NAME"

# conda python
PY="/home/dahe/miniconda3/envs/amaterasu/bin/python"

echo "=============================================="
echo "V15mosugu Pipeline (scdet_vulkan)"
echo "  Video: $VIDEO"
echo "  Work:  $WORK_DIR"
echo "=============================================="

mkdir -p "$WORK_DIR"

# ── 00: scdet_vulkan (C, 全GPU) ──
echo
echo "=== [00] scdet_vulkan ==="
"$SCRIPT_DIR/00_scdet" "$VIDEO" "$WORK_DIR"

# ── 01: skeleton ──
echo
echo "=== [01] skeleton ==="
"$SCRIPT_DIR/01_skeleton" "$WORK_DIR"

# ── 02: select_frames ──
echo
echo "=== [02] select_frames ==="
$PY "$SCRIPT_DIR/02_select_frames.py" "$WORK_DIR"

# ── 03: ASR ──
echo
echo "=== [03] ASR ==="
$PY "$SCRIPT_DIR/asr.py" "$WORK_DIR"

# ── 04: DINO visual cluster ──
echo
echo "=== [04] dino_cluster ==="
$PY "$SCRIPT_DIR/dino_cluster.py" "$WORK_DIR"

# ── 05: text cluster ──
echo
echo "=== [05] text_cluster ==="
$PY "$SCRIPT_DIR/text_cluster.py" "$WORK_DIR"

# ── 06: merge ──
echo
echo "=== [06] merge ==="
$PY "$SCRIPT_DIR/graph_merge.py" "$WORK_DIR"

# ── 06.5: fragment ──
echo
echo "=== [fragment] fragment ==="
$PY "$SCRIPT_DIR/fragment.py" "$WORK_DIR"

echo
echo "=============================================="
echo "Done! Output: $WORK_DIR"
echo "  Stop before VLM. Run vlm.py manually."
echo "=============================================="
