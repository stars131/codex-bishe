#!/bin/bash
set -euo pipefail

# Realistic Linux/GPU server guide for the current repository state.
# This script assumes you want to run the supported CIC-IDS-2017 pipeline.

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"
CONFIG="${CONFIG:-src/config/config_server.yaml}"
DATA_DIR="${DATA_DIR:-data/raw/CIC-IDS-2017}"
GPU_ID="${GPU_ID:-0}"

cd "$PROJECT_DIR"

echo "== Environment =="
$PYTHON --version
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

echo "== Install dependencies =="
$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install -r requirements.txt

echo "== Notes =="
echo "1. This script targets the supported CIC-IDS-2017 pipeline."
echo "2. Heterogeneous dataset fusion and threat-intel feature augmentation are disabled by default."
echo "3. Train and evaluate with the same config file."

echo "== Preprocess =="
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON main.py \
  --data_dir "$DATA_DIR" \
  --mode preprocess \
  --config "$CONFIG"

echo "== Train =="
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON main.py \
  --mode train \
  --config "$CONFIG"

LATEST_EXP=$($PYTHON - <<'PY'
import os
root = "outputs"
experiments = [
    d for d in os.listdir(root)
    if os.path.isdir(os.path.join(root, d)) and d.startswith("exp_")
]
print(sorted(experiments)[-1] if experiments else "")
PY
)

if [ -z "$LATEST_EXP" ]; then
  echo "No experiment directory found under outputs/exp_*"
  exit 1
fi

echo "== Evaluate =="
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON main.py \
  --mode evaluate \
  --experiment "$LATEST_EXP" \
  --config "$CONFIG"

echo "== Report =="
CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON main.py \
  --mode report \
  --experiment "$LATEST_EXP" \
  --config "$CONFIG"

echo "== Outputs =="
echo "Experiment: outputs/$LATEST_EXP"
echo "Checkpoint: outputs/$LATEST_EXP/checkpoints/best_model.pth"
echo "Results: outputs/$LATEST_EXP/results/"
echo "TensorBoard: tensorboard --logdir outputs --host 0.0.0.0"
