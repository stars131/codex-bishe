#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"
INPUT_FILE="${INPUT_FILE:-}"
THREAT_INTEL_FILE="${THREAT_INTEL_FILE:-}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-256}"

cd "$PROJECT_DIR"

if [ -z "$INPUT_FILE" ]; then
  echo "INPUT_FILE is required"
  echo "Example:"
  echo "  INPUT_FILE=data/processed/um_nids_cicids2018.csv bash server_train_um_nids_agentic.sh"
  exit 1
fi

echo "== Environment =="
$PYTHON --version
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

echo "== Install dependencies =="
$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install -r requirements.txt

echo "== Run UM-NIDS agentic pipeline =="
if [ -n "$THREAT_INTEL_FILE" ]; then
  $PYTHON run_um_nids_agentic.py \
    --input "$INPUT_FILE" \
    --threat-intel "$THREAT_INTEL_FILE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE"
else
  $PYTHON run_um_nids_agentic.py \
    --input "$INPUT_FILE" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE"
fi
