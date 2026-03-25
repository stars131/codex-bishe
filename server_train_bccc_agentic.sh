#!/bin/bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"
DATASET_DIR="${DATASET_DIR:-data/数据集BCCC-CSE-CIC-IDS-2018}"
SAMPLE_PER_MEMBER="${SAMPLE_PER_MEMBER:-120}"
MAX_MEMBERS="${MAX_MEMBERS:-6}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MEMBER_KEYWORDS="${MEMBER_KEYWORDS:-benign,bf_ftp,bf_ssh,bot,sql_injection,infiltration}"

cd "$PROJECT_DIR"

echo "== Environment =="
$PYTHON --version
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

echo "== Install dependencies =="
$PYTHON -m pip install --upgrade pip
$PYTHON -m pip install -r requirements.txt

echo "== Run BCCC agentic workflow =="
$PYTHON run_bccc_agentic_demo.py \
  --dataset-dir "$DATASET_DIR" \
  --sample-per-member "$SAMPLE_PER_MEMBER" \
  --max-members "$MAX_MEMBERS" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --member-keywords "$MEMBER_KEYWORDS"
