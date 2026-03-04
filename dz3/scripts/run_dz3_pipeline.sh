#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install -r requirements.txt

ruff check .
ruff format --check .
python3 -m pytest -q

python3 -m src.rl.datasets --make-train --make-val --print-summary
python3 -m src.rl.gold --input data/train/train.jsonl --output data/train/train.jsonl

python3 -m src.rl.train_sft --config configs/train.yaml --dry-run
python3 -m src.rl.train_grpo --config configs/train.yaml --dry-run
python3 -m src.rl.train_sft_grpo --config configs/train.yaml --dry-run
python3 -m src.rl.train_srft --config configs/train.yaml --dry-run

if python3 - <<'PY'
import torch
print(torch.cuda.is_available())
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
then
  echo "[dz3] GPU detected, running full experiment pipeline"

  python3 -m src.rl.hard_mining --config configs/hard.yaml
  python3 -m src.rl.gold --input data/hard/hard_train.jsonl --output data/hard/hard_train.jsonl

  python3 -m src.rl.passk_eval --config configs/train.yaml --model baseline --split val
  python3 -m src.rl.passk_eval --config configs/train.yaml --model baseline --split hard_val

  python3 -m src.rl.train_grpo --config configs/train.yaml
  python3 -m src.rl.train_sft --config configs/train.yaml
  python3 -m src.rl.train_sft_grpo --config configs/train.yaml
  python3 -m src.rl.train_srft --config configs/train.yaml

  python3 -m src.rl.evaluate --config configs/train.yaml --model grpo --split both
  python3 -m src.rl.evaluate --config configs/train.yaml --model sft --split both
  python3 -m src.rl.evaluate --config configs/train.yaml --model sft_grpo --split both
  python3 -m src.rl.evaluate --config configs/train.yaml --model srft --split both

  python3 -m src.rl.plot_passk
  python3 -m src.rl.plot_training
else
  echo "[dz3] GPU not detected. Completed checks + dry-run commands only."
fi
