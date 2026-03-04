#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
mkdir -p logs/night

echo "[night] Using python: ${PYTHON_BIN}"
echo "[night] PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"

${PYTHON_BIN} -m src.rl.hard_mining --config configs/hard.night.yaml | tee logs/night/01_hard_mining.log
${PYTHON_BIN} -m src.rl.gold --input data/hard/hard_train.jsonl --output data/hard/hard_train.jsonl | tee logs/night/02_gold_hard_train.log
${PYTHON_BIN} -m src.rl.passk_eval --config configs/train.night.yaml --model baseline --split hard_val --batch-size 2 --seed 2026 | tee logs/night/03_baseline_hard_val.log

${PYTHON_BIN} -m src.rl.train_sft_grpo --config configs/train.night.yaml | tee logs/night/04_train_sft_grpo.log
${PYTHON_BIN} -m src.rl.train_srft --config configs/train.night.yaml | tee logs/night/05_train_srft.log

${PYTHON_BIN} -m src.rl.evaluate --config configs/train.night.yaml --model baseline --split both | tee logs/night/06_eval_baseline.log
${PYTHON_BIN} -m src.rl.evaluate --config configs/train.night.yaml --model sft_grpo --split both | tee logs/night/07_eval_sft_grpo.log
${PYTHON_BIN} -m src.rl.evaluate --config configs/train.night.yaml --model srft --split both | tee logs/night/08_eval_srft.log

${PYTHON_BIN} -m src.rl.plot_passk | tee logs/night/09_plot_passk.log
${PYTHON_BIN} -m src.rl.plot_training | tee logs/night/10_plot_training.log

echo "[night] Completed successfully."
