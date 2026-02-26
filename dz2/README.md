# DZ2: RL Agent for One-Step Modular Arithmetic Environment

This repository implements Homework 2 for the course "Selected Topics in AI Research".
Task: one-step environment where an LLM computes an arithmetic expression modulo `M`.

## Implemented Components

- `Env` + `Verifier` for modular arithmetic expressions.
- Difficulty mapping (`1..10`) and direct hyper-parameter overrides.
- Train dataset sampling + fixed reproducible eval datasets.
- GRPO training script for `Qwen2.5-1.5B-Instruct` with Unsloth.
- vLLM inference + evaluation metrics + grouped bar plotting.
- Unit tests for generation, parsing, verification, reproducibility, and train dry-run.

## Project Structure

- `src/base/`: base interfaces (`Data`, `Env`, `Verifier`)
- `src/envs/arithmetic_mod/`: environment, verifier, prompt, expression generator
- `src/rl/`: datasets, reward, training, inference, evaluation, plotting, HF upload
- `configs/`: data and training configs
- `scripts/run_local_pipeline.sh`: one-command local pipeline
- `tests/`: unit/smoke tests
- `reports/report.md`: experiment report

## Setup (Local)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## One Command (Local Runtime)

Run everything that is guaranteed to work locally in one command:

```bash
bash scripts/run_local_pipeline.sh
```

This command performs:

- lint and format checks (`ruff`)
- tests (`pytest`)
- train/eval dataset generation
- GRPO dry-run

If local hardware supports Unsloth + vLLM (GPU runtime), the script will automatically continue with full training, evaluation, and plotting.

## Manual Commands

```bash
python -m src.rl.datasets --make-train --make-eval --print-summary
python -m src.rl.train_grpo --config configs/train.yaml --dry-run
```

GPU-only full steps (executed automatically by the script only if supported):

```bash
python -m src.rl.train_grpo --config configs/train.yaml
python -m src.rl.evaluate --model baseline
python -m src.rl.evaluate --model trained
python -m src.rl.plot_results
```

## Upload to Hugging Face

```bash
python -m src.rl.upload_hf \
  --model-repo-id <username/dz2-mod-arith-model> \
  --dataset-repo-id <username/dz2-mod-arith-eval>
```

Update links after upload:

- Model (trained): `<HF_MODEL_URL>`
- Eval datasets: `<HF_DATASET_URL>`

## Notes

- Verifier compares normalized prediction: `predicted % M`.
- `extract_answer` first parses `<answer>...</answer>`, then falls back to last integer.
- Fixed seeds for reproducibility are configured in `configs/data.yaml`.
