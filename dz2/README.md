# DZ2: RL Agent for One-Step Modular Arithmetic Environment

This repository implements Homework 2 for the course "Selected Topics in AI Research".
Task: one-step environment where an LLM computes an arithmetic expression modulo `M`.

## Implemented Components

- `Env` + `Verifier` for modular arithmetic expressions.
- Difficulty mapping (`1..10`) and direct hyper-parameter overrides.
- Train dataset sampling + fixed reproducible eval datasets.
- GRPO training script for `Qwen2.5-1.5B-Instruct` with Unsloth.
- vLLM inference + evaluation metrics + grouped bar plotting.
- Unit tests for generation, parsing, verification, reproducibility, reward contract, and train dry-run.

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

```bash
bash scripts/run_local_pipeline.sh
```

This command performs:

- lint and format checks (`ruff`)
- tests (`pytest`)
- train/eval dataset generation
- GRPO dry-run

If local hardware supports Unsloth + vLLM (GPU runtime), the script automatically continues with full training, evaluation, and plotting.

## Manual Commands

```bash
python -m src.rl.datasets --make-train --make-eval --print-summary
python -m src.rl.train_grpo --config configs/train.yaml --dry-run
```

GPU-only full steps:

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

## Training Configuration (strict PDF prompt)

`configs/train.yaml` defaults:

- `learning_rate: 5.0e-6`
- `max_prompt_length: 448`
- `max_completion_length: 64`
- `mask_truncated_completions: true`

`SYSTEM_PROMPT` follows PDF format exactly:

```text
Respond in the following format:
<think>
...
</think>
<answer>
...
</answer>
```

## Important Notes

- Reward is implemented as a wrapper over `Env.verify` (`src/rl/reward.py`).
- `extract_answer` first parses `<answer>...</answer>`, then falls back to the last integer.
- Verifier compares normalized prediction: `predicted % M`.
- Fixed seeds for reproducibility are configured in `configs/data.yaml`.

### IterableDataset Deviation (documented)

The PDF mentions iterable datasets for GRPO. With the pinned stack (`trl==0.24.0`), `GRPOTrainer` raises `NotImplementedError` for `IterableDataset`. Because of this, training intentionally uses `Dataset.from_list` (map-style) in `src/rl/train_grpo.py`.

### Unsloth/TRL Shape Mismatch Guard

A temporary guard truncates mismatched completion tensors in `src/rl/train_grpo.py` to keep local training stable with the current Unsloth/TRL combination. This is a compatibility workaround, not a change in task semantics.
