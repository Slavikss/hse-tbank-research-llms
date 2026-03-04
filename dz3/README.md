# DZ3: Learning from "Unsolvable" Tasks (SRFT + RL + Supervised Signal)

Standalone Homework 3 repository based on the modular arithmetic environment from DZ2.

Goal: construct hard subsets where baseline has `pass@128 = 0`, then improve to `pass@128 > 0`
using:

- Baseline
- GRPO-only
- SFT-only
- SFT->GRPO
- SRFT (single-stage supervised + RL objective)

## Structure

- `src/base/`: `Data`, `Env`, `Verifier`
- `src/envs/arithmetic_mod/`: modular arithmetic task environment
- `src/rl/`: dataset generation, hard mining, gold trajectories, training, pass@k evaluation, plotting
- `configs/`: `data.yaml`, `hard.yaml`, `train.yaml`
- `scripts/run_dz3_pipeline.sh`: reproducible pipeline
- `tests/`: unit + smoke tests
- `reports/report.md`: experiment report template

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick local check

```bash
ruff check .
ruff format --check .
pytest -q
```

## Data + hard subset + gold trajectories

```bash
python -m src.rl.datasets --make-train --make-val --print-summary
python -m src.rl.gold --input data/train/train.jsonl --output data/train/train.jsonl
python -m src.rl.hard_mining --config configs/hard.yaml
python -m src.rl.gold --input data/hard/hard_train.jsonl --output data/hard/hard_train.jsonl
```

## Training

```bash
python -m src.rl.train_grpo --config configs/train.yaml
python -m src.rl.train_sft --config configs/train.yaml
python -m src.rl.train_sft_grpo --config configs/train.yaml
python -m src.rl.train_srft --config configs/train.yaml
```

Dry-run mode (CPU-safe):

```bash
python -m src.rl.train_grpo --config configs/train.yaml --dry-run
python -m src.rl.train_sft --config configs/train.yaml --dry-run
python -m src.rl.train_sft_grpo --config configs/train.yaml --dry-run
python -m src.rl.train_srft --config configs/train.yaml --dry-run
```

## Evaluation (pass@k)

```bash
python -m src.rl.passk_eval --config configs/train.yaml --model baseline --split val
python -m src.rl.passk_eval --config configs/train.yaml --model baseline --split hard_val
python -m src.rl.evaluate --config configs/train.yaml --model grpo --split both
python -m src.rl.evaluate --config configs/train.yaml --model sft --split both
python -m src.rl.evaluate --config configs/train.yaml --model sft_grpo --split both
python -m src.rl.evaluate --config configs/train.yaml --model srft --split both
```

## Plotting

```bash
python -m src.rl.plot_passk
python -m src.rl.plot_training
```

## Full pipeline

```bash
bash scripts/run_dz3_pipeline.sh
```

The script always runs checks/tests/data generation and dry-runs.
If CUDA is available, it also runs full hard-mining, training, evaluation, and plotting.
