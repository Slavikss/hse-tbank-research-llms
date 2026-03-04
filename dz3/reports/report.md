# DZ3 Report: SRFT on Hard Subsets

## 1. Experimental Setup

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Generation params (fixed across all comparisons):
  - `temperature=0.7`
  - `top_p=0.95`
  - `max_tokens=96`
- Splits:
  - `train`
  - `val (full)`
  - `Hard-train` (`pass@128=0` for baseline)
  - `Hard-val` (`pass@128=0` for baseline)

## 2. Baseline Hard Subset Verification

Fill from `results/passk/hard_mining_report.json` and baseline pass@k files.

- Hard-train size:
- Hard-val size:
- Baseline pass@128 on Hard-train:
- Baseline pass@128 on Hard-val:

## 3. Main Quality Comparison

Compare these models:

- Baseline
- GRPO-only
- SFT-only
- SFT->GRPO
- SRFT

Use:

- `results/figures/passk_val.png`
- `results/figures/passk_hard_val.png`

## 4. Training Dynamics

Use:

- `results/figures/reward_curve.png`
- `results/figures/gen_len_curve.png`
- `results/figures/entropy_curve.png`

## 5. Did We Break the Zero?

State clearly:

- `Hard-val pass@128`: baseline `0` -> trained `> 0` (or not)

## 6. Length and Diversity Trade-off

- Mean/median generation lengths on `val` and `Hard-val`
- Did full-val pass@k drop after RL?
- If yes, discuss magnitude and likely reasons

## 7. Notes and Limitations

- Runtime/memory constraints
- Any deviations from planned protocol
