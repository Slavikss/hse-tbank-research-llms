#!/usr/bin/env bash
set -euo pipefail

# Local one-command pipeline (works on CPU-only machines).
# Steps:
# 1) install local deps
# 2) static checks + tests
# 3) generate datasets
# 4) run GRPO dry-run
# 5) optionally run full train/eval/plot if local GPU runtime supports Unsloth + vLLM

python3 -m pip install -r requirements.txt

ruff check .
ruff format --check .
python3 -m pytest -q

python3 -m src.rl.datasets --make-train --make-eval --print-summary
python3 -m src.rl.train_grpo --config configs/train.yaml --dry-run

if python3 - <<'PY'
import importlib.util
import sys

try:
    import torch
except Exception:
    sys.exit(1)

if not torch.cuda.is_available():
    sys.exit(1)

if importlib.util.find_spec("unsloth") is None:
    sys.exit(1)
if importlib.util.find_spec("vllm") is None:
    sys.exit(1)

try:
    import unsloth  # noqa: F401
except Exception:
    sys.exit(1)

sys.exit(0)
PY
then
  echo "[local-pipeline] GPU runtime detected. Running full train/eval/plot..."
  python3 -m src.rl.train_grpo --config configs/train.yaml
  python3 -m src.rl.evaluate --model baseline
  python3 -m src.rl.evaluate --model trained
  python3 -m src.rl.plot_results
  echo "[local-pipeline] Full pipeline completed."
else
  echo "[local-pipeline] GPU runtime for Unsloth/vLLM not available."
  echo "[local-pipeline] Completed local pipeline: checks + tests + datasets + dry-run training."
fi
