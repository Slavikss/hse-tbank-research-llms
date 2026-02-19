# ДЗ1: RL на CartPole + Behaviour Cloning

Проект реализует:
- Vanilla Policy Gradient (VPG)
- VPG с baseline:
  - moving-average reward baseline
  - value-function baseline
  - RLOO-style baseline
- entropy regularization (включая linear decay scheduler)
- behaviour cloning (BC) от лучшей RL-политики
- эксперименты, показывающие ограничения BC

## Установка

```bash
uv venv
source .venv/bin/activate
uv pip install -e '.[dev]'
```

## Быстрый smoke-прогон

```bash
python -m scripts.run_rl_suite --config configs/rl_smoke.yaml
python -m scripts.run_bc_suite --config configs/bc_smoke.yaml --expert artifacts/rl/best_expert.pt
python -m scripts.make_report_artifacts --input artifacts --output report/assets
```

## Полный balanced-прогон

```bash
python -m scripts.run_rl_suite --config configs/rl_balanced.yaml
python -m scripts.run_bc_suite --config configs/bc_balanced.yaml --expert artifacts/rl/best_expert.pt
python -m scripts.make_report_artifacts --input artifacts --output report/assets
```

## Единая команда воспроизведения

```bash
python -m scripts.run_full_pipeline --rl-config configs/rl_balanced.yaml --bc-config configs/bc_balanced.yaml
```

## Структура

- `src/rl`: обучение policy gradient
- `src/bc`: датасет эксперта, обучение BC, эксперименты на недостатки BC
- `src/utils`: логирование и построение графиков
- `scripts`: CLI для полного пайплайна
- `tests`: unit/smoke tests
- `report/report.md`: секция экспериментов

## Ссылки

- Gymnasium CartPole: <https://gymnasium.farama.org/environments/classic_control/cart_pole/>
- CleanRL: <https://github.com/vwxyzjn/cleanrl>
- HuggingFace Deep RL Course: <https://huggingface.co/learn/deep-rl-course/en/unit0/introduction>
