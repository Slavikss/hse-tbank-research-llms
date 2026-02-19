# Эксперименты: Policy Gradient и Behaviour Cloning на CartPole-v1

## 1. Постановка задачи

Цель работы — аккуратно сравнить несколько вариантов policy gradient на `CartPole-v1`, а затем проверить, насколько хорошо `Behaviour Cloning` (BC) воспроизводит эксперта и где у этого подхода проявляются слабые места.

Эксперименты разделены на две части:
- RL: обучение политик разными loss-функциями и baseline-стратегиями;
- BC: обучение на экспертных траекториях и стресс-тесты на ограничения BC.

## 2. Экспериментальный сетап

Общее:
- Среда: `CartPole-v1`
- Политика: MLP `4 -> 64 -> 64 -> 2`, активация `Tanh`
- Seeds: `0, 1, 2`
- Конфиги: `configs/rl_balanced.yaml`, `configs/bc_balanced.yaml`

Сравниваемые RL-варианты:
- `VPG`
- `PG + cumulative average reward baseline`
- `PG + value baseline`
- `PG + trajectory-level RLOO baseline`
- `PG + value baseline + entropy regularization` (два режима `beta`)

## 3. RL: лоссы и реализация

Ниже формулы в plain-text (без LaTeX), как они реализованы в коде.

1. `Vanilla Policy Gradient`
$$
L_vpg = -E_t[ log pi_theta(a_t | s_t) * G_t ]
$$

`G_t` — discounted return.

2. `PG + cumulative average reward baseline`

$$
A_t = G_t - R_global_mean
L_pg = -E_t[ log pi_theta(a_t | s_t) * A_t ]
$$

`R_global_mean` — кумулятивная средняя награда по всем эпизодам, увиденным в обучении.

3. `PG + value baseline`

$$
A_t = G_t - V_phi(s_t)
L_policy = -E_t[ log pi_theta(a_t | s_t) * A_t ]
L_value = MSE( V_phi(s_t), G_t )
$$

4. `PG + trajectory-level RLOO`

Для траектории `i`:

$$
R_i = sum_t r_t^(i)
b_i^{\mathrm{LOO}} = (sum_{j != i} R_j) / (N - 1) \\
A_i = R_i - b_i^{\mathrm{LOO}} \\
L_rloo = -(1/N) * sum_i[ (sum_t log pi_theta(a_t^(i) | s_t^(i))) * A_i ]
$$

Entropy-часть для всех вариантов:

$$
L_total = L_pg - beta * Entropy(pi_theta(. | s))
$$

`beta` меняется по `constant` или `linear` schedule.

## 4. RL: результаты

Артефакты:
- `artifacts/rl/aggregate_results.csv`
- `artifacts/rl/experiment_summary.csv`
- `artifacts/rl/combined_metrics.csv`
- `artifacts/rl/learning_curves.png`

Сводка (mean по 3 seed):

| Метод | best_eval_reward_mean | steps_to_495_mean |
|---|---:|---:|
| VPG | 500.0 | 72917 |
| VPG + cumulative average baseline | 357.7 | NaN |
| VPG + value baseline | 497.7 | 73592 |
| VPG + trajectory-level RLOO | 412.9 | 86509 |
| VPG + value + entropy (`beta=0.001`, const) | 500.0 | 63665 |
| VPG + value + entropy (`beta: 0.01 -> 0`, linear) | 494.7 | 55331 (1 успешный seed) |

Ключевые наблюдения:
- В этом сетапе лучший баланс скорости/стабильности показал `value baseline + entropy (0.001 const)`.
- `VPG` оказался очень сильным и стабильно дошел до `500`, но уступает лучшему варианту по sample efficiency.
- `Cumulative average baseline` и `trajectory-level RLOO` в текущих гиперпараметрах оказались заметно менее стабильными.
- Проверка корректности baseline-ветки выполнена: `vpg` и `vpg_moving_avg` теперь различаются существенно (не на уровне `1e-6`).

## 5. Behaviour Cloning

Пайплайн:
1. Берем лучшую RL-политику как эксперта.
2. Генерируем датасет `(state, action)` из экспертных траекторий.
3. Обучаем новую policy-сеть в supervised-режиме.

Артефакты:
- `artifacts/bc/expert_dataset.npz`
- `artifacts/bc/bc_main/summary.json`
- `artifacts/bc/suite_summary.json`

Факты:
- Экспертный датасет: `59000` пар `(state, action)`
- Отобрано эпизодов: `118/120` (порог `min_reward=475`)
- BC main: `best_val_loss=0.00401`, `eval_reward_mean=500.0`, `eval_reward_std=0.0`

Вывод: при хорошем эксперте и достаточном покрытии BC может практически идеально воспроизвести политику на `CartPole`.

## 6. Ограничения BC: стресс-тесты

Артефакты:
- `artifacts/bc/failure_experiments/coverage_vs_performance.csv`
- `artifacts/bc/failure_experiments/distribution_shift.csv`
- `artifacts/bc/failure_experiments/error_compounding.csv`
- `artifacts/bc/failure_experiments/error_compounding.png`

### 6.1 Coverage vs performance

- 10% данных -> `eval_reward_mean=490.8`
- 30% данных -> `500.0`
- 50% данных -> `494.3`
- 100% данных -> `500.0`

Недостаточное покрытие ухудшает надежность BC.

### 6.2 Distribution shift

- `std=0.00`: BC `491.4`, expert `500.0`
- `std=0.05`: BC `464.8`, expert `488.1`

При сдвиге распределения наблюдений качество BC заметно падает.

### 6.3 Error compounding

- Средний mismatch rate за первые 100 шагов: `0.0004`
- Средний mismatch rate по всем шагам с валидным count: `0.00104`

Накопление ошибки небольшое, но не нулевое.

## 7. Итог

- Все требуемые в ДЗ режимы RL/BC реализованы и прогнаны на `CartPole-v1`.
- В текущем запуске наиболее практичным вариантом RL оказался `value baseline + умеренная entropy regularization`.
- BC дает отличное качество внутри распределения экспертных данных, но ожидаемо уязвим к coverage gaps и distribution shift.

## 8. Воспроизведение

```bash
python -m scripts.run_full_pipeline --rl-config configs/rl_balanced.yaml --bc-config configs/bc_balanced.yaml
```
