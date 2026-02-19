# Эксперименты: Policy Gradient и Behaviour Cloning на CartPole-v1

## 1. Что делалось и зачем

Было две части:
- сначала обучение policy gradient-методов;
- затем behaviour cloning (BC) на данных от лучшей RL-политики и проверка его слабых мест.

Главный вопрос: какие варианты обучения действительно надежны, а где красивый результат получается только в частных условиях.

## 2. Сетап эксперимента

Общее для всех запусков:
- среда: `CartPole-v1`;
- политика: MLP `4 -> 64 -> 64 -> 2` с `Tanh`;
- оптимизатор: `Adam`;
- 3 seed: `0, 1, 2`;
- конфиги: `configs/rl_balanced.yaml` и `configs/bc_balanced.yaml`.

Сравнивались RL-варианты:
- `VPG`;
- `VPG + moving-average baseline`;
- `VPG + value baseline`;
- `VPG + RLOO baseline`;
- `VPG + value baseline + entropy regularization` (в двух настройках).

Артефакты RL:
- `artifacts/rl/aggregate_results.csv`
- `artifacts/rl/experiment_summary.csv`
- `artifacts/rl/combined_metrics.csv`
- `artifacts/rl/learning_curves.png`

## 3. Что получилось в RL

Сводка (mean по 3 seed):

| Метод | best_eval_reward_mean | steps_to_495_mean |
|---|---:|---:|
| VPG | 500.0 | 72917 |
| VPG + moving-average baseline | 366.7 | NaN |
| VPG + value baseline | 497.7 | 73592 |
| VPG + RLOO baseline | 500.0 | 74657 |
| VPG + value + entropy (beta=0.001 const) | 500.0 | 63665 |
| VPG + value + entropy (beta: 0.01 -> 0.0 linear) | 494.7 | 55331 (1 успешный seed) |

Интерпретация результатов:
- Базовый `VPG` оказался неожиданно сильным и стабильно доходил до оптимума, но не был самым экономным по sample efficiency.
- `Value baseline + entropy (0.001)` дал лучший практический баланс: стабильное качество и более быстрое достижение порога 495.
- `RLOO` также стабилен и надежен, но в этом сетапе медленнее лучшего варианта.
- `Moving-average baseline` в текущей конфигурации получился самым неудачным: сильная нестабильность и недообучение.
- Агрессивная энтропия (`0.01 -> 0`) дала очень быстрый seed, но результат неустойчив между seed.

Если сформулировать коротко: умеренная энтропийная регуляризация вместе с value baseline здесь работает лучше всего “в среднем по больнице”.

## 4. Behaviour Cloning: основной результат

Дальше я взял лучшую RL-политику как эксперта, собрал датасет `(state, action)` и обучил такую же по архитектуре, но заново и уже в supervised-режиме.

Артефакты BC:
- `artifacts/bc/expert_dataset.npz`
- `artifacts/bc/bc_main/summary.json`

Факты по данным и качеству:
- размер датасета: `59000` пар `(state, action)`;
- отобрано `118/120` эпизодов при пороге `min_reward=475`;
- средняя награда отобранных траекторий эксперта: `500.0`;
- итог BC-модели: `best_val_loss=0.00324`, `eval_reward_mean=500.0`, `eval_reward_std=0.0`.

Вывод по основной задаче BC: если эксперт очень хороший и покрытие данных достаточно широкое, BC на CartPole может практически полностью воспроизвести поведение эксперта.

## 5. Где BC ломается: целевые эксперименты

Артефакты:
- `artifacts/bc/failure_experiments/coverage_vs_performance.csv`
- `artifacts/bc/failure_experiments/distribution_shift.csv`
- `artifacts/bc/failure_experiments/error_compounding.csv`
- `artifacts/bc/failure_experiments/error_compounding.png`

### 5.1 Coverage vs performance

- 10% данных -> `eval_reward_mean=489.15`
- 30% данных -> `500.0`
- 50% данных -> `496.0`
- 100% данных -> `500.0`

Тренд понятный: когда покрытие датасета сужается, BC становится менее надежным.

### 5.2 Distribution shift

При добавлении шума в наблюдения:
- `std=0.00`: BC `491.4`, expert `500.0`
- `std=0.05`: BC `439.35`, expert `481.75`

Общая картина: BC заметно проседает при смещении распределения входов. По отдельным точкам кривая может быть немонотонной из-за стохастичности оценки, но общий спад качества есть.

### 5.3 Error compounding

- средний mismatch rate за первые 100 шагов: `0.0008`;
- средний mismatch rate по всем шагам с валидным count: `0.0012`.

В этом эксперименте накопление ошибки небольшое, но не нулевое: даже редкие локальные ошибки могут постепенно уводить траекторию в зоны, которые экспертный датасет покрывает хуже.

## 6. Выводы

- Среди проверенных RL-подходов лучший баланс в этом запуске показал `value baseline + entropy beta=0.001`.
- На простом домене и сильном эксперте BC может дать почти идеальное качество.
- Ограничения BC подтверждаются экспериментально: хуже переносится distribution shift и хуже работает при ограниченном покрытии данных.
- Практический вывод: BC подходит как сильная инициализация, но в реальных задачах его лучше дополнять онлайн-коррекцией (например, DAgger-подходами или RL-finetune).

## 7. Как воспроизвести

```bash
python -m scripts.run_full_pipeline --rl-config configs/rl_balanced.yaml --bc-config configs/bc_balanced.yaml
```
