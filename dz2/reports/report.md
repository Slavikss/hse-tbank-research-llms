# Отчет по ДЗ 2: RL-агент для one-step среды

## 1. Постановка задачи

Реализована среда `Modular Arithmetic Expressions`, где модель за один запрос вычисляет значение арифметического выражения по модулю `M`.

## 2. Дизайн среды

- Класс среды: `ArithmeticModEnv` (`src/envs/arithmetic_mod/env.py`)
- Класс верификатора: `ArithmeticModVerifier` (`src/envs/arithmetic_mod/verifier.py`)
- Формат ответа: `<answer>...</answer>`
- Проверка: нормализация предсказания через `% M` и сравнение с `data.answer`

### 2.1 Управление сложностью

Использован диапазон `difficulty` от 1 до 10 с ростом:

- количества термов,
- диапазона чисел,
- глубины скобок,
- модуля.

Поддержаны прямые overrides через kwargs: `n_terms`, `abs_max`, `operators`, `max_parentheses_depth`, `modulus`, `allow_negative_literals`, `seed`.

## 3. Данные

### 3.1 Train

- Размер: 24,000
- Баланс по сложностям: равномерный по `1..10` (по 2,400 на уровень)
- Seed: 2026
- Файл: `data/train/train.jsonl`

### 3.2 Eval

- Сложности: `[2, 4, 6, 8, 10]`
- Размер на сложность: 300
- Seed: `9000 + difficulty`
- Файлы: `data/eval/difficulty_{2,4,6,8,10}.jsonl`
- Датасеты фиксированные и воспроизводимые.

## 4. Обучение (GRPO)

- База: `Qwen/Qwen2.5-1.5B-Instruct`
- Инструменты: Unsloth + GRPO (`src/rl/train_grpo.py`)
- Reward: `correctness_reward_func` как обертка над `Env.verify` (`src/rl/reward.py`)
- `SYSTEM_PROMPT` в формате из PDF:
  - `<think>...</think><answer>...</answer>`
  - дополнительная инструкция для стабильности: `Stop immediately after </answer>`

Ключевые гиперпараметры (актуальные):

- `max_steps=800`
- `learning_rate=5e-6`
- `per_device_train_batch_size=2`
- `gradient_accumulation_steps=8`
- `num_generations=4`
- `max_prompt_length=416`
- `max_completion_length=96`
- `mask_truncated_completions=true`
- `temperature=0.2`
- `top_p=0.9`
- `generation_kwargs.stop_strings=["</answer>"]`

## 5. Важное отклонение от формулировки PDF

В тексте задания упомянут iterable dataset для GRPO. В `trl==0.24.0` `GRPOTrainer` не поддерживает `IterableDataset` и выбрасывает `NotImplementedError`. Поэтому используется map-style `Dataset.from_list`.

Это зафиксированное техническое ограничение стека, не влияющее на семантику задачи.

## 6. Текущее состояние проверки

Подтверждено локально:

- `ruff check .` — проходит
- `ruff format --check .` — проходит
- `pytest -q` — проходит (актуальный набор тестов, включая strict prompt/reward contract/length guard)
- Генерация train/eval датасетов проходит полностью
- Dry-run тренировки проходит

## 7. Результаты baseline vs trained

Артефакты полного GPU-прогона (`results/metrics/*`, `results/figures/*`) в текущем состоянии репозитория не зафиксированы.

После запуска на CUDA GPU необходимо:

1. Запустить `python -m src.rl.train_grpo --config configs/train.yaml`
2. Запустить `python -m src.rl.evaluate --model baseline`
3. Запустить `python -m src.rl.evaluate --model trained`
4. Запустить `python -m src.rl.plot_results`
5. Обновить этот отчет фактическими метриками и анализом ошибок (`parse/arithmetic/modulo_sign`)

## 8. Ограничения и стабильность

- Полный цикл обучения/оценки требует GPU-устройство, поддерживаемое Unsloth/vLLM.
- Для текущего стека добавлен временный shape-guard в `src/rl/train_grpo.py`, который выравнивает длины completion тензоров при редком mismatch в Unsloth/TRL.

## 9. Воспроизводимость

- One-command запуск: `bash scripts/run_local_pipeline.sh`
- Конфиги: `configs/data.yaml`, `configs/train.yaml`
- Зависимости: `requirements.txt`
