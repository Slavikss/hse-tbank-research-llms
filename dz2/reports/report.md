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
- Reward: `correctness_reward_func` через `Verifier.verify` (`src/rl/reward.py`)

Ключевые гиперпараметры:

- `max_steps=800`
- `learning_rate=1e-5`
- `per_device_train_batch_size=2`
- `gradient_accumulation_steps=8`
- `num_generations=4`
- `max_prompt_length=384`
- `max_completion_length=96`

## 5. Текущее состояние результатов

### 5.1 Что уже подтверждено локально

- Все юнит/смоук тесты проходят (`12/12`), включая:
  - генерацию,
  - parsing `<answer>` и fallback,
  - verify,
  - воспроизводимость eval,
  - dry-run тренировки.
- Линт и формат проходят: `ruff check .`, `ruff format --check .`.
- Генерация train/eval датасетов проходит полностью.
- One-command локальный pipeline (`bash scripts/run_local_pipeline.sh`) исполняется успешно.

### 5.2 Что ещё не посчитано

Таблица baseline vs trained accuracy и итоговые метрики (`macro accuracy`, `parse success`, распределение ошибок) пока не заполнены, потому что для этого шага нужен локальный GPU-рантайм, который поддерживается Unsloth/vLLM.

## 6. Анализ поведения (план интерпретации)

После запуска baseline/trained в одном и том же evaluation pipeline нужно проверить:

- как меняется gain от RL по мере роста сложности;
- растет ли доля ошибок `arithmetic` на больших уровнях;
- уменьшается ли `parse`-ошибка после обучения на заданном формате ответа;
- есть ли рост ошибок `modulo_sign` на сложных выражениях.

## 7. Ограничения и дальнейшие шаги

- Полный цикл обучения/оценки требует GPU-устройство, поддерживаемое Unsloth.
- На текущей машине (macOS arm64, без CUDA GPU) запускается только локальный CPU-контур.
- Для финального submission осталось:
  1. Запустить `python -m src.rl.train_grpo --config configs/train.yaml` на поддерживаемом GPU.
  2. Запустить `python -m src.rl.evaluate --model baseline` и `python -m src.rl.evaluate --model trained`.
  3. Построить график `python -m src.rl.plot_results`.
  4. Обновить отчет фактическими метриками и интерпретацией.
  5. Загрузить модель и eval datasets на Hugging Face.

## 8. Воспроизводимость

- Локальный one-command запуск: `bash scripts/run_local_pipeline.sh`.
- Локальные зависимости: `requirements.txt`.
