# ДЗ-5: RL для function calling
## Запуск

Разберем основные команды для прогона домашки. Они выполняются из папки `5` в окружении с установленными зависимостями из `requirements.txt`.

Настройка GPU и подготовка данных:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=7
python prepare_data.py
```

Бейзлайн и четыре независимых RL-эксперимента выполняются последовательно:

```bash
python evaluate.py Qwen/Qwen3.5-2B --output results/base

python train.py --config configs/grpo_exact.json
python export_model.py grpo_exact
python evaluate.py checkpoints/grpo_exact/merged --output results/grpo_exact

python train.py --config configs/grpo_shaped.json
python export_model.py grpo_shaped
python evaluate.py checkpoints/grpo_shaped/merged --output results/grpo_shaped

python train.py --config configs/dr_grpo_shaped.json
python export_model.py dr_grpo_shaped
python evaluate.py checkpoints/dr_grpo_shaped/merged --output results/dr_grpo_shaped

python train.py --config configs/grpo_token_is.json
python export_model.py grpo_token_is
python evaluate.py checkpoints/grpo_token_is/merged --output results/grpo_token_is
```

Таблицы, графики, разбор ошибок и перевод отчёта Markdown → PDF:

```bash
python analyze_results.py
python render_report.py report.md report.pdf
```

## Лучший результат

**Дополнительный запуск:** `grpo_token_is` завершён, включён в отчёт; BFCL overall — 70.00%.

Лучший полный чекпоинт находится локально в `checkpoints/grpo_token_is/merged` (выбран шаг 150). Его метрики и ответы — в `results/grpo_token_is/`. 

## Файлы

| Путь | Назначение |
|---|---|
| `prepare_data.py`, `common.py` | Загрузка, разбор, split, фильтрация, EDA, сравнение вызовов |
| `rewards.py` | Разбор с Qwen3CoderToolParser и награда |
| `requirements.txt` | Версии зависимостей |
| `train.py`, `configs/` | Четыре независимых RL-запуска |
| `export_model.py` | Merge LoRA с исходной ревизией; полный HF-чекпоинт |
| `analyze_results.py` | Проверка полноты результатов, таблицы, графики, ошибки, обновление MD |
| `render_report.py` | Перевод уже подготовленного MD в PDF |
| `artifacts/eda.json`, `artifacts/figures/` | Реальные EDA и графики |
| `artifacts/<run>/run.json`, `logs.jsonl`, `trainer_state.json`, `export.json` | Настройки, версии, выбранный шаг, кривые, хеши исходных файлов токенизатора |
| `results/base/`, `results/<run>/` | `metrics.json`, `predictions.jsonl`, JSON-копия ответов |
| `data/`, `checkpoints/`, `.cache/` | Локальные данные, веса и кэш; не для Git |
