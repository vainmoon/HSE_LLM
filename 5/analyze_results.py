import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import RUNS, file_hash, read_jsonl, write_json

GROUPS = {"simple": ["simple"], "multiple": ["multiple", "live_multiple"],
          "parallel": ["parallel"], "parallel_multiple": ["parallel_multiple"],
          "irrelevance": ["irrelevance", "live_irrelevance"]}
LABELS = ["Baseline", "GRPO exact", "GRPO shaped", "Dr. GRPO shaped"]


def overall(rows):
    return mean(mean(mean(float(r["check"]["valid"]) for r in rows if r["category"] == category)
                     for category in categories) for categories in GROUPS.values())


def replace_section(text, name, content):
    start, end = f"<!-- {name}:start -->", f"<!-- {name}:end -->"
    left, rest = text.split(start, 1)
    _, right = rest.split(end, 1)
    return left + start + "\n\n" + content.strip() + "\n\n" + end + right


def describe_data():
    eda = json.loads(Path("artifacts/eda.json").read_text())
    data_text = (
        f"В исходной версии ToolACE {eda['raw_dialogues']} диалогов; "
        f"после удаления точных дубликатов — {eda['unique_dialogues']}. "
        f"В разобранных JSON-описаниях найдено {eda['unique_tools']} уникальных имён инструментов.\n\n"
        "| Сплит | Примеров | Диалогов | Без вызова | Один вызов | Несколько вызовов |\n"
        "|---|---:|---:|---:|---:|---:|\n")
    for split, info in eda["splits"].items():
        counts = info["kinds"]
        data_text += (f"| {split} | {info['examples']} | {info['dialogues']} | "
                      f"{counts.get('no_call', 0)} | {counts.get('single', 0)} | {counts.get('parallel', 0)} |\n")
    data_text += "\nОтфильтрованные случаи: " + "; ".join(f"`{k}` — {v}" for k, v in eda["skipped"].items()) + "."
    data_text += "\n\n| Длина, токены | Медиана | p90 | p99 | Максимум |\n|---|---:|---:|---:|---:|\n"
    for key, label in [("prompt", "Промпт"), ("answer", "Эталон")]:
        q = eda["length_quantiles_before_filter"][key]
        data_text += f"| {label} | {q['p50']:.0f} | {q['p90']:.0f} | {q['p99']:.0f} | {q['p100']:.0f} |\n"
    data_text += "\nКвантили вычислены до фильтрации по длине, после разбора и проверки схем."
    data_text += "\n\n![Распределения до фильтрации по длине](artifacts/figures/eda.png)"
    return data_text


def describe_reward():
    from transformers import AutoTokenizer
    from rewards import ToolReward
    from common import BASE_MODEL, BASE_REVISION
    reward = ToolReward(AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION))
    data_examples = json.loads(Path("artifacts/data_examples.json").read_text())
    sample = data_examples["single"] or data_examples["parallel"]
    tools, gold = json.loads(sample["tools_json"]), json.loads(sample["gold_json"])
    def xml(calls):
        parts = []
        for call in calls:
            params = "".join(f"<parameter={key}>\n{value if isinstance(value, str) else json.dumps(value)}\n</parameter>\n"
                             for key, value in call["arguments"].items())
            parts.append(f"<tool_call>\n<function={call['name']}>\n{params}</function>\n</tool_call>")
        return "\n".join(parts)
    variants = [("Правильный ответ", xml(gold), False),
                ("Пропущены аргументы", xml([{**call, "arguments": {}} for call in gold]), False),
                ("Пропущен вызов", "Please clarify your request.", False),
                ("Лишний повторный вызов", xml(gold + gold[:1]), False),
                ("Обрезанный ответ", xml(gold), True)]
    reward_rows = [{"case": case, "response": text, "truncated": cut,
                    **reward.score(text, tools, gold, cut)} for case, text, cut in variants]
    no_call = data_examples.get("no_call")
    if no_call:
        reward_rows.append({"case": "Ответ без инструмента на no-call запрос", "response": no_call["reference_text"],
                            **reward.score(no_call["reference_text"], json.loads(no_call["tools_json"]), [])})
    write_json("artifacts/reward_examples.json", {"source_id": sample["id"], "gold": gold, "cases": reward_rows})
    reward_table = ("| Пример | Exact | F1 имён | F1 аргументов | Shaped |\n|---|---:|---:|---:|---:|\n" +
                    "\n".join(f"| {r['case']} | {r['exact']:.2f} | {r['names']:.2f} | {r['arguments']:.2f} | {r['shaped']:.2f} |"
                               for r in reward_rows))
    reward_table = "Эталон положительного примера:\n```json\n" + json.dumps(gold, ensure_ascii=False) + "\n```\n\n" + reward_table
    reward_table += f"\n\nПоложительный пример взят из train ToolACE, id `{sample['id']}`. "
    reward_table += "Варианты с ошибками созданы только для проверки награды; они не добавляются в обучение."

    return reward_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    if args.data_only:
        report = Path("report.md").read_text()
        report = replace_section(report, "eda", describe_data())
        Path("report.md").write_text(report)
        print("В report.md обновлён EDA; результаты моделей не изменены.")
        return
    if Path("results/grpo_token_is/metrics.json").exists():
        RUNS.append("grpo_token_is")
        LABELS.append("GRPO token IS")
    benchmark = {r["id"]: r for r in read_jsonl("bfcl.jsonl")}
    metrics, predictions, manifests, decisions = {}, {}, {}, {}
    for run in ["base"] + RUNS:
        folder = Path("results") / run
        metrics[run] = json.loads((folder / "metrics.json").read_text())
        predictions[run] = read_jsonl(folder / "predictions.jsonl")
        decisions[run] = {
            "text_instead_of_call": sum(r["check"].get("error") == "Ожидался вызов инструмента."
                                        for r in predictions[run]),
            "call_in_irrelevance": sum("irrelevance" in r["category"] and bool(r["calls"])
                                       for r in predictions[run]),
        }
        ids = [r["id"] for r in predictions[run]]
        if len(ids) != len(set(ids)) or set(ids) != set(benchmark):
            raise ValueError(f"{run}: набор ответов отличается от BFCL")
        if metrics[run]["benchmark_sha256"] != file_hash("bfcl.jsonl"):
            raise ValueError(f"{run}: использован другой benchmark")
        if abs(overall(predictions[run]) - metrics[run]["overall"]) > 1e-10:
            raise ValueError(f"{run}: метрика не совпадает с predictions")
        # В PDF задания требуется JSONL, в тексте объявления упомянут JSON.
        write_json(folder / "predictions.json", predictions[run])
        if run != "base":
            manifests[run] = json.loads((Path("artifacts") / run / "run.json").read_text())
            if manifests[run]["status"] != "complete" or manifests[run]["smoke"]:
                raise ValueError(f"{run}: в таблице допустим только законченный основной запуск")
    baseline = metrics["base"]
    for run in RUNS:
        if metrics[run]["decoding"] != baseline["decoding"] or metrics[run]["versions"] != baseline["versions"]:
            raise ValueError(f"{run}: различаются параметры генерации или версии evaluator")
    if len({m["train_sha256"] for m in manifests.values()}) != 1:
        raise ValueError("Для контролируемого сравнения нужны одинаковые данные")

    table = ["| Запуск | Шаг | Overall | Simple | Multiple | Parallel | Parallel multiple | Irrelevance |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    csv_rows = []
    for label, run in zip(LABELS, ["base"] + RUNS):
        step = manifests[run]["selected_step"] if run != "base" else 0
        values = [metrics[run]["overall"]] + [metrics[run]["groups"][g] for g in GROUPS]
        table.append(f"| {label} | {step} | " + " | ".join(f"{v:.2%}" for v in values) + " |")
        csv_rows.append({"run": run, "step": step, "overall": values[0],
                         **metrics[run]["groups"], **decisions[run]})
    with Path("artifacts/summary.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_rows[0])
        writer.writeheader()
        writer.writerows(csv_rows)
    Path("artifacts/figures").mkdir(parents=True, exist_ok=True)
    training_summary = {}
    fig, axes = plt.subplots(len(RUNS), 3, figsize=(10, 3 * len(RUNS)))
    for row, run in enumerate(RUNS):
        logs = read_jsonl(Path("artifacts") / run / "logs.jsonl")
        selected_logs = [r for r in logs if "loss" in r and r["step"] <= manifests[run]["selected_step"]]
        training_summary[run] = {
            "first10_exact": mean(r["rewards/exact_match/mean"] for r in selected_logs[:10]),
            "selected_last10_exact": mean(r["rewards/exact_match/mean"] for r in selected_logs[-10:]),
            "first10_reward": mean(r["reward"] for r in selected_logs[:10]),
            "selected_last10_reward": mean(r["reward"] for r in selected_logs[-10:]),
            "zero_std_fraction": mean(r["frac_reward_zero_std"] for r in selected_logs),
            "full_run_mean_importance_correction": mean(r["sampling/importance_sampling_ratio/mean"]
                                                        for r in logs if "loss" in r),
        }
        for col, keys in enumerate([["loss"], ["reward", "eval_reward"],
                                    ["rewards/exact_match/mean", "eval_rewards/exact_match/mean"]]):
            for key in keys:
                points = [x for x in logs if key in x]
                if points:
                    steps, values = [x["step"] for x in points], [x[key] for x in points]
                    label = key.replace("rewards/exact_match/mean", "exact")
                    if not key.startswith("eval_") and len(values) >= 10:
                        axes[row, col].plot(steps, values, color="C0", alpha=0.2, linewidth=0.7)
                        axes[row, col].plot(steps[9:], np.convolve(values, np.ones(10) / 10, mode="valid"),
                                            color="C0", label=label + " (10 steps)")
                    else:
                        axes[row, col].plot(steps, values, "o-", color="C1", label=label)
            axes[row, col].axvline(manifests[run]["selected_step"], color="gray", ls="--", alpha=0.5)
            axes[row, col].set(title=f"{LABELS[row + 1]}: {['loss', 'reward', 'exact'][col]}", xlabel="Шаг")
            axes[row, col].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig("artifacts/figures/training.png", dpi=170)
    plt.close(fig)
    write_json("artifacts/training_summary.json", training_summary)
    fig, ax = plt.subplots(figsize=(10, 3))
    x = np.arange(len(GROUPS))
    width = 0.8 / len(LABELS)
    for i, (label, run) in enumerate(zip(LABELS, ["base"] + RUNS)):
        ax.bar(x + (i - (len(LABELS) - 1) / 2) * width,
               [metrics[run]["groups"][g] for g in GROUPS], width=width, label=label)
    ax.set(xticks=x, xticklabels=list(GROUPS), ylabel="Accuracy", ylim=(0, 1))
    ax.legend(fontsize=9, ncol=len(LABELS), loc="lower center", bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout()
    fig.savefig("artifacts/figures/bfcl.png", dpi=170)
    plt.close(fig)

    data_text = describe_data()

    # Парный bootstrap: одинаковые индексы для обеих моделей, отдельно по категориям.
    rng = np.random.default_rng(42)
    comparisons = []
    pairs = [("base", run) for run in RUNS] + [(RUNS[0], RUNS[1]), (RUNS[1], RUNS[2])]
    if "grpo_token_is" in RUNS:
        pairs.append(("grpo_shaped", "grpo_token_is"))
    for old, new in pairs:
        old_by_id = {r["id"]: r for r in predictions[old]}
        draws = {}
        for categories in GROUPS.values():
            for category in categories:
                delta = np.array([int(r["check"]["valid"]) - int(old_by_id[r["id"]]["check"]["valid"])
                                  for r in predictions[new] if r["category"] == category])
                draws[category] = rng.choice(delta, size=(2000, len(delta)), replace=True).mean(axis=1)
        samples = np.mean([np.mean([draws[c] for c in cats], axis=0) for cats in GROUPS.values()], axis=0)
        low, high = np.quantile(samples, [0.025, 0.975]) * 100
        comparisons.append({"from": old, "to": new, "delta_pp": 100 * (metrics[new]["overall"] - metrics[old]["overall"]),
                            "ci95_pp": [float(low), float(high)]})
    write_json("artifacts/comparisons.json", comparisons)
    effects = "\n".join(f"- `{c['from']} → {c['to']}`: {c['delta_pp']:+.2f} п.п.; "
                        f"95% парный bootstrap-интервал [{c['ci95_pp'][0]:+.2f}; {c['ci95_pp'][1]:+.2f}] п.п."
                        for c in comparisons)
    largest = max((c for c in comparisons if c["from"] != "base"), key=lambda c: abs(c["delta_pp"]))
    effects += (f"\n\nНаибольший по модулю сдвиг между RL-конфигурациями: `{largest['from']} → {largest['to']}`. "
                "В паре exact → shaped одинаковы данные, начальные веса, seed, бюджет и прочие гиперпараметры. "
                "Правило выбора чекпоинта также одинаково, но выбранные шаги могут различаться: "
                "сравниваются результаты полных запусков с этим правилом выбора. "
                "Это сравнение одного seed. Bootstrap отражает неопределённость на примерах BFCL, "
                "но не разброс между повторными обучениями. У Dr. GRPO одновременно изменены две нормировки; "
                "отделить их индивидуальные эффекты этим дизайном нельзя.")
    if "grpo_token_is" in RUNS:
        effects += ("\n\nВ дополнительной паре GRPO shaped → GRPO token IS меняется только режим коррекции "
                    "вероятностей vLLM; остальные содержательные поля конфигурации совпадают. Средний "
                    "зарегистрированный коэффициент коррекции по всем 150 train-шагам составляет "
                    f"{training_summary['grpo_shaped']['full_run_mean_importance_correction']:.3f} и "
                    f"{training_summary['grpo_token_is']['full_run_mean_importance_correction']:.3f} соответственно. "
                    "Это согласуется с гипотезой об ослаблении подавления весов, но не доказывает, "
                    "что именно длина ответа объясняет выигрыш: генерации между запусками различаются.")

    best = max(["base"] + RUNS, key=lambda run: metrics[run]["overall"])
    best_rl = max(RUNS, key=lambda run: metrics[run]["overall"])
    result_summary = (f"Лучший обученный вариант — `{best_rl}`: {metrics[best_rl]['overall']:.2%}, "
                      f"изменение к исходной модели {100 * (metrics[best_rl]['overall'] - baseline['overall']):+.2f} п.п. " +
                      ("В этих экспериментах обойти исходную модель не удалось." if best == "base" else
                       "Это лучший результат среди оценённых моделей при данном seed."))
    if best != "base":
        interval = next(c["ci95_pp"] for c in comparisons if c["from"] == "base" and c["to"] == best)
        if interval[0] <= 0 <= interval[1]:
            result_summary += (f" Парный 95% bootstrap-интервал разницы [{interval[0]:+.2f}; "
                               f"{interval[1]:+.2f}] п.п. включает ноль, поэтому устойчивое улучшение "
                               "по этой выборке и одному seed не подтверждено.")
    errors = [r for r in predictions[best] if not r["check"]["valid"]]
    examples, selected = [], set()
    selectors = {
        "Лишний вызов": lambda r: "irrelevance" in r["category"] and bool(r["calls"]),
        "Пропущенный нужный вызов": lambda r: "irrelevance" not in r["category"] and not r["calls"],
        "Ошибка нескольких вызовов": lambda r: "parallel" in r["category"] and bool(r["calls"]),
        "Ошибка выбора функции или аргументов": lambda r: r["category"] in {"simple", "multiple", "live_multiple"} and bool(r["calls"]),
    }
    error_text = [f"Лучшая модель по Overall: `{best}` ({metrics[best]['overall']:.2%}). Ошибок: {len(errors)} из {len(benchmark)}."]
    for title, condition in selectors.items():
        prediction = next((r for r in errors if condition(r) and r["id"] not in selected), None)
        if prediction is None:
            error_text.append(f"**{title}:** среди оставшихся ответов такого примера не найдено.")
            continue
        selected.add(prediction["id"])
        example = benchmark[prediction["id"]]
        examples.append({"type": title, "example": example, "prediction": prediction})
        question = " ".join(m["content"] for m in example["question"])
        predicted_names = [name for call in prediction["calls"] for name in call]
        expected_names = [name for call in example["ground_truth"] for name in call]
        if "irrelevance" in prediction["category"]:
            explanation = "В этой категории инструмент вызывать не нужно, но модель сформировала вызов."
        elif not predicted_names:
            explanation = "В эталоне есть вызов, а в ответе не удалось разобрать ни одного вызова."
        elif sorted(predicted_names) != sorted(expected_names):
            explanation = f"Набор или кратность функций отличаются: ожидались {expected_names}, получены {predicted_names}."
        else:
            explanation = "Имена и кратность функций совпали; AST-checker отклонил аргументы либо разметку ответа."
        if best in {"base", "grpo_token_is"}:
            explanation = {
                "irrelevance_233": "Для конвертации валют модель выбрала погодную функцию и подставила London. "
                                   "Связь фунтов с Великобританией не делает этот инструмент подходящим для запроса.",
                "simple_241": "Модель ответила на исторический вопрос текстом. Даже содержательно верный ответ "
                              "не заменяет обязательный вызов US_President_During_Event.",
                "parallel_105": "Первые два вызова верны. В третьем модель перенесла температуру 350 K из "
                                "предыдущего условия, тогда как эталон требует 300 K при объёме 1.5 м³. "
                                "Из-за одной ошибки отклоняется весь набор вызовов.",
                "simple_386": "Функция выбрана правильно, но room_type содержит 'luxury room' вместо "
                              "допустимого 'luxury' или 'Luxury'. Смысл близок, однако это ошибка "
                              "нормализации аргумента по правилам benchmark.",
            }.get(prediction["id"], explanation)
        shown_answer = json.dumps(prediction["calls"], ensure_ascii=False) if predicted_names else prediction["response"]
        check_label = prediction["check"].get("error_type", prediction["check"].get("error", ""))
        error_text.append(f"**{title}: `{prediction['id']}` ({prediction['category']}).**\n\n"
                          f"Запрос: {question}\n\n"
                          "Эталон (списки задают допустимые значения):\n```json\n" +
                          json.dumps(example["ground_truth"], ensure_ascii=False) + "\n```\n" +
                          ("Распарсенные вызовы:" if predicted_names else "Текст ответа:") +
                          "\n```text\n" + shown_answer + "\n```\n"
                          f"Проверка: `{str(check_label).rstrip('.')}`. {explanation}")
    error_text.append("Полные запросы, исходные ответы и сообщения AST-checker сохранены в `artifacts/error_examples.json`.")
    write_json("artifacts/error_examples.json", examples)

    reward_table = describe_reward()

    reward_relation = ("BFCL измерен для выбранного чекпоинта каждого запуска. По этим данным нельзя установить "
                       "пошаговую связь между ростом train reward и BFCL: промежуточные BFCL-замеры не проводились. "
                       "Сравнимые кривые точного совпадения приведены отдельно, поскольку масштабы exact и shaped различаются.\n\n"
                       "| Запуск | Train exact: начало → выбранный шаг | Validation exact | Доля групп с одинаковой наградой | BFCL Overall |\n|---|---:|---:|---:|---:|\n" +
                       "\n".join(f"| {run} | {training_summary[run]['first10_exact']:.2%} → "
                                 f"{training_summary[run]['selected_last10_exact']:.2%} | "
                                 f"{manifests[run]['best_validation_exact']:.2%} | "
                                 f"{training_summary[run]['zero_std_fraction']:.2%} | "
                                 f"{metrics[run]['overall']:.2%} |" for run in RUNS))
    reward_relation += ("\n\nTrain exact усреднён по первым десяти шагам и десяти шагам перед выбранным чекпоинтом. "
                        "Это разные обучающие запросы, поэтому изменение этого среднего само по себе не доказывает улучшение. "
                        "Доля групп с одинаковой наградой посчитана до выбранного шага; такие группы не дают policy-gradient сигнала.")
    reward_relation += "\n\nСредняя обучающая награда на тех же двух окнах: " + "; ".join(
        f"`{run}` {training_summary[run]['first10_reward']:.3f} → "
        f"{training_summary[run]['selected_last10_reward']:.3f}" for run in RUNS) + "."
    reward_relation += ("\n\nРазличия возможны из-за распределений ToolACE и BFCL, строгого сравнения с единственным "
                        "эталоном ToolACE и разрешённых вариантов AST-checker, а также stochastic/greedy декодирования. "
                        "Это объясняющие гипотезы, а не доказанные причины.")
    reward_relation += ("\n\nОтдельно считаю два вида ошибок решения о вызове на BFCL:\n\n"
                        "| Модель | Текст вместо нужного вызова | Вызов в Irrelevance |\n|---|---:|---:|\n" +
                        "\n".join(f"| {label} | {decisions[run]['text_instead_of_call']} | "
                                  f"{decisions[run]['call_in_irrelevance']} |"
                                  for label, run in zip(LABELS, ["base"] + RUNS)))
    reward_relation += ("\n\nПоказатель «Текст вместо нужного вызова» учитывает завершённые текстовые ответы без нужного вызова. "
                        "Это не все ошибки: отдельно остаются неверные аргументы, состав вызовов и повреждённые ответы.")
    reward_relation += ("\n\nВ трёх основных RL-запусках уменьшилось число лишних вызовов, но выросло число "
                        "пропусков нужного инструмента. Например, в `simple_26` бейзлайн правильно вызывает "
                        "`calculate_velocity(distance=50, duration=2)`, а GRPO exact вычисляет 25 км/ч текстом. "
                        "Численный ответ верен, но BFCL требует вызов. Это конкретный пример ухудшения "
                        "агентского поведения, а не ошибки арифметики. Устойчивого роста обучающей награды "
                        "в этих коротких запусках также не наблюдаю.")
    if "grpo_token_is" in RUNS:
        old, new = "grpo_shaped", "grpo_token_is"
        base_valid = {r["id"]: r["check"]["valid"] for r in predictions["base"]}
        reward_relation += (
            f"\n\nТокенная коррекция уменьшила число текстовых ответов вместо нужного вызова "
            f"с {decisions[old]['text_instead_of_call']} до {decisions[new]['text_instead_of_call']}, "
            f"но число лишних вызовов выросло с {decisions[old]['call_in_irrelevance']} "
            f"до {decisions[new]['call_in_irrelevance']}. При этом validation exact ниже "
            f"({manifests[new]['best_validation_exact']:.2%} против {manifests[old]['best_validation_exact']:.2%}), "
            f"а BFCL выше ({metrics[new]['overall']:.2%} против {metrics[old]['overall']:.2%}). "
            "Значит, ранжирование конфигураций по ToolACE не обязано совпадать с BFCL. "
            "В `simple_24` бейзлайн вычисляет НОД 12 и 18 текстом, а GRPO token IS правильно "
            "вызывает `math.gcd(num1=12, num2=18)`. Относительно бейзлайна эта модель исправила "
            f"{sum(r['check']['valid'] and not base_valid[r['id']] for r in predictions[new])} "
            "ошибок и добавила "
            f"{sum(not r['check']['valid'] and base_valid[r['id']] for r in predictions[new])} новых.")

    report = Path("report.md").read_text()
    if "grpo_token_is" in RUNS and "| GRPO token IS |" not in report:
        last_row = "| Dr. GRPO shaped | dr_grpo | Без деления на std | Shaped | Общая нормировка токенов меняет обучение ответов разной длины |"
        report = report.replace(last_row, last_row + "\n| GRPO token IS | grpo | На std внутри группы | Shaped | Токенная коррекция vLLM ослабляет накопление различий вероятностей на длинных ответах |")
        report = report.replace("Общие настройки:",
            "Дополнительный запуск GRPO token IS меняет относительно GRPO shaped только "
            "`vllm_importance_sampling_mode`: `sequence_mask` → `token_mask`. Это коррекция "
            "различий между вероятностями vLLM и обучающей модели, отдельная от PPO ratio. "
            "В основных трёх запусках используется произведение отношений по ответу; в дополнительном — "
            "отдельное отношение для каждого токена. Правило отсечения весов выше 3 одинаковое. "
            "Гипотеза возникла после анализа основных запусков: произведение может сильно уменьшать "
            "вес длинных ответов. Данные и остальные настройки сохранены.\n\nОбщие настройки:", 1)
    sections = {"status": "Эксперименты завершены. Числа ниже прочитаны из сохранённых результатов; графики построены по логам.",
                "eda": data_text, "reward_examples": reward_table,
                "results": result_summary + "\n\n" + "\n".join(table) + "\n\n![BFCL по группам](artifacts/figures/bfcl.png)\n\n"
                           "![Loss, reward и exact match](artifacts/figures/training.png)\n\n"
                           "Пунктир отмечает выбранный шаг. Бледные линии — исходные train-значения, "
                           "синие — среднее по десяти шагам, оранжевые точки — validation. "
                           "Абсолютные значения loss у GRPO и Dr. GRPO "
                           "напрямую не сравниваю: у них разные знаменатели. Loss около нуля при "
                           "центрированном advantage также не означает отсутствия градиента или ошибок модели.",
                "effects": effects, "reward_relation": reward_relation, "errors": "\n\n".join(error_text)}
    for name, content in sections.items():
        report = replace_section(report, name, content)
    Path("report.md").write_text(report)
    readme = Path("README.md")
    status = (f"**Текущее состояние:** бейзлайн и {len(RUNS)} RL-эксперимента завершены. "
              f"Лучший BFCL overall — {metrics[best]['overall']:.2%} (`{best}`). "
              "Метрики, ответы и логи сохранены; `report.md` содержит фактические результаты.")
    readme_lines = []
    for line in readme.read_text().splitlines():
        if line.startswith("**Текущее состояние:**"):
            line = status
        if line.startswith("**Дополнительный запуск:**") and "grpo_token_is" in RUNS:
            line = (f"**Дополнительный запуск:** `grpo_token_is` завершён, включён в отчёт; "
                    f"BFCL overall — {metrics['grpo_token_is']['overall']:.2%}.")
        readme_lines.append(line)
    readme.write_text("\n".join(readme_lines) + "\n")
    print("Обновлены report.md, таблица, графики и примеры ошибок. Теперь выполните render_report.py.")


if __name__ == "__main__":
    main()
