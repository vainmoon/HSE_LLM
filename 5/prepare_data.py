import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

from common import (BASE_MODEL, BASE_REVISION, DATASET, DATA_REVISION, extract_tools,
                    file_hash, parse_reference, write_json, write_jsonl)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--max-prompt-tokens", type=int, default=3072)
    parser.add_argument("--max-answer-tokens", type=int, default=1024)
    parser.add_argument("--max-train", type=int, default=6000)
    parser.add_argument("--max-validation", type=int, default=256)
    parser.add_argument("--no-call-fraction", type=float, default=0.25)
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    from jsonschema import Draft7Validator
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source = hf_hub_download(DATASET, "data.json", repo_type="dataset", revision=DATA_REVISION)
    raw = json.loads(Path(source).read_text())
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    rng = random.Random(args.seed)
    # Одинаковые диалоги получают один id даже при различии system prompt.
    ids = [hashlib.sha256(json.dumps(row["conversations"], sort_keys=True,
                                  ensure_ascii=False).encode()).hexdigest() for row in raw]
    unique_ids = sorted(set(ids))
    rng.shuffle(unique_ids)
    validation_ids = set(unique_ids[:round(len(unique_ids) * args.validation_fraction)])
    rows = {"train": [], "validation": []}
    skipped, seen, tool_names, dialogue_lengths = Counter(), set(), Counter(), []
    lengths = {"prompt": [], "answer": [], "tools": [], "calls": []}
    for source_index, (row, dialogue_id) in enumerate(zip(raw, ids)):
        if dialogue_id in seen:
            skipped["duplicate_dialogue"] += 1
            continue
        seen.add(dialogue_id)
        dialogue_lengths.append(len(row["conversations"]))
        try:
            tools = extract_tools(row["system"])
        except (ValueError, TypeError, KeyError):
            skipped["unsupported_tools_format"] += 1
            continue
        schemas = {t["function"]["name"]: t["function"]["parameters"] for t in tools}
        tool_names.update(schemas.keys())
        current_time = re.search(r"The current time is ([0-9:\- ]+)", row["system"])
        messages = ([{"role": "system", "content": f"The current time is {current_time[1]}."}]
                    if current_time else [])
        split = "validation" if dialogue_id in validation_ids else "train"
        for turn, message in enumerate(row["conversations"]):
            role, content = message["from"], message["value"]
            if role == "assistant":
                try:
                    calls = parse_reference(content, tools)
                except (ValueError, SyntaxError, TypeError):
                    # Следующие префиксы уже содержали бы испорченный ответ.
                    skipped["invalid_reference_or_history"] += 1
                    break
                if messages and messages[-1]["role"] == "user":
                    if any(not Draft7Validator(schemas[c["name"]]).is_valid(c["arguments"])
                           for c in calls):
                        skipped["reference_schema_mismatch"] += 1
                    else:
                        prompt = tokenizer.apply_chat_template(
                            messages, tools=tools, tokenize=False,
                            add_generation_prompt=True, enable_thinking=False)
                        answer = {"role": "assistant", "content": "" if calls else content}
                        if calls:
                            answer["tool_calls"] = [{"type": "function", "function": c} for c in calls]
                        full = tokenizer.apply_chat_template(
                            messages + [answer], tools=tools, tokenize=False, enable_thinking=False)
                        prompt_length = len(tokenizer.encode(prompt, add_special_tokens=False))
                        answer_length = len(tokenizer.encode(full[len(prompt):], add_special_tokens=False))
                        lengths["prompt"].append(prompt_length)
                        lengths["answer"].append(answer_length)
                        lengths["tools"].append(len(tools))
                        lengths["calls"].append(len(calls))
                        if prompt_length > args.max_prompt_tokens:
                            skipped["long_prompt"] += 1
                        elif answer_length > args.max_answer_tokens:
                            skipped["long_answer"] += 1
                        elif not content.strip():
                            skipped["empty_answer"] += 1
                        else:
                            rows[split].append({
                                "id": f"{dialogue_id[:16]}_{turn}", "dialogue_id": dialogue_id,
                                "source_index": source_index, "turn": turn, "prompt": prompt,
                                "tools_json": json.dumps(tools, ensure_ascii=False),
                                "gold_json": json.dumps(calls, ensure_ascii=False),
                                "reference_text": content, "prompt_tokens": prompt_length,
                                "answer_tokens": answer_length, "n_tools": len(tools),
                                "kind": "no_call" if not calls else "single" if len(calls) == 1 else "parallel",
                            })
                converted = {"role": "assistant", "content": "" if calls else content}
                if calls:
                    converted["tool_calls"] = [{"type": "function", "function": c} for c in calls]
                messages.append(converted)
            elif role in {"user", "tool"}:
                messages.append({"role": role, "content": content})
            else:
                skipped["unknown_role"] += 1
                break

    train_prompts = {r["prompt"] for r in rows["train"]}
    clean_validation = [r for r in rows["validation"] if r["prompt"] not in train_prompts]
    skipped["duplicate_validation_prompt"] = len(rows["validation"]) - len(clean_validation)
    rows["validation"] = clean_validation
    before_sampling = {split: dict(Counter(r["kind"] for r in values)) for split, values in rows.items()}
    positive = [r for r in rows["train"] if r["kind"] != "no_call"]
    negative = [r for r in rows["train"] if r["kind"] == "no_call"]
    rng.shuffle(positive)
    rng.shuffle(negative)
    n_positive = min(len(positive), round(args.max_train * (1 - args.no_call_fraction)))
    n_negative = min(len(negative), round(n_positive * args.no_call_fraction / (1 - args.no_call_fraction)))
    rows["train"] = positive[:n_positive] + negative[:n_negative]
    rng.shuffle(rows["train"])
    rng.shuffle(rows["validation"])
    rows["validation"] = rows["validation"][:args.max_validation]
    if not rows["train"] or not rows["validation"]:
        raise ValueError("Пустой сплит: проверьте разбор ToolACE")
    for split, values in rows.items():
        write_jsonl(f"data/{split}.jsonl", values)
    write_json("data/split_ids.json", {
        "seed": args.seed, "train": sorted(set(unique_ids) - validation_ids),
        "validation": sorted(validation_ids),
    })
    metadata = {
        "dataset": DATASET, "revision": DATA_REVISION, "raw_sha256": file_hash(source),
        "base_model": BASE_MODEL, "base_revision": BASE_REVISION, "settings": vars(args),
        "raw_dialogues": len(raw), "unique_dialogues": len(unique_ids),
        "raw_messages": sum(len(r["conversations"]) for r in raw),
        "unique_dialogue_messages": sum(dialogue_lengths), "unique_tools": len(tool_names),
        "skipped": dict(skipped), "before_sampling": before_sampling,
        "splits": {split: {"examples": len(values),
                           "dialogues": len({r["dialogue_id"] for r in values}),
                           "kinds": dict(Counter(r["kind"] for r in values)),
                           "sha256": file_hash(f"data/{split}.jsonl")}
                   for split, values in rows.items()},
        "length_quantiles_before_filter": {
            key: dict(zip(["p0", "p50", "p90", "p95", "p99", "p100"],
                          np.percentile(value, [0, 50, 90, 95, 99, 100]).tolist()))
            for key, value in lengths.items()},
    }
    write_json("artifacts/eda.json", metadata)
    write_json("data/manifest.json", metadata)
    write_json("artifacts/data_examples.json", {
        kind: next((r for r in rows["train"] if r["kind"] == kind), None)
        for kind in ["single", "parallel", "no_call"]})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    titles = {"prompt": "Длина промпта", "answer": "Длина эталона",
              "tools": "Доступные инструменты", "calls": "Вызовы в эталоне"}
    for ax, (key, value) in zip(axes.flat, lengths.items()):
        if key in {"tools", "calls"}:
            counts = Counter(value)
            ax.bar(sorted(counts), [counts[k] for k in sorted(counts)])
            ax.set_xticks(sorted(counts))
        else:
            ax.hist(value, bins=40)
        ax.set(title=titles[key], ylabel="Число примеров",
               xlabel="Токены" if key in {"prompt", "answer"} else "Количество")
    fig.tight_layout()
    Path("artifacts/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("artifacts/figures/eda.png", dpi=160)
    plt.close(fig)
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
