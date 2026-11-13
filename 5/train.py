import argparse
import json
import os
import time
from importlib.metadata import version
from pathlib import Path

from common import BASE_MODEL, BASE_REVISION, file_hash, read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--smoke", action="store_true", help="Два шага, отдельные артефакты")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--no-vllm", action="store_true", help="Генерации Transformers на той же GPU")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "7":
        raise ValueError("Перед запуском задайте CUDA_VISIBLE_DEVICES=7")

    import torch
    from datasets import Dataset
    from huggingface_hub import snapshot_download
    from peft import LoraConfig
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration, TrainerCallback, set_seed
    from trl import GRPOConfig, GRPOTrainer
    from rewards import ToolReward

    config = json.loads(args.config.read_text())
    settings = json.loads(Path("configs/common.json").read_text()) | config["trainer"]
    run = args.config.stem + ("_smoke" if args.smoke else "")
    output = Path("checkpoints") / run
    artifacts = Path("artifacts") / run
    artifacts.mkdir(parents=True, exist_ok=True)
    if (artifacts / "run.json").exists() and not args.resume:
        raise FileExistsError(f"Запуск {run} уже существует; используйте --resume или новое имя конфигурации")
    if args.smoke:
        settings.update(max_steps=2, eval_strategy="no", save_strategy="no", load_best_model_at_end=False)
    if args.no_vllm:
        settings["use_vllm"] = False
    settings.update(output_dir=str(output), run_name=run)
    set_seed(settings["seed"])
    # Локальный snapshot не даёт vLLM незаметно загрузить новую main-ревизию.
    snapshot = snapshot_download(BASE_MODEL, revision=BASE_REVISION)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, padding_side="left")
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        snapshot, dtype=torch.bfloat16, attn_implementation="sdpa")
    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=(r"model\.language_model\.layers\.\d+\."
                        r"(self_attn\.(q_proj|k_proj|v_proj|o_proj)|"
                        r"linear_attn\.(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)|"
                        r"mlp\.(gate_proj|up_proj|down_proj))"),
    )
    train_rows, val_rows = read_jsonl("data/train.jsonl"), read_jsonl("data/validation.jsonl")
    if args.smoke:
        train_rows = train_rows[:16]
        val_rows = val_rows[:4]
    eos_ids = model.generation_config.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    reward = ToolReward(tokenizer, config["reward"], settings["max_completion_length"], eos_ids)
    monitor = ToolReward(tokenizer, "exact", settings["max_completion_length"], eos_ids)

    def training_reward(**kwargs):
        return reward(**kwargs)

    def exact_match(**kwargs):
        kwargs.pop("log_metric", None)
        return monitor(**kwargs)

    class SaveLogs(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if state.is_world_process_zero and logs:
                with (artifacts / "logs.jsonl").open("a") as stream:
                    stream.write(json.dumps({"step": state.global_step, **logs}) + "\n")

    trainer = GRPOTrainer(
        model=model, processing_class=tokenizer, peft_config=lora,
        args=GRPOConfig(**settings), reward_funcs=[training_reward, exact_match],
        train_dataset=Dataset.from_list(train_rows), eval_dataset=Dataset.from_list(val_rows),
        callbacks=[SaveLogs()],
    )
    manifest = {
        "run": run, "status": "running", "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
        "experiment": config, "trainer": trainer.args.to_dict(), "lora": lora.to_dict(),
        "data": json.loads(Path("data/manifest.json").read_text()),
        "train_sha256": file_hash("data/train.jsonl"), "validation_sha256": file_hash("data/validation.jsonl"),
        "versions": {p: version(p) for p in ["torch", "transformers", "trl", "peft", "vllm", "datasets"]},
        "source_sha256": {p: file_hash(p) for p in ["train.py", "rewards.py", "common.py", "prepare_data.py"]},
        "gpu": torch.cuda.get_device_name(0), "smoke": args.smoke,
    }
    # PEFT хранит некоторые поля как set.
    manifest["lora"] = json.loads(json.dumps(manifest["lora"], default=list))
    write_json(artifacts / "run.json", manifest)
    start = time.monotonic()
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(output / "adapter"))
    tokenizer.save_pretrained(output / "adapter")
    trainer.save_state()
    write_json(artifacts / "trainer_state.json", json.loads((output / "trainer_state.json").read_text()))
    best = trainer.state.best_model_checkpoint
    manifest.update(status="complete", best_checkpoint=best,
                    selected_step=int(Path(best).name.split("-")[-1]) if best else trainer.state.global_step,
                    best_validation_exact=trainer.state.best_metric,
                    elapsed_seconds=round(time.monotonic() - start, 1),
                    adapter=str(output / "adapter"))
    write_json(artifacts / "run.json", manifest)


if __name__ == "__main__":
    main()
