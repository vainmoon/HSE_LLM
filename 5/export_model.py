import argparse
import json
import shutil
from pathlib import Path

from common import BASE_MODEL, BASE_REVISION, file_hash, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    args = parser.parse_args()
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import Qwen3_5ForConditionalGeneration

    manifest = json.loads((Path("artifacts") / args.run / "run.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("Обучение ещё не завершено")
    output = Path("checkpoints") / args.run / "merged"
    snapshot = Path(snapshot_download(BASE_MODEL, revision=BASE_REVISION))
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        snapshot, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, manifest["adapter"])
    model = model.merge_and_unload(safe_merge=True)
    model.save_pretrained(output, safe_serialization=True)
    # Сохраняем исходные файлы токенизатора и процессоров побайтно.
    assets = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt",
              "chat_template.jinja", "preprocessor_config.json", "video_preprocessor_config.json"]
    for name in assets:
        shutil.copy2(snapshot / name, output / name)
    provenance = {
        "base_model": BASE_MODEL, "base_revision": BASE_REVISION,
        "run": args.run, "selected_step": manifest["selected_step"],
        "export_source_sha256": file_hash(__file__),
        "processor_files_sha256": {name: file_hash(output / name) for name in assets},
    }
    write_json(output / "training_provenance.json", provenance)
    write_json(Path("artifacts") / args.run / "export.json", provenance)
    print(f"Полный чекпоинт: {output}")


if __name__ == "__main__":
    main()
