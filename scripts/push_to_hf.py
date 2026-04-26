import argparse, json, shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", required=True, help="Example: username/krull-micro")
    p.add_argument("--model", default="artifacts/krull_micro.pt")
    p.add_argument("--onnx", default="artifacts/krull_micro_int8.onnx")
    p.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    p.add_argument("--config", default="configs/krull_micro.json")
    p.add_argument("--private", action="store_true")
    args = p.parse_args()
    out = Path("hf_package")
    out.mkdir(exist_ok=True)
    shutil.copy(args.model, out / "krull_micro.pt")
    if Path(args.onnx).exists(): shutil.copy(args.onnx, out / "krull_micro_int8.onnx")
    shutil.copy(args.tokenizer, out / "tokenizer.json")
    shutil.copy(args.config, out / "config.json")
    (out / "README.md").write_text("""---
language: en
license: apache-2.0
tags:
- causal-lm
- tiny-transformer
- edge-ai
- onnx
---

# Krull-Micro

Krull-Micro is a tiny decoder-only Transformer language model for edge text generation.

This repository contains PyTorch checkpoint, tokenizer, config, and optional INT8 ONNX export.
""", encoding="utf-8")
    create_repo(args.repo_id, private=args.private, exist_ok=True)
    upload_folder(repo_id=args.repo_id, folder_path=str(out))
    print(f"Uploaded to https://huggingface.co/{args.repo_id}")

if __name__ == "__main__":
    main()
