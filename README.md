# Krull

Krull is an acronym for Knowledge Running Under Lightweight Language.

Krull-Micro is a tiny decoder-only Transformer language model for text generation on edge devices such as smartphones and low-RAM embedded systems.

Krull-Micro is designed for very constrained devices:

- Layers: 2
- Hidden size: 192
- Heads: 3 or 4
- FFN size: 512
- Context length: 128–256
- Parameters: 3M

It uses a TinyBERT-inspired comprehensive distillation method, adapted to a causal GPT-style student:

- embedding-layer output distillation
- transformer hidden-state distillation
- causal attention-map distillation
- soft-target prediction distillation
- standard next-token cross-entropy

> Important: this repository is a compact reference implementation. For a useful model, replace `data/tiny_corpus.txt` with a large, clean corpus.

---

## 1. Project structure

```text
krull_micro/
├── configs/
│   └── krull_micro.json
├── data/
│   └── tiny_corpus.txt
├── krull/
│   ├── __init__.py
│   ├── data.py
│   └── model.py
├── scripts/
│   ├── train_tokenizer.py
│   ├── train_lm.py
│   ├── distill_krull.py
│   ├── generate.py
│   ├── export_onnx.py
│   ├── quantize_onnx.py
│   ├── onnx_infer.py
│   └── push_to_hf.py
├── requirements.txt
└── README.md
```

---

## 2. Create environment

```bash
cd krull_micro
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
cd krull_micro
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Prepare corpus

The demo corpus is here:

```text
data/tiny_corpus.txt
```

For a real model, replace it with a larger file:

```bash
cp /path/to/your_corpus.txt data/train.txt
```

Recommended corpus size:

```text
Demo:      < 1 MB
Toy model: 10 MB - 100 MB
Useful:    1 GB+
```

Use plain UTF-8 text.

---

## 4. Train tokenizer

```bash
python scripts/train_tokenizer.py \
  --corpus data/train.txt \
  --out artifacts/tokenizer.json \
  --vocab-size 8000
```

For the included demo:

```bash
python scripts/train_tokenizer.py \
  --corpus data/tiny_corpus.txt \
  --out artifacts/tokenizer.json \
  --vocab-size 8000
```

---

## 5. Train Krull-Micro without teacher

This trains the student directly with next-token prediction.

```bash
python scripts/train_lm.py \
  --config configs/krull_micro.json \
  --tokenizer artifacts/tokenizer.json \
  --corpus data/train.txt \
  --out artifacts/krull_micro.pt \
  --epochs 3 \
  --batch-size 8 \
  --lr 3e-4
```

Demo command:

```bash
python scripts/train_lm.py \
  --config configs/krull_micro.json \
  --tokenizer artifacts/tokenizer.json \
  --corpus data/tiny_corpus.txt \
  --out artifacts/krull_micro.pt \
  --epochs 3 \
  --batch-size 4
```

---

## 6. Distill from a causal teacher

Krull uses a TinyBERT-like distillation strategy, but the teacher should be a causal language model, not BERT.

Default teacher:

```text
distilgpt2
```

Run distillation:

```bash
python scripts/distill_krull.py \
  --teacher distilgpt2 \
  --config configs/krull_micro.json \
  --tokenizer artifacts/tokenizer.json \
  --corpus data/train.txt \
  --out artifacts/krull_micro_distilled.pt \
  --epochs 1 \
  --batch-size 2 \
  --temperature 3.0
```

Demo command:

```bash
python scripts/distill_krull.py \
  --teacher distilgpt2 \
  --config configs/krull_micro.json \
  --tokenizer artifacts/tokenizer.json \
  --corpus data/tiny_corpus.txt \
  --out artifacts/krull_micro_distilled.pt \
  --epochs 1 \
  --batch-size 2
```

### Distillation losses

The distillation objective is:

```text
Total Loss =
  alpha   * embedding MSE
+ beta    * hidden-state MSE
+ gamma   * causal attention-map MSE
+ delta   * soft-logit KL divergence
+ epsilon * hard next-token CE
```

Default weights:

```text
alpha   = 1.0
beta    = 1.0
gamma   = 1.0
delta   = 2.0
epsilon = 0.5
temperature = 3.0
```

### Important production note

The simple demo distillation script assumes student and teacher token IDs are compatible enough for experimentation. For serious training, use one of these approaches:

1. Train Krull with the teacher tokenizer.
2. Tokenize the same text separately for teacher and student and align by text spans.
3. Use teacher-generated synthetic data and train Krull on the generated text.

---

## 7. Generate text with PyTorch

Using directly trained model:

```bash
python scripts/generate.py \
  --model artifacts/krull_micro.pt \
  --tokenizer artifacts/tokenizer.json \
  --prompt "Krull is" \
  --max-new-tokens 80
```

Using distilled model:

```bash
python scripts/generate.py \
  --model artifacts/krull_micro_distilled.pt \
  --tokenizer artifacts/tokenizer.json \
  --prompt "Krull is" \
  --max-new-tokens 80
```

---

## 8. Convert Krull to ONNX

Export PyTorch checkpoint to ONNX:

```bash
python scripts/export_onnx.py \
  --model artifacts/krull_micro_distilled.pt \
  --out artifacts/krull_micro.onnx \
  --seq-len 32
```

If you trained without distillation:

```bash
python scripts/export_onnx.py \
  --model artifacts/krull_micro.pt \
  --out artifacts/krull_micro.onnx \
  --seq-len 32
```

---

## 9. Quantize ONNX to INT8

```bash
python scripts/quantize_onnx.py \
  --onnx artifacts/krull_micro.onnx \
  --out artifacts/krull_micro_int8.onnx
```

This creates a smaller model for CPU inference.

---

## 10. Run ONNX inference

```bash
python scripts/onnx_infer.py \
  --onnx artifacts/krull_micro_int8.onnx \
  --tokenizer artifacts/tokenizer.json \
  --prompt "Krull is" \
  --max-new-tokens 60
```

---

## 11. Export to Hugging Face Hub

### 11.1 Login

```bash
huggingface-cli login
```

Paste your Hugging Face access token.

### 11.2 Upload model package

Replace `YOUR_USERNAME` with your Hugging Face username or organization.

```bash
python scripts/push_to_hf.py \
  --repo-id YOUR_USERNAME/krull-micro \
  --model artifacts/krull_micro_distilled.pt \
  --onnx artifacts/krull_micro_int8.onnx \
  --tokenizer artifacts/tokenizer.json \
  --config configs/krull_micro.json
```

For a private repository:

```bash
python scripts/push_to_hf.py \
  --repo-id YOUR_USERNAME/krull-micro \
  --private
```

The script uploads:

```text
krull_micro.pt
krull_micro_int8.onnx
tokenizer.json
config.json
README.md
```

---

## 12. Suggested edge deployment targets

For mobile or embedded deployment:

```text
Android: ONNX Runtime Mobile
Linux ARM: ONNX Runtime CPU
iOS: ONNX Runtime Mobile or CoreML conversion path
Micro edge Linux: INT8 ONNX Runtime
```

Recommended settings:

```text
context length: 128-256
batch size: 1
precision: INT8 first, INT4 later
sampling: top-k 40-50, temperature 0.7-0.9
```

---

## 13. Krull-Micro model size

Default config:

```text
layers: 2
hidden size: 192
heads: 4
FFN multiplier: 4
context length: 256
vocab size: 8000
```

Approximate target:

```text
FP32: ~15-35 MB depending on vocab
INT8: ~4-12 MB depending on export and runtime
```

---

## 14. License suggestion

Use Apache-2.0 or MIT for open release.

---

## 15. Minimal full demo

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python scripts/train_tokenizer.py --corpus data/tiny_corpus.txt --out artifacts/tokenizer.json --vocab-size 8000
python scripts/train_lm.py --corpus data/tiny_corpus.txt --out artifacts/krull_micro.pt --epochs 3 --batch-size 4
python scripts/generate.py --model artifacts/krull_micro.pt --prompt "Krull is"
python scripts/export_onnx.py --model artifacts/krull_micro.pt --out artifacts/krull_micro.onnx
python scripts/quantize_onnx.py --onnx artifacts/krull_micro.onnx --out artifacts/krull_micro_int8.onnx
python scripts/onnx_infer.py --onnx artifacts/krull_micro_int8.onnx --prompt "Krull is"
```
