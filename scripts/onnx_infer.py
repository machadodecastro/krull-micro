import argparse, numpy as np, onnxruntime as ort
from tokenizers import Tokenizer


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", default="artifacts/krull_micro_int8.onnx")
    p.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    p.add_argument("--prompt", default="Krull is")
    p.add_argument("--max-new-tokens", type=int, default=40)
    args = p.parse_args()
    tok = Tokenizer.from_file(args.tokenizer)
    ids = tok.encode(args.prompt).ids
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    for _ in range(args.max_new_tokens):
        inp = np.array([ids[-256:]], dtype=np.int64)
        logits = sess.run(None, {"input_ids": inp})[0][0, -1]
        probs = softmax(logits / 0.8)
        next_id = int(np.random.choice(len(probs), p=probs))
        ids.append(next_id)
    print(tok.decode(ids))

if __name__ == "__main__":
    main()
