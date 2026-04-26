import argparse, json, torch
from tokenizers import Tokenizer
from krull.model import KrullConfig, KrullForCausalLM


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="artifacts/krull_micro.pt")
    p.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    p.add_argument("--prompt", default="Krull is")
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.8)
    args = p.parse_args()
    ckpt = torch.load(args.model, map_location="cpu")
    cfg = KrullConfig(**ckpt["config"])
    model = KrullForCausalLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tokenizer = Tokenizer.from_file(args.tokenizer)
    ids = torch.tensor([tokenizer.encode(args.prompt).ids], dtype=torch.long)
    out = model.generate(ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    print(tokenizer.decode(out[0].tolist()))

if __name__ == "__main__":
    main()
