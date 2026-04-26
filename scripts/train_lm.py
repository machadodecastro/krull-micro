import argparse, json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from tqdm import tqdm
from krull.model import KrullConfig, KrullForCausalLM
from krull.data import TextTokenDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/krull_micro.json")
    p.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    p.add_argument("--corpus", default="data/tiny_corpus.txt")
    p.add_argument("--out", default="artifacts/krull_micro.pt")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    args = p.parse_args()

    cfg = KrullConfig(**{k:v for k,v in json.load(open(args.config)).items() if k in KrullConfig.__annotations__})
    tokenizer = Tokenizer.from_file(args.tokenizer)
    cfg.vocab_size = tokenizer.get_vocab_size()
    ds = TextTokenDataset(args.corpus, tokenizer, cfg.block_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # If only CPU
    # device = "cpu"
    model = KrullForCausalLM(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    model.train()
    for epoch in range(args.epochs):
        bar = tqdm(dl, desc=f"epoch {epoch+1}")
        for x, y in bar:
            x, y = x.to(device), y.to(device)
            loss = model(x, labels=y)["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            bar.set_postfix(loss=float(loss.detach().cpu()))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg.__dict__, "model": model.state_dict()}, args.out)
    print(f"Saved model to {args.out}")

if __name__ == "__main__":
    main()
