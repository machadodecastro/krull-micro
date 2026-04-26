import argparse, json
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tokenizers import Tokenizer
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from krull.model import KrullConfig, KrullForCausalLM
from krull.data import TextTokenDataset


def mse_with_projection(student, teacher, proj=None):
    if proj is not None:
        student = proj(student)
    min_t = min(student.shape[1], teacher.shape[1])
    min_c = min(student.shape[-1], teacher.shape[-1])
    return F.mse_loss(student[:, :min_t, :min_c], teacher[:, :min_t, :min_c])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="distilgpt2")
    p.add_argument("--config", default="configs/krull_micro.json")
    p.add_argument("--tokenizer", default="artifacts/tokenizer.json")
    p.add_argument("--corpus", default="data/tiny_corpus.txt")
    p.add_argument("--out", default="artifacts/krull_micro_distilled.pt")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--temperature", type=float, default=3.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--delta", type=float, default=2.0)
    p.add_argument("--epsilon", type=float, default=0.5)
    args = p.parse_args()

    cfg_data = json.load(open(args.config))
    cfg = KrullConfig(**{k:v for k,v in cfg_data.items() if k in KrullConfig.__annotations__})
    tokenizer = Tokenizer.from_file(args.tokenizer)
    cfg.vocab_size = tokenizer.get_vocab_size()
    ds = TextTokenDataset(args.corpus, tokenizer, cfg.block_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # If only CPU
    # device = "cpu" 

    student = KrullForCausalLM(cfg).to(device)
    teacher = AutoModelForCausalLM.from_pretrained(args.teacher, output_hidden_states=True, output_attentions=True).to(device)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False

    opt = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=0.01)

    for epoch in range(args.epochs):
        bar = tqdm(dl, desc=f"distill epoch {epoch+1}")
        for x, y in bar:
            x, y = x.to(device), y.to(device)
            # This assumes compatible token IDs. For serious training, use the teacher tokenizer
            # or build a corpus pre-tokenized for both student and teacher.
            with torch.no_grad():
                t = teacher(input_ids=x, labels=y, output_hidden_states=True, output_attentions=True)
            s = student(x, labels=y, output_hidden_states=True, output_attentions=True)

            embed_loss = mse_with_projection(s["embedding_output"], t.hidden_states[0])

            hidden_loss = 0.0
            pairs = min(len(s["hidden_states"]), len(t.hidden_states))
            for i in range(1, pairs):
                t_idx = round(i * (len(t.hidden_states)-1) / max(1, len(s["hidden_states"])-1))
                hidden_loss = hidden_loss + mse_with_projection(s["hidden_states"][i], t.hidden_states[t_idx])
            hidden_loss = hidden_loss / max(1, pairs-1)

            attn_loss = 0.0
            attn_pairs = min(len(s["attentions"]), len(t.attentions))
            for i in range(attn_pairs):
                t_idx = round(i * (len(t.attentions)-1) / max(1, len(s["attentions"])-1))
                s_att = s["attentions"][i]
                t_att = t.attentions[t_idx]
                h = min(s_att.shape[1], t_att.shape[1])
                n = min(s_att.shape[-1], t_att.shape[-1])
                attn_loss = attn_loss + F.mse_loss(s_att[:, :h, :n, :n], t_att[:, :h, :n, :n])
            attn_loss = attn_loss / max(1, attn_pairs)

            T = args.temperature
            vocab = min(s["logits"].shape[-1], t.logits.shape[-1])
            soft_loss = F.kl_div(
                F.log_softmax(s["logits"][..., :vocab] / T, dim=-1),
                F.softmax(t.logits[..., :vocab] / T, dim=-1),
                reduction="batchmean",
            ) * (T * T)

            loss = args.alpha*embed_loss + args.beta*hidden_loss + args.gamma*attn_loss + args.delta*soft_loss + args.epsilon*s["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()
            bar.set_postfix(loss=float(loss.detach().cpu()))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": cfg.__dict__, "model": student.state_dict()}, args.out)
    print(f"Saved distilled Krull to {args.out}")

if __name__ == "__main__":
    main()
