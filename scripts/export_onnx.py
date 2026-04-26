import argparse, torch
from pathlib import Path
from krull.model import KrullConfig, KrullForCausalLM

class KrullOnnxWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, input_ids):
        return self.model(input_ids)["logits"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="artifacts/krull_micro.pt")
    p.add_argument("--out", default="artifacts/krull_micro.onnx")
    p.add_argument("--seq-len", type=int, default=32)
    args = p.parse_args()
    ckpt = torch.load(args.model, map_location="cpu")
    cfg = KrullConfig(**ckpt["config"])
    model = KrullForCausalLM(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    wrapper = KrullOnnxWrapper(model)
    dummy = torch.randint(0, cfg.vocab_size, (1, args.seq_len), dtype=torch.long)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        dummy,
        args.out,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "sequence"}, "logits": {0: "batch", 1: "sequence"}},
        opset_version=17,
    )
    print(f"Exported ONNX to {args.out}")

if __name__ == "__main__":
    main()
