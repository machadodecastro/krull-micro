import argparse
from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/tiny_corpus.txt")
    p.add_argument("--out", default="artifacts/tokenizer.json")
    p.add_argument("--vocab-size", type=int, default=8000)
    args = p.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab_size,
        special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
        min_frequency=1,
    )
    tokenizer.train([args.corpus], trainer)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.save(args.out)
    print(f"Saved tokenizer to {args.out}")

if __name__ == "__main__":
    main()
