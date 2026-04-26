from pathlib import Path
import torch
from torch.utils.data import Dataset


class TextTokenDataset(Dataset):
    def __init__(self, text_file, tokenizer, block_size=256):
        text = Path(text_file).read_text(encoding="utf-8")
        ids = tokenizer.encode(text).ids
        if len(ids) < block_size + 1:
            ids = ids * ((block_size + 1) // max(1, len(ids)) + 2)
        self.data = torch.tensor(ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return max(1, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]
