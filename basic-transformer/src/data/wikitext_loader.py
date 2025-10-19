import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter


class WikiTextDataset(Dataset):
    def __init__(self, cfg):
        """
        cfg.dataset.name: "wikitext-2-v1" or "wikitext-103-v1"
        cfg.dataset.seq_len
        cfg.dataset.vocab_size
        """
        self.seq_len = cfg.dataset.seq_len
        self.name = cfg.dataset.name
        self.vocab_size = cfg.dataset.vocab_size

        print(f"[WikiTextDataset] Loading {self.name} (train) from Hugging Face...")
        dataset = load_dataset("wikitext", self.name, split="train")
        text = " ".join(dataset["text"]).replace("\n", " ")

        # whitespace tokenizer
        tokens = text.split()
        vocab = Counter(tokens)
        self.itos = ["<pad>", "<unk>"] + [t for t, _ in vocab.most_common(self.vocab_size)]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

        ids = [self.stoi.get(t, 1) for t in tokens]  # 1=<unk>
        self.data = torch.tensor(ids, dtype=torch.long)
        self.pad_id = 0

        print(f"[WikiTextDataset] Loaded {len(self.data):,} tokens.")

    def __len__(self):
        return len(self.data) // self.seq_len - 1

    def __getitem__(self, idx):
        start = idx * self.seq_len
        src = self.data[start:start + self.seq_len]
        tgt_inp = self.data[start:start + self.seq_len]
        tgt_out = self.data[start + 1:start + 1 + self.seq_len]
        return src, tgt_inp, tgt_out


def get_dataloader(cfg, split="train"):
    """
    cfg: 전체 YAML 파싱된 Namespace or dict
    cfg.dataset.batch_size, seq_len 등 사용
    """
    print("cfg:", cfg)
    dataset = WikiTextDataset(cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
        drop_last=True,
    )
    return loader, dataset
