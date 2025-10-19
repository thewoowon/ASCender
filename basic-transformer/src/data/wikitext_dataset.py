import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from collections import Counter


class WikiTextDataset(Dataset):
    """
    Hugging Face WikiText Dataset
    (wikitext-2-v1 / wikitext-103-v1 둘 다 지원)
    기존 구조와 완전히 호환되도록 설계.
    """

    def __init__(self, name="wikitext-2-v1", split="train", seq_len=64, vocab_size=30000):
        self.seq_len = seq_len
        self.name = name

        print(f"[WikiTextDataset] Loading {name} ({split}) from Hugging Face...")
        dataset = load_dataset("wikitext", name, split=split)
        text = " ".join(dataset["text"]).replace("\n", " ")

        # --- 기존 whitespace tokenizer 유지 ---
        tokens = text.split()
        vocab = Counter(tokens)
        self.itos = ["<pad>", "<unk>"] + [t for t, _ in vocab.most_common(vocab_size)]
        self.stoi = {t: i for i, t in enumerate(self.itos)}

        ids = [self.stoi.get(t, 1) for t in tokens]  # 1 = <unk>
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


class WikiText2Dataset(WikiTextDataset):
    """WikiText-2 전용"""
    def __init__(self, split="train", seq_len=64):
        super().__init__(name="wikitext-2-v1", split=split, seq_len=seq_len)


class WikiText103Dataset(WikiTextDataset):
    """WikiText-103 전용"""
    def __init__(self, split="train", seq_len=64):
        super().__init__(name="wikitext-103-v1", split=split, seq_len=seq_len)


def get_dataloader(batch_size=16, seq_len=64, split="train", use_103=False):
    """
    기존 코드와 완전히 동일한 인터페이스 유지.
    단, 내부는 Hugging Face 기반으로 작동.
    """
    dataset_cls = WikiText103Dataset if use_103 else WikiText2Dataset
    dataset = dataset_cls(split=split, seq_len=seq_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split == "train"))
    return loader, dataset
