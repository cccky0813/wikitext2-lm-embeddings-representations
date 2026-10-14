"""WikiText-2 loading, preprocessing, vocabulary and LM batching (Parts I & II).

Pipeline:
    raw splits (HF datasets) -> lowercase + whitespace tokenize + <eos> per line
    -> Vocab built on train only -> token-id tensors cached in data/processed/
    -> contiguous (input, target) windows of length seq_len for training/eval.
"""
import html
import json
import re
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from .utils import ROOT

PAD, UNK, EOS = "<pad>", "<unk>", "<eos>"
SPECIALS = [PAD, UNK, EOS]


class Vocab:
    """Word <-> id mapping. Built from the training split only (no leakage)."""

    def __init__(self, tokens: list[str] | None = None, min_freq: int = 1,
                 itos: list[str] | None = None):
        if itos is None:
            counts = Counter(tokens)
            for s in SPECIALS:
                counts.pop(s, None)
            words = sorted((w for w, c in counts.items() if c >= min_freq),
                           key=lambda w: (-counts[w], w))
            itos = SPECIALS + words
        self.itos = itos
        self.stoi = {w: i for i, w in enumerate(itos)}
        self.pad_idx, self.unk_idx, self.eos_idx = (self.stoi[s] for s in SPECIALS)

    def __len__(self) -> int:
        return len(self.itos)

    def __contains__(self, word: str) -> bool:
        return word in self.stoi

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(t, self.unk_idx) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self.itos[i] for i in ids]

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump(self.itos, f)

    @classmethod
    def load(cls, path: str | Path) -> "Vocab":
        with open(path) as f:
            return cls(itos=json.load(f))


def tokenize(text: str, lowercase: bool = True) -> list[str]:
    """Whitespace tokenization; WikiText-2 is already pre-tokenized."""
    if lowercase:
        text = text.lower()
    return text.split()


_PUNCT = re.compile(r"([.,!?;:()\[\]{}\"$%&*/\\#])")


def tokenize_raw(text: str, lowercase: bool = True) -> list[str]:
    """Approximate WikiText's tokenization for *raw* text (Part III, AG News):
    split punctuation and clitics, and use WikiText's @-@ / @,@ / @.@ markers
    for intra-word hyphens and number separators."""
    text = html.unescape(text).replace("\\", " ")
    text = re.sub(r"#\d+;", " ", text)                    # leftover entities, e.g. "#39;"
    text = re.sub(r"(\w)-(\w)", r"\1 @-@ \2", text)
    text = re.sub(r"(\w)-(\w)", r"\1 @-@ \2", text)       # second pass for a-b-c
    text = re.sub(r"(\d),(\d)", r"\1 @,@ \2", text)
    text = re.sub(r"(\d)\.(\d)", r"\1 @.@ \2", text)
    text = re.sub(r"(\w)n't\b", r"\1 n't", text)
    text = re.sub(r"(\w)'(s|re|ve|ll|d|m)\b", r"\1 '\2", text, flags=re.I)
    text = _PUNCT.sub(r" \1 ", text)
    return tokenize(text, lowercase)


def _cache_dir(cfg: dict) -> Path:
    return ROOT / cfg["data"]["processed_dir"]


def load_wikitext2(cfg: dict) -> tuple[dict[str, torch.Tensor], Vocab]:
    """Return {'train'|'valid'|'test': 1-D LongTensor of ids} and the train vocab.

    Caches to cfg['data']['processed_dir'] so later runs skip the download.
    """
    cache = _cache_dir(cfg)
    ids_path, vocab_path = cache / "wikitext2_ids.pt", cache / "vocab.json"
    if ids_path.exists() and vocab_path.exists():
        return torch.load(ids_path), Vocab.load(vocab_path)

    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-v1")
    lower = cfg["data"]["lowercase"]
    tokens = {}
    for split, hf_split in [("train", "train"), ("valid", "validation"), ("test", "test")]:
        toks = []
        for line in ds[hf_split]["text"]:
            words = tokenize(line, lower)
            if words:                       # drop blank lines
                toks.extend(words + [EOS])
        tokens[split] = toks

    vocab = Vocab(tokens["train"], min_freq=cfg["data"]["min_freq"])
    ids = {s: torch.tensor(vocab.encode(t), dtype=torch.long) for s, t in tokens.items()}

    cache.mkdir(parents=True, exist_ok=True)
    torch.save(ids, ids_path)
    vocab.save(vocab_path)
    return ids, vocab


def split_sentences(ids: torch.Tensor, vocab: Vocab) -> list[list[str]]:
    """Recover lines (each ending in <eos>) from an id stream — Word2Vec input."""
    words = vocab.decode(ids.tolist())
    sents, cur = [], []
    for w in words:
        cur.append(w)
        if w == EOS:
            sents.append(cur)
            cur = []
    if cur:
        sents.append(cur)
    return sents


def make_windows(ids: torch.Tensor, seq_len: int, pad_idx: int) -> torch.Tensor:
    """Cut a 1-D stream into windows of seq_len+1 tokens with a 1-token overlap,
    so the targets ids[1:] are each predicted exactly once. The final partial
    window is right-padded with <pad> (excluded from the loss)."""
    n_targets = len(ids) - 1
    n_win = -(-n_targets // seq_len)
    padded = torch.full((n_win * seq_len + 1,), pad_idx, dtype=torch.long)
    padded[: len(ids)] = ids
    starts = torch.arange(n_win) * seq_len
    return torch.stack([padded[s: s + seq_len + 1] for s in starts.tolist()])


def make_lm_dataloader(ids: torch.Tensor, seq_len: int, batch_size: int,
                       shuffle: bool, pad_idx: int, seed: int = 42) -> DataLoader:
    """Each item is (x = w[:-1], y = w[1:]); every target is predicted from
    preceding tokens *within the same window* only (no state carried across
    windows), which is the common evaluation setup for all models."""
    w = make_windows(ids, seq_len, pad_idx)
    ds = TensorDataset(w[:, :-1].contiguous(), w[:, 1:].contiguous())
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g)
