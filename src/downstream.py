"""Part III: frozen-LM representations + logistic-regression classifier."""
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import Vocab, tokenize_raw

AG_LABELS = ["World", "Sports", "Business", "Sci/Tech"]


def load_ag_news(cfg: dict, seed: int) -> dict:
    """Stratified train/val subsets (disjoint) from the official train split and
    a stratified test subset from the official test split.
    Returns {split: {"text": [...], "label": np.ndarray}}."""
    from datasets import load_dataset
    ds = load_dataset("fancyzhx/ag_news")
    t = cfg["task"]
    tr_text, tr_y = ds["train"]["text"], np.array(ds["train"]["label"])
    idx = np.arange(len(tr_y))
    idx_tr, idx_rest = train_test_split(idx, train_size=t["train_size"], stratify=tr_y,
                                        random_state=seed)
    idx_val, _ = train_test_split(idx_rest, train_size=t["val_size"], stratify=tr_y[idx_rest],
                                  random_state=seed)
    te_text, te_y = ds["test"]["text"], np.array(ds["test"]["label"])
    idx_te, _ = train_test_split(np.arange(len(te_y)), train_size=t["test_size"],
                                 stratify=te_y, random_state=seed)
    pick = lambda text, y, ii: {"text": [text[i] for i in ii], "label": y[ii]}  # noqa: E731
    return {"train": pick(tr_text, tr_y, idx_tr), "val": pick(tr_text, tr_y, idx_val),
            "test": pick(te_text, te_y, idx_te)}


def encode_texts(vocab: Vocab, texts: list[str], max_len: int) -> tuple[torch.Tensor, dict]:
    """Tokenize like WikiText, map to LM ids, truncate and right-pad.
    Returns ids[N, max_len] and tokenization stats (OOV rate, truncation)."""
    seqs, n_tok, n_unk, n_trunc = [], 0, 0, 0
    for text in texts:
        ids = vocab.encode(tokenize_raw(text))
        n_tok += len(ids)
        n_unk += sum(i == vocab.unk_idx for i in ids)
        n_trunc += len(ids) > max_len
        ids = ids[:max_len] or [vocab.unk_idx]
        seqs.append(ids + [vocab.pad_idx] * (max_len - len(ids)))
    stats = {"mean_len": n_tok / len(texts), "unk_rate": n_unk / n_tok,
             "truncated_frac": n_trunc / len(texts)}
    return torch.tensor(seqs), stats


@torch.no_grad()
def extract_features(model: nn.Module, ids: torch.Tensor, pad_idx: int, pooling: str,
                     batch_size: int, device: torch.device) -> np.ndarray:
    """Run the frozen LM and pool top-layer states into one vector per text.

    pooling='last'    : state at the last non-pad position.
    pooling='mean'    : mean over non-pad positions (mask excludes padding).
    pooling='emb_mean': mean of input embeddings over non-pad positions.
    """
    model.eval()
    feats = []
    for i in range(0, len(ids), batch_size):
        x = ids[i: i + batch_size].to(device)
        mask = (x != pad_idx).unsqueeze(-1).float()
        lengths = mask.sum(1)                                   # [B, 1]
        if pooling == "emb_mean":
            h = model.embedding(x)
        else:
            _, h = model(x)
        if pooling == "last":
            last = (lengths.squeeze(1).long() - 1).clamp(min=0)
            f = h[torch.arange(len(x), device=device), last]
        elif pooling in ("mean", "emb_mean"):
            f = (h * mask).sum(1) / lengths.clamp(min=1)
        else:
            raise ValueError(pooling)
        feats.append(f.float().cpu().numpy())
    return np.concatenate(feats)


def fit_and_evaluate(X: dict, y: dict, cfg: dict, seed: int) -> dict:
    """Standardize -> LogisticRegression, select C on val (macro-F1), refit on
    train, report accuracy and macro-F1 on test plus predictions."""
    c = cfg["classifier"]

    def make(C):
        steps = [StandardScaler()] if c["standardize"] else []
        return make_pipeline(*steps, LogisticRegression(C=C, max_iter=c["max_iter"],
                                                        random_state=seed))

    val_scores = {}
    for C in c["C_grid"]:
        clf = make(C).fit(X["train"], y["train"])
        val_scores[C] = f1_score(y["val"], clf.predict(X["val"]), average="macro")
    best_C = max(val_scores, key=val_scores.get)
    clf = make(best_C).fit(X["train"], y["train"])
    pred = clf.predict(X["test"])
    return {
        "best_C": best_C, "val_macro_f1_by_C": {str(k): v for k, v in val_scores.items()},
        "test_accuracy": accuracy_score(y["test"], pred),
        "test_macro_f1": f1_score(y["test"], pred, average="macro"),
        "test_f1_per_class": f1_score(y["test"], pred, average=None).tolist(),
        "confusion_matrix": confusion_matrix(y["test"], pred).tolist(),
        "pred": pred.tolist(),
    }
