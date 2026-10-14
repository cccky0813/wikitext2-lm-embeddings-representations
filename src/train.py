"""Training / evaluation loop for neural LMs (Parts I & II)."""
import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .utils import timer


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, pad_idx: int) -> dict:
    """Token-level mean cross-entropy (nats, padding excluded) and perplexity."""
    model.eval()
    nll, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _ = model(x)
        nll += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                               ignore_index=pad_idx, reduction="sum").item()
        count += (y != pad_idx).sum().item()
    ce = nll / count
    return {"cross_entropy": ce, "perplexity": math.exp(ce), "num_tokens": count}


def train_lm(model: nn.Module, loaders: dict, cfg: dict, device: torch.device,
             pad_idx: int, ckpt_path: str) -> dict:
    """Train with Adam + grad clipping + early stopping on val loss.

    Returns a history dict: per-epoch train/val loss, gradient-norm stats,
    epoch wall times, total training time, best epoch. The best-val weights are
    saved to ckpt_path and loaded back into `model` before returning.
    """
    t = cfg["train"]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=t["lr"], weight_decay=t["weight_decay"])
    hist = {k: [] for k in ["train_loss", "val_loss", "val_ppl", "epoch_seconds",
                            "grad_norm_mean", "grad_norm_max", "grad_norm_std"]}
    best, best_state, bad = float("inf"), None, 0

    for epoch in range(1, t["epochs"] + 1):
        model.train()
        tot, cnt, norms = 0.0, 0, []
        with timer(device) as tm:
            for x, y in tqdm(loaders["train"], desc=f"epoch {epoch}", leave=False):
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1),
                                       ignore_index=pad_idx)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                norms.append(nn.utils.clip_grad_norm_(params, t["grad_clip"]).item())
                opt.step()
                n = (y != pad_idx).sum().item()
                tot += loss.item() * n
                cnt += n
        val = evaluate(model, loaders["valid"], device, pad_idx)
        hist["train_loss"].append(tot / cnt)
        hist["val_loss"].append(val["cross_entropy"])
        hist["val_ppl"].append(val["perplexity"])
        hist["epoch_seconds"].append(tm["seconds"])
        hist["grad_norm_mean"].append(float(np.mean(norms)))
        hist["grad_norm_max"].append(float(np.max(norms)))
        hist["grad_norm_std"].append(float(np.std(norms)))
        print(f"epoch {epoch:2d} | train {tot / cnt:.4f} | val {val['cross_entropy']:.4f} "
              f"(ppl {val['perplexity']:.1f}) | {tm['seconds']:.0f}s")

        if val["cross_entropy"] < best - 1e-4:
            best, bad = val["cross_entropy"], 0
            best_state = copy.deepcopy(model.state_dict())
            hist["best_epoch"] = epoch
            torch.save(best_state, ckpt_path)
        else:
            bad += 1
            if bad >= t["early_stopping_patience"]:
                print(f"early stop at epoch {epoch} (best {hist['best_epoch']})")
                break

    model.load_state_dict(best_state)
    hist["epochs_run"] = len(hist["train_loss"])
    hist["train_seconds_total"] = float(sum(hist["epoch_seconds"]))
    hist["train_seconds_to_best"] = float(sum(hist["epoch_seconds"][: hist["best_epoch"]]))
    return hist
