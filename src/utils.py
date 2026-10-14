"""Shared helpers: config loading, seeding, device selection, timing, I/O."""
import json
import platform
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path, base: str | Path = ROOT / "configs" / "base.yaml") -> dict:
    """Load an experiment config layered on top of configs/base.yaml."""
    with open(base) as f:
        cfg = yaml.safe_load(f)
    with open(path) as f:
        cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(pref: str = "auto") -> torch.device:
    if pref != "auto":
        return torch.device(pref)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    """Block until queued GPU work finishes, so wall-clock timings are honest."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


@contextmanager
def timer(device: torch.device | None = None):
    """Usage: `with timer(dev) as t: ...; t["seconds"]`."""
    out = {}
    if device is not None:
        sync(device)
    start = time.perf_counter()
    try:
        yield out
    finally:
        if device is not None:
            sync(device)
        out["seconds"] = time.perf_counter() - start


def count_params(model: torch.nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def hardware_info() -> dict:
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "device": str(get_device()),
    }


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)
