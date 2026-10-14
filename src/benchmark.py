"""Inference cost on a common workload (Part I cost comparison)."""
import itertools

import torch
import torch.nn as nn

from .utils import timer


@torch.no_grad()
def measure_throughput(model: nn.Module, loader, device: torch.device,
                       num_batches: int, warmup_batches: int) -> dict:
    """Forward-only timing over the same first `num_batches` (unshuffled) test
    batches for every model, after warm-up, with device synchronisation around
    the timed region. Returns seconds, tokens/sec and ms/batch."""
    model.eval()
    batches = [(x.to(device), y) for x, y in itertools.islice(loader, num_batches)]
    for x, _ in batches[:warmup_batches]:
        model(x)
    n_tokens = sum(x.numel() for x, _ in batches)
    with timer(device) as tm:
        for x, _ in batches:
            model(x)
    return {"seconds": tm["seconds"], "tokens_per_sec": n_tokens / tm["seconds"],
            "ms_per_batch": 1000 * tm["seconds"] / len(batches),
            "num_batches": len(batches), "num_tokens": n_tokens}
