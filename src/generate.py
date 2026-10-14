"""Text generation with matched decoding settings across all models."""
import torch
import torch.nn as nn


@torch.no_grad()
def generate(model: nn.Module, prompt_ids: list[int], max_new_tokens: int,
             top_k: int, temperature: float, seq_len: int, seed: int,
             device: torch.device, banned: list[int] = ()) -> list[int]:
    """Autoregressive top-k sampling. The context is truncated to the last
    seq_len tokens so every model sees at most its training context length.
    Sampling runs on CPU with a seeded generator for reproducibility."""
    model.eval()
    g = torch.Generator().manual_seed(seed)
    ids = list(prompt_ids)
    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-seq_len:]], device=device)
        logits = model(x)[0][0, -1].float().cpu() / temperature
        logits[list(banned)] = -float("inf")
        vals, idx = logits.topk(top_k)
        nxt = idx[torch.multinomial(vals.softmax(-1), 1, generator=g)].item()
        ids.append(nxt)
    return ids[len(prompt_ids):]


def load_prompts(path) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def write_samples(name: str, cfg: dict, samples: list[tuple[str, list[str]]], path) -> dict:
    """Write prompt -> continuation pairs and return aggregated repetition stats."""
    g = cfg["generate"]
    stats = [repetition_stats(toks) for _, toks in samples]
    with open(path, "w") as f:
        f.write(f"# {name} | top_k={g['top_k']} T={g['temperature']} "
                f"max_new_tokens={g['max_new_tokens']} seed={g['seed']}\n\n")
        for prompt, toks in samples:
            f.write(f"[{prompt}] {' '.join(toks)}\n\n")
    return {k: sum(s[k] for s in stats) / len(stats) for k in stats[0]}


def repetition_stats(tokens: list[str]) -> dict:
    """Simple quantitative proxies for the qualitative comparison:
    distinct-1/2 ratios and fraction of repeated trigrams."""
    def ngrams(n):
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    uni, bi, tri = ngrams(1), ngrams(2), ngrams(3)
    return {
        "distinct_1": len(set(uni)) / max(len(uni), 1),
        "distinct_2": len(set(bi)) / max(len(bi), 1),
        "repeated_trigram_frac": 1 - len(set(tri)) / max(len(tri), 1),
        "unk_frac": tokens.count("<unk>") / max(len(tokens), 1),
    }
