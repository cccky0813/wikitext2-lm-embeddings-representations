"""Step 0: download + preprocess WikiText-2, build vocab, cache to data/processed/.

    python scripts/prepare_data.py
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_wikitext2  # noqa: E402
from src.utils import ROOT, load_config, save_json  # noqa: E402


def main():
    cfg = load_config(ROOT / "configs" / "part1_lstm.yaml")
    ids, vocab = load_wikitext2(cfg)
    stats = {"vocab_size": len(vocab)}
    for split, t in ids.items():
        stats[f"{split}_tokens"] = len(t)
        stats[f"{split}_unk_rate"] = round((t == vocab.unk_idx).float().mean().item(), 4)
    top = Counter(ids["train"].tolist()).most_common(10)
    stats["top10_train"] = [vocab.itos[i] for i, _ in top]
    save_json(stats, ROOT / "results" / "metrics" / "data_stats.json")
    for k, v in stats.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
