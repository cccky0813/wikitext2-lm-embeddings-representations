"""Part II: train own Skip-gram embeddings on the LM training split (separate cost).

    python scripts/train_word2vec.py --config configs/part2_lstm_w2v_frozen.yaml

Writes checkpoints/word2vec_100d.kv and results/metrics/word2vec_cost.json.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_wikitext2, split_sentences  # noqa: E402
from src.embeddings import train_word2vec  # noqa: E402
from src.utils import ROOT, hardware_info, load_config, save_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    ids, vocab = load_wikitext2(cfg)
    sents = split_sentences(ids["train"], vocab)       # train split only: no leakage
    out = ROOT / cfg["model"]["embedding_path"]
    stats = train_word2vec(sents, cfg, cfg["seed"], out)
    stats["hardware"] = hardware_info()
    save_json(stats, ROOT / "results" / "metrics" / "word2vec_cost.json")
    print(stats)

    from src.embeddings import load_word2vec
    kv = load_word2vec(out)
    for w in ["king", "france", "album", "war"]:
        if w in kv.key_to_index:
            print(w, "->", [x for x, _ in kv.most_similar(w, topn=5)])


if __name__ == "__main__":
    main()
