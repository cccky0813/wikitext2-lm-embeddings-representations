"""Part III: extract frozen-LM representations and train classifiers.

    python scripts/run_downstream.py --config configs/part3.yaml

Requires the LM checkpoints from Parts I-II. Writes results/metrics/part3.json
and results/samples/part3_errors.md.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_wikitext2  # noqa: E402
from src.downstream import (AG_LABELS, encode_texts, extract_features,  # noqa: E402
                            fit_and_evaluate, load_ag_news)
from src.models import build_model  # noqa: E402
from src.utils import (ROOT, get_device, hardware_info, load_config,  # noqa: E402
                       load_json, save_json, set_seed, timer)


def load_lm(lm_cfg: dict, vocab, device):
    """Rebuild an LM from its run config and load its best checkpoint."""
    placeholder = None
    if lm_cfg["model"]["embedding"].endswith("_frozen"):     # weights come from the ckpt
        placeholder = torch.zeros(len(vocab), lm_cfg["model"]["emb_dim"])
    model = build_model(lm_cfg, len(vocab), vocab.pad_idx, placeholder)
    state = torch.load(ROOT / "checkpoints" / f"{lm_cfg['name']}.pt", map_location="cpu")
    model.load_state_dict(state)
    return model.to(device).eval()


def write_errors(path, data, results, names, k):
    """Representative errors: (a) misclassified by every representation, and
    (b) for the best vs worst contextual representation, cases where they disagree."""
    y = data["test"]["label"]
    preds = {n: np.array(results[n]["pred"]) for n in names}
    lines = ["# Part III error analysis\n"]
    all_wrong = np.where(np.all([preds[n] != y for n in names], axis=0))[0]
    lines.append(f"## Misclassified by all {len(names)} representations "
                 f"({len(all_wrong)} / {len(y)})\n")
    for i in all_wrong[:k]:
        votes = np.bincount([preds[n][i] for n in names], minlength=4)
        lines.append(f"- **true {AG_LABELS[y[i]]}**, majority pred {AG_LABELS[votes.argmax()]}: "
                     f"{data['test']['text'][i][:300]}\n")
    ranked = sorted(names, key=lambda n: -results[n]["test_macro_f1"])
    best, worst = ranked[0], ranked[-1]
    for a, b in [(best, worst), (worst, best)]:
        idx = np.where((preds[a] == y) & (preds[b] != y))[0]
        lines.append(f"\n## Correct with `{a}`, wrong with `{b}` ({len(idx)})\n")
        for i in idx[:k // 2]:
            lines.append(f"- **true {AG_LABELS[y[i]]}**, `{b}` said {AG_LABELS[preds[b][i]]}: "
                         f"{data['test']['text'][i][:300]}\n")
    Path(path).write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--lm-suffix", default="", help="e.g. _smoke to use smoke-test LMs")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    device = get_device()
    t = cfg["task"]

    lm_base = load_config(ROOT / "configs" / "part1_lstm.yaml")
    _, vocab = load_wikitext2(lm_base)                 # same vocab for every LM
    data = load_ag_news(cfg, cfg["seed"])
    enc, tok_stats = {}, {}
    for s in data:
        enc[s], tok_stats[s] = encode_texts(vocab, data[s]["text"], t["max_len"])
    print("tokenization:", tok_stats["test"])
    y = {s: data[s]["label"] for s in data}

    results, lms = {}, {}
    for rep in cfg["representations"]:
        lm_cfg = load_config(ROOT / rep["lm_config"])
        lm_cfg["name"] += args.lm_suffix
        if lm_cfg["name"] not in lms:
            lms[lm_cfg["name"]] = load_lm(lm_cfg, vocab, device)
        model = lms[lm_cfg["name"]]
        with timer(device) as tm:
            X = {s: extract_features(model, enc[s], vocab.pad_idx, rep["pooling"],
                                     t["batch_size"], device) for s in enc}
        r = fit_and_evaluate(X, y, cfg, cfg["seed"])
        lm_metrics = load_json(ROOT / "results" / "metrics" / f"{lm_cfg['name']}.json")
        r.update({"lm": lm_cfg["name"], "pooling": rep["pooling"], "feature_dim": X["train"].shape[1],
                  "extract_seconds": tm["seconds"], "lm_test_ppl": lm_metrics["test"]["perplexity"]})
        results[rep["name"]] = r
        print(f"{rep['name']:18s} acc {r['test_accuracy']:.4f}  macro-F1 {r['test_macro_f1']:.4f}  "
              f"(C={r['best_C']}, LM ppl {r['lm_test_ppl']:.1f})")

    write_errors(ROOT / "results" / "samples" / f"part3_errors{args.lm_suffix}.md", data, results,
                 [r["name"] for r in cfg["representations"] if r["pooling"] != "emb_mean"],
                 cfg["num_error_examples"])
    save_json({"config": cfg, "hardware": hardware_info(), "tokenization": tok_stats,
               "results": results}, ROOT / "results" / "metrics" / f"part3{args.lm_suffix}.json")


if __name__ == "__main__":
    main()
