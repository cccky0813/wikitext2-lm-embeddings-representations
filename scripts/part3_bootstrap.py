"""Part III uncertainty: bootstrap 95% CIs for macro-F1, paired bootstrap for key
differences, and Spearman rank correlation between LM perplexity and downstream F1.

    python scripts/part3_bootstrap.py

Writes results/metrics/part3_bootstrap.json.
"""
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.downstream import load_ag_news  # noqa: E402
from src.utils import ROOT, load_json, save_json  # noqa: E402

PAIRS = [("transformer_mean", "lstm_mean"), ("transformer_last", "lstm_last"),
         ("lstm_mean", "lstm_last"), ("transformer_mean", "transformer_last"),
         ("lstm_mean", "lstm_emb_mean"), ("rnn_mean", "lstm_mean"),
         ("lstm_w2v_mean", "lstm_mean"), ("lstm_glove_mean", "lstm_mean"),
         ("lstm_glove_mean", "lstm_w2v_mean"), ("lstm_w2v_mean", "transformer_mean")]


def main(n_boot: int = 1000, seed: int = 0):
    cfg = yaml.safe_load(open(ROOT / "configs" / "part3.yaml"))
    y = load_ag_news(cfg, cfg["seed"])["test"]["label"]
    res = load_json(ROOT / "results" / "metrics" / "part3.json")["results"]
    pred = {k: np.array(v["pred"]) for k, v in res.items()}
    idx = np.random.default_rng(seed).integers(0, len(y), (n_boot, len(y)))
    f1 = lambda p, i: f1_score(y[i], p[i], average="macro")  # noqa: E731

    out = {"n_boot": n_boot, "ci": {}, "paired": {}}
    for k in res:
        s = np.array([f1(pred[k], i) for i in idx])
        out["ci"][k] = [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]
    for a, b in PAIRS:
        d = np.array([f1(pred[a], i) - f1(pred[b], i) for i in idx])
        out["paired"][f"{a} - {b}"] = {
            "diff": res[a]["test_macro_f1"] - res[b]["test_macro_f1"],
            "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_le_0": float(np.mean(d <= 0))}
    mean_reps = [k for k in res if res[k]["pooling"] == "mean"]
    rho = spearmanr([res[k]["lm_test_ppl"] for k in mean_reps],
                    [-res[k]["test_macro_f1"] for k in mean_reps])
    out["spearman_ppl_vs_f1"] = {"reps": mean_reps, "rho": float(rho.statistic),
                                 "p": float(rho.pvalue)}
    save_json(out, ROOT / "results" / "metrics" / "part3_bootstrap.json")
    for k, v in out["paired"].items():
        print(f"{k:42s} {v['diff']:+.3f}  CI [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]")
    print("Spearman(PPL rank, F1 rank):", out["spearman_ppl_vs_f1"])


if __name__ == "__main__":
    main()
