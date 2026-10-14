"""Collect results/metrics/*.json and results/logs/*_history.json into the
report's tables (results/metrics/summary_*.csv/.md) and figures (results/figures/).

    python scripts/make_figures.py
"""
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import ROOT, load_json  # noqa: E402

M, LOGS, FIG = ROOT / "results" / "metrics", ROOT / "results" / "logs", ROOT / "results" / "figures"

# One fixed colour per model entity (validated categorical palette, fixed order).
COLORS = {"part1_rnn": "#2a78d6", "part1_lstm": "#eb6834", "part1_transformer": "#1baf7a",
          "part1_ngram": "#eda100", "part2_lstm_w2v_frozen": "#e87ba4",
          "part2_lstm_glove_frozen": "#4a3aa7"}
LABELS = {"part1_ngram": "3-gram KN", "part1_rnn": "RNN", "part1_lstm": "LSTM",
          "part1_transformer": "Transformer", "part2_lstm_w2v_frozen": "LSTM + own Skip-gram (frozen)",
          "part2_lstm_glove_frozen": "LSTM + GloVe (frozen)"}
P2_LABELS = {"part1_lstm": "random, trainable", "part2_lstm_w2v_frozen": "own Skip-gram, frozen",
             "part2_lstm_glove_frozen": "GloVe, frozen"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
    "lines.linewidth": 2, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def have(name):
    return (M / f"{name}.json").exists()


def table(rows, name):
    df = pd.DataFrame(rows)
    df.to_csv(M / f"summary_{name}.csv", index=False)
    (M / f"summary_{name}.md").write_text(df.to_markdown(index=False, floatfmt=".2f"))
    print(f"\n== {name} ==\n{df.to_string(index=False)}")
    return df


def loss_curves(names, labels, fname, title):
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    for n in names:
        if not (LOGS / f"{n}_history.json").exists():
            continue
        h = load_json(LOGS / f"{n}_history.json")
        ep = range(1, len(h["train_loss"]) + 1)
        ax.plot(ep, h["train_loss"], color=COLORS[n], ls="--", lw=1.5, alpha=0.8)
        ax.plot(ep, h["val_loss"], color=COLORS[n], label=labels[n])
        b = h["best_epoch"]
        ax.plot(b, h["val_loss"][b - 1], "o", ms=6, color=COLORS[n], mec="white", mew=1.5)
    ax.plot([], [], color=MUTED, ls="--", lw=1.5, label="train (dashed)")
    ax.plot([], [], "o", color=MUTED, mec="white", label="best val epoch")
    ax.set(xlabel="epoch", ylabel="cross-entropy (nats/token)", title=title)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=8)
    fig.savefig(FIG / fname)
    plt.close(fig)


def part1():
    rows = []
    for n in ["part1_ngram", "part1_rnn", "part1_lstm", "part1_transformer"]:
        if not have(n):
            continue
        d = load_json(M / f"{n}.json")
        rows.append({
            "model": LABELS[n],
            "test_CE": d["test"]["cross_entropy"], "test_PPL": d["test"]["perplexity"],
            "val_PPL": d["valid"]["perplexity"],
            "params_M": d["params"]["total"] / 1e6 if "params" in d else None,
            "ngram_entries_M": d["size"]["total_entries"] / 1e6 if "size" in d else None,
            "train_s": d["train_seconds"], "best_epoch": d.get("best_epoch"),
            "infer_ktok_s": d["throughput"]["tokens_per_sec"] / 1e3,
            "distinct_2": d["generation"]["distinct_2"],
            "rep_trigram": d["generation"]["repeated_trigram_frac"],
            "unk_frac_gen": d["generation"]["unk_frac"],
        })
    if rows:
        table(rows, "part1")
    loss_curves(["part1_rnn", "part1_lstm", "part1_transformer"], LABELS,
                "part1_loss_curves.png", "Part I: training vs validation loss")


def part2():
    names = ["part1_lstm", "part2_lstm_w2v_frozen", "part2_lstm_glove_frozen"]
    rows = []
    w2v = load_json(M / "word2vec_cost.json") if have("word2vec_cost") else {}
    for n in names:
        if not have(n):
            continue
        d, h = load_json(M / f"{n}.json"), load_json(LOGS / f"{n}_history.json")
        cov = d.get("embedding_coverage") or {}
        rows.append({
            "embedding": P2_LABELS[n], "test_PPL": d["test"]["perplexity"],
            "best_epoch": d["best_epoch"], "final_gap": h["train_loss"][-1] - h["val_loss"][-1],
            "trainable_M": d["params"]["trainable"] / 1e6,
            "lm_train_s": d["train_seconds"],
            "emb_train_s": w2v.get("seconds") if "w2v" in n else (0 if n == "part1_lstm" else None),
            "grad_norm_mean": sum(h["grad_norm_mean"]) / len(h["grad_norm_mean"]),
            "grad_norm_max": max(h["grad_norm_max"]),
            "type_cov": cov.get("type_coverage"), "token_cov": cov.get("token_coverage"),
            "distinct_2": d["generation"]["distinct_2"],
        })
    if rows:
        table(rows, "part2")
    loss_curves(names, P2_LABELS, "part2_loss_curves.png", "Part II: LSTM by input embedding")

    fig, ax = plt.subplots(figsize=(5.2, 3.0))
    for n in names:
        if (LOGS / f"{n}_history.json").exists():
            h = load_json(LOGS / f"{n}_history.json")
            ep = range(1, len(h["grad_norm_mean"]) + 1)
            ax.plot(ep, h["grad_norm_mean"], color=COLORS[n], label=P2_LABELS[n])
    ax.set(xlabel="epoch", ylabel="mean pre-clip grad norm", title="Part II: training stability")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=8)
    fig.savefig(FIG / "part2_grad_norm.png")
    plt.close(fig)


def part3():
    if not have("part3"):
        return
    res = load_json(M / "part3.json")["results"]
    rows = [{"representation": k, "LM": LABELS.get(v["lm"], v["lm"]), "pooling": v["pooling"],
             "dim": v["feature_dim"], "test_acc": v["test_accuracy"],
             "test_macro_F1": v["test_macro_f1"], "best_C": v["best_C"],
             "LM_test_PPL": v["lm_test_ppl"]} for k, v in res.items()]
    df = table(rows, "part3")

    d = df.sort_values("test_macro_F1")
    fig, ax = plt.subplots(figsize=(5.2, 0.32 * len(d) + 0.8))
    colors = [COLORS[res[k]["lm"]] for k in d["representation"]]
    ax.barh(d["representation"], d["test_macro_F1"], color=colors, height=0.6)
    for y, v in enumerate(d["test_macro_F1"]):
        ax.text(v + 0.005, y, f"{v:.3f}", va="center", fontsize=8, color=INK)
    ax.set(xlabel="test macro-F1 (AG News, 4 classes)", xlim=(0, 1),
           title="Part III: downstream classification")
    ax.grid(axis="y", visible=False)
    fig.savefig(FIG / "part3_scores.png")
    plt.close(fig)

    # LM quality vs downstream quality, mean pooling only (one point per LM)
    m = df[df["pooling"] == "mean"]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    for _, r in m.iterrows():
        lm = res[r["representation"]]["lm"]
        ax.scatter(r["LM_test_PPL"], r["test_macro_F1"], s=60, color=COLORS[lm],
                   edgecolor="white", linewidth=1.5, zorder=3)
        ax.annotate(LABELS[lm], (r["LM_test_PPL"], r["test_macro_F1"]), fontsize=7.5,
                    color=INK, xytext=(5, 4), textcoords="offset points")
    ax.set(xlabel="LM test perplexity (lower = better)", ylabel="downstream macro-F1",
           title="Does better LM => better features? (mean pooling)")
    fig.savefig(FIG / "ppl_vs_downstream.png")
    plt.close(fig)


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    part1()
    part2()
    part3()
    print(f"\nfigures -> {FIG}")


if __name__ == "__main__":
    main()
