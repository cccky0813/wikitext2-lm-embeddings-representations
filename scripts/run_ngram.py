"""Part I: fit + evaluate the n-gram baseline.

    python scripts/run_ngram.py --config configs/part1_ngram.yaml

Writes results/metrics/<name>.json and results/samples/<name>.txt.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_wikitext2, make_windows, tokenize  # noqa: E402
from src.generate import load_prompts, write_samples  # noqa: E402
from src.ngram import NGramLM  # noqa: E402
from src.utils import ROOT, hardware_info, load_config, save_json, set_seed, timer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = load_config(ap.parse_args().config)
    set_seed(cfg["seed"])
    name, t, g, b = cfg["name"], cfg["train"], cfg["generate"], cfg["benchmark"]
    ids, vocab = load_wikitext2(cfg)

    model = NGramLM(cfg["model"]["n"], cfg["model"]["discount"])
    with timer() as fit_t:
        model.fit(ids["train"].tolist(), len(vocab))
    print(f"fit in {fit_t['seconds']:.1f}s", model.size())

    windows = {s: make_windows(ids[s], t["seq_len"], vocab.pad_idx) for s in ("valid", "test")}
    val = model.evaluate(windows["valid"], vocab.pad_idx)
    test = model.evaluate(windows["test"], vocab.pad_idx)
    print(f"val ppl {val['perplexity']:.2f} | test ppl {test['perplexity']:.2f}")

    # common inference workload: same first num_batches*batch_size test windows
    bench_w = windows["test"][: b["num_batches"] * t["batch_size"]]
    with timer() as bt:
        bench = model.evaluate(bench_w, vocab.pad_idx)
    throughput = {"seconds": bt["seconds"], "tokens_per_sec": bench["num_tokens"] / bt["seconds"],
                  "num_tokens": bench["num_tokens"], "note": "scalar KN prob per target, CPU"}

    samples = []
    for i, prompt in enumerate(load_prompts(ROOT / g["prompts_file"])):
        out = model.generate(vocab.encode(tokenize(prompt)), g["max_new_tokens"], g["top_k"],
                             g["temperature"], g["seed"] + i, banned=[vocab.pad_idx])
        samples.append((prompt, vocab.decode(out)))
    gen_stats = write_samples(name, cfg, samples, ROOT / "results" / "samples" / f"{name}.txt")

    save_json({"name": name, "config": cfg, "hardware": hardware_info(),
               "size": model.size(), "train_seconds": fit_t["seconds"],
               "valid": val, "test": test, "throughput": throughput,
               "generation": gen_stats}, ROOT / "results" / "metrics" / f"{name}.json")


if __name__ == "__main__":
    main()
