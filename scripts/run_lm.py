"""Parts I & II: train one neural LM, then evaluate, benchmark and generate.

    python scripts/run_lm.py --config configs/part1_lstm.yaml
    python scripts/run_lm.py --config configs/part2_lstm_glove_frozen.yaml

Outputs (keyed by cfg['name']):
    checkpoints/<name>.pt              best-val weights
    results/metrics/<name>.json        test CE/PPL, params, train time, throughput
    results/logs/<name>_history.json   per-epoch train/val loss (for curves)
    results/samples/<name>.txt         continuations for configs/prompts.txt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.benchmark import measure_throughput  # noqa: E402
from src.data import load_wikitext2, make_lm_dataloader, tokenize  # noqa: E402
from src.embeddings import build_embedding_matrix, load_glove, load_word2vec  # noqa: E402
from src.generate import generate, load_prompts, write_samples  # noqa: E402
from src.models import build_model  # noqa: E402
from src.train import evaluate, train_lm  # noqa: E402
from src.utils import (ROOT, count_params, get_device, hardware_info,  # noqa: E402
                       load_config, save_json, set_seed)


def pretrained_matrix(cfg, vocab, ids):
    m = cfg["model"]
    if m["embedding"] == "random_trainable":
        return None, None
    if m["embedding"] == "word2vec_frozen":
        kv = load_word2vec(ROOT / m["embedding_path"])
    elif m["embedding"] == "glove_frozen":
        kv = load_glove(m["embedding_path"])
    else:
        raise ValueError(m["embedding"])
    return build_embedding_matrix(vocab, kv, m["emb_dim"], ids["train"],
                                  m.get("oov_init", "normal"), cfg["seed"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--epochs", type=int, default=None, help="override, e.g. 1 for a smoke test")
    ap.add_argument("--suffix", default="", help="appended to the run name, e.g. _smoke")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    cfg["name"] += args.suffix
    name, t, g = cfg["name"], cfg["train"], cfg["generate"]

    set_seed(cfg["seed"])
    device = get_device(t["device"])
    ids, vocab = load_wikitext2(cfg)
    loaders = {s: make_lm_dataloader(ids[s], t["seq_len"], t["batch_size"],
                                     shuffle=(s == "train"), pad_idx=vocab.pad_idx,
                                     seed=cfg["seed"])
               for s in ("train", "valid", "test")}

    pretrained, coverage = pretrained_matrix(cfg, vocab, ids)
    if coverage:
        print(f"embedding coverage: types {coverage['type_coverage']:.3f}, "
              f"tokens {coverage['token_coverage']:.3f}")
    set_seed(cfg["seed"])                       # same init stream for every embedding setting
    model = build_model(cfg, len(vocab), vocab.pad_idx, pretrained).to(device)
    params = count_params(model)
    print(f"{name}: {params['total']:,} params ({params['trainable']:,} trainable) on {device}")

    ckpt = ROOT / "checkpoints" / f"{name}.pt"
    hist = train_lm(model, loaders, cfg, device, vocab.pad_idx, str(ckpt))
    save_json(hist, ROOT / "results" / "logs" / f"{name}_history.json")

    test = evaluate(model, loaders["test"], device, vocab.pad_idx)
    val = evaluate(model, loaders["valid"], device, vocab.pad_idx)
    print(f"best val ppl {val['perplexity']:.2f} | test ppl {test['perplexity']:.2f}")

    b = cfg["benchmark"]
    throughput = measure_throughput(model, loaders["test"], device,
                                    b["num_batches"], b["warmup_batches"])

    samples = []
    for i, prompt in enumerate(load_prompts(ROOT / g["prompts_file"])):
        out = generate(model, vocab.encode(tokenize(prompt)), g["max_new_tokens"], g["top_k"],
                       g["temperature"], t["seq_len"], g["seed"] + i, device,
                       banned=[vocab.pad_idx])
        samples.append((prompt, vocab.decode(out)))
    gen_stats = write_samples(name, cfg, samples, ROOT / "results" / "samples" / f"{name}.txt")

    save_json({"name": name, "config": cfg, "hardware": hardware_info(), "params": params,
               "embedding_coverage": coverage, "train_seconds": hist["train_seconds_total"],
               "train_seconds_to_best": hist["train_seconds_to_best"],
               "best_epoch": hist["best_epoch"], "epochs_run": hist["epochs_run"],
               "valid": val, "test": test, "throughput": throughput,
               "generation": gen_stats}, ROOT / "results" / "metrics" / f"{name}.json")


if __name__ == "__main__":
    main()
