"""Count-based n-gram LM with interpolated Kneser-Ney smoothing (Part I baseline).

Implemented from scratch over token ids so the smoothing is transparent and the
model is evaluated on exactly the same windows / vocab as the neural models.

Interpolated KN (Chen & Goodman, 1998), for an order-k context h:
    P_k(w|h) = max(c_k(h,w) - D, 0) / c_k(h,.) + D * N1+(h,.) / c_k(h,.) * P_{k-1}(w|h')
c_k are raw counts at the highest order used and continuation counts
N1+(.,h,w) at lower orders. Unseen contexts back off entirely. The unigram
level interpolates with a uniform distribution so no word gets zero mass.
"""
from collections import Counter, defaultdict

import numpy as np
import torch


class _Table:
    """For one order and one count type: context -> (word counts, total, #types)."""

    def __init__(self, counts: dict[tuple, Counter]):
        self.rows = {h: (c, sum(c.values()), len(c)) for h, c in counts.items()}

    def get(self, h: tuple):
        return self.rows.get(h)

    def n_entries(self) -> int:
        return sum(r[2] for r in self.rows.values())


class NGramLM:
    def __init__(self, n: int = 3, discount: float = 0.75):
        self.n = n
        self.D = discount

    def fit(self, ids: list[int], vocab_size: int) -> "NGramLM":
        self.V = vocab_size
        n = self.n
        raw = [defaultdict(Counter) for _ in range(n + 1)]   # raw[k][h][w], len(h)=k-1
        for k in range(1, n + 1):
            for i in range(k - 1, len(ids)):
                raw[k][tuple(ids[i - k + 1: i])][ids[i]] += 1
        # continuation counts for orders < n: N1+(., h, w) = #distinct left extensions
        cont = [defaultdict(Counter) for _ in range(n)]
        for k in range(1, n):
            for h_full, ws in raw[k + 1].items():
                h = h_full[1:]
                for w in ws:
                    cont[k][h][w] += 1
        self.raw = [None] + [_Table(raw[k]) for k in range(1, n + 1)]
        self.cont = [None] + [_Table(cont[k]) for k in range(1, n)]
        return self

    # ----- probabilities -------------------------------------------------
    def _table(self, k: int, highest: bool) -> _Table:
        return self.raw[k] if highest or k == self.n else self.cont[k]

    def prob(self, context: tuple[int, ...], w: int) -> float:
        """P(w | last n-1 tokens of context), interpolated KN."""
        h = tuple(context[-(self.n - 1):]) if self.n > 1 else ()
        return self._prob(h, w, highest=True)

    def _prob(self, h: tuple, w: int, highest: bool) -> float:
        k = len(h) + 1
        row = self._table(k, highest).get(h)
        if k == 1:
            c, total, types = row
            return max(c.get(w, 0) - self.D, 0) / total + self.D * types / total / self.V
        if row is None:                                     # unseen context: back off
            return self._prob(h[1:], w, highest)
        c, total, types = row
        lower = self._prob(h[1:], w, highest=False)
        return max(c.get(w, 0) - self.D, 0) / total + self.D * types / total * lower

    def distribution(self, context: tuple[int, ...]) -> np.ndarray:
        """Full next-token distribution (vectorised), used for sampling."""
        h = tuple(context[-(self.n - 1):]) if self.n > 1 else ()
        return self._dist(h, highest=True)

    def _dist(self, h: tuple, highest: bool) -> np.ndarray:
        k = len(h) + 1
        row = self._table(k, highest).get(h)
        if k == 1:
            c, total, types = row
            p = np.full(self.V, self.D * types / total / self.V)
        elif row is None:
            return self._dist(h[1:], highest)
        else:
            c, total, types = row
            p = self._dist(h[1:], highest=False) * (self.D * types / total)
        idx = np.fromiter(c.keys(), dtype=np.int64, count=len(c))
        val = np.fromiter(c.values(), dtype=np.float64, count=len(c))
        p[idx] += np.maximum(val - self.D, 0) / total
        return p

    # ----- evaluation / generation ---------------------------------------
    def evaluate(self, windows: torch.Tensor, pad_idx: int) -> dict:
        """Mean next-token cross-entropy (nats) and perplexity over the same
        (window, position) targets as the neural models: the context for
        position i is window[max(0, i-n+1) : i]."""
        nll, count = 0.0, 0
        for w in windows.tolist():
            for i in range(1, len(w)):
                if w[i] == pad_idx:
                    break
                nll -= np.log(self.prob(tuple(w[max(0, i - self.n + 1): i]), w[i]))
                count += 1
        ce = nll / count
        return {"cross_entropy": ce, "perplexity": float(np.exp(ce)), "num_tokens": count}

    def generate(self, prompt_ids: list[int], max_new_tokens: int, top_k: int,
                 temperature: float, seed: int, banned: list[int] = ()) -> list[int]:
        """Top-k sampling with the same settings as the neural models."""
        rng = np.random.default_rng(seed)
        ids = list(prompt_ids)
        for _ in range(max_new_tokens):
            logp = np.log(self.distribution(tuple(ids)) + 1e-30) / temperature
            logp[list(banned)] = -np.inf
            top = np.argpartition(-logp, top_k)[:top_k]
            p = np.exp(logp[top] - logp[top].max())
            ids.append(int(rng.choice(top, p=p / p.sum())))
        return ids[len(prompt_ids):]

    def size(self) -> dict:
        """Number of stored (context, word) entries per order — the model size."""
        out = {f"order{k}_raw": self.raw[k].n_entries() for k in range(1, self.n + 1)}
        out.update({f"order{k}_cont": self.cont[k].n_entries() for k in range(1, self.n)})
        out["total_entries"] = sum(out.values())
        return out
