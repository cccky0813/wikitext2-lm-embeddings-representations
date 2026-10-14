"""Word embeddings for Part II: own Word2Vec and public GloVe -> embedding matrix."""
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .data import Vocab
from .utils import timer


def train_word2vec(sentences: list[list[str]], cfg: dict, seed: int,
                   out_path: str | Path) -> dict:
    """Train Word2Vec on the LM *training* split only; save KeyedVectors.

    Returns cost stats (wall time, epochs, corpus size) reported separately
    from LM training, as the assignment requires.
    """
    from gensim.models import Word2Vec
    w = cfg["word2vec"]
    with timer() as tm:
        model = Word2Vec(sentences, sg=1 if w["method"] == "skipgram" else 0,
                         vector_size=w["vector_size"], window=w["window"],
                         min_count=w["min_count"], negative=w["negative"],
                         epochs=w["epochs"], workers=w["workers"], seed=seed)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    model.wv.save(str(out_path))
    return {"method": w["method"], "seconds": tm["seconds"], "epochs": w["epochs"],
            "workers": w["workers"], "corpus_tokens": sum(map(len, sentences)),
            "num_sentences": len(sentences), "vocab_size": len(model.wv),
            "vector_size": w["vector_size"]}


def load_word2vec(path: str | Path):
    from gensim.models import KeyedVectors
    return KeyedVectors.load(str(path))


def load_glove(name: str = "glove-wiki-gigaword-100"):
    """Load public GloVe vectors via gensim.downloader (cached in ~/gensim-data)."""
    import gensim.downloader
    return gensim.downloader.load(name)


def build_embedding_matrix(vocab: Vocab, keyed_vectors, emb_dim: int,
                           train_ids: torch.Tensor, oov_init: str = "normal",
                           seed: int = 42) -> tuple[torch.Tensor, dict]:
    """Rows for in-vocab words copied from keyed_vectors; OOV rows drawn from
    N(0, sigma) with sigma = std of the pretrained vectors (or zeros);
    <pad> row = zeros.

    Returns (matrix[V, emb_dim], coverage stats: type coverage, token coverage
    on the training split, most frequent OOV words).
    """
    assert keyed_vectors.vector_size == emb_dim, (keyed_vectors.vector_size, emb_dim)
    rng = np.random.default_rng(seed)
    sigma = float(np.std(keyed_vectors.vectors))
    if oov_init == "normal":
        mat = rng.normal(0.0, sigma, size=(len(vocab), emb_dim))
    elif oov_init == "zeros":
        mat = np.zeros((len(vocab), emb_dim))
    else:
        raise ValueError(oov_init)

    found = np.zeros(len(vocab), dtype=bool)
    for i, w in enumerate(vocab.itos):
        if w in keyed_vectors.key_to_index:
            mat[i] = keyed_vectors[w]
            found[i] = True
    mat[vocab.pad_idx] = 0.0
    found[vocab.pad_idx] = True          # pad is never a target; not a real OOV

    freq = Counter(train_ids.tolist())
    tok_total = sum(freq.values())
    tok_found = sum(c for i, c in freq.items() if found[i])
    oov = sorted((i for i in range(len(vocab)) if not found[i]), key=lambda i: -freq.get(i, 0))
    stats = {
        "type_coverage": float(found[1:].mean()),
        "token_coverage": tok_found / tok_total,
        "num_oov_types": len(oov),
        "top_oov": [(vocab.itos[i], freq.get(i, 0)) for i in oov[:20]],
        "oov_init": oov_init, "oov_sigma": sigma,
    }
    return torch.tensor(mat, dtype=torch.float32), stats
