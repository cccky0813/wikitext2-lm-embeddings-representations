"""Neural next-token LMs (Part I) with a pluggable input-embedding table (Part II).

All models share the interface:
    forward(x: LongTensor[B, T]) -> (logits[B, T, V], hidden_states[B, T, H])
`hidden_states` is the per-token top-layer representation reused in Part III.

Padding is always on the right. Recurrent models and causal attention only
look left, so states at non-pad positions never depend on padding and no
attention key mask is needed.
"""
import torch
import torch.nn as nn


def build_embedding(vocab_size: int, emb_dim: int, pad_idx: int,
                    pretrained: torch.Tensor | None = None,
                    freeze: bool = False) -> nn.Embedding:
    """Random-init trainable table, or a pretrained matrix (frozen if requested).

    When frozen, the *entire* table (incl. OOV rows) has requires_grad=False.
    """
    emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
    if pretrained is not None:
        assert pretrained.shape == (vocab_size, emb_dim), pretrained.shape
        with torch.no_grad():
            emb.weight.copy_(pretrained)
    emb.weight.requires_grad = not freeze
    return emb


class _RecurrentLM(nn.Module):
    cell: type[nn.RNNBase]

    def __init__(self, embedding: nn.Embedding, hidden_dim: int, num_layers: int,
                 dropout: float, **cell_kwargs):
        super().__init__()
        self.embedding = embedding
        self.drop = nn.Dropout(dropout)
        self.rnn = self.cell(embedding.embedding_dim, hidden_dim, num_layers,
                             batch_first=True,
                             dropout=dropout if num_layers > 1 else 0.0, **cell_kwargs)
        self.out = nn.Linear(hidden_dim, embedding.num_embeddings)
        self.hidden_dim = hidden_dim

    def forward(self, x: torch.Tensor):
        h, _ = self.rnn(self.drop(self.embedding(x)))
        return self.out(self.drop(h)), h


class RNNLM(_RecurrentLM):
    """Vanilla (tanh) Elman RNN LM."""
    cell = nn.RNN


class LSTMLM(_RecurrentLM):
    """LSTM LM. Also the architecture used for the Part II embedding ablation."""
    cell = nn.LSTM


class TransformerLM(nn.Module):
    """Decoder-only (pre-LN) Transformer with causal self-attention and learned
    positions. emb_dim -> d_model via a linear projection so the embedding table
    stays comparable to the RNN/LSTM ones."""

    def __init__(self, embedding: nn.Embedding, d_model: int, num_layers: int,
                 num_heads: int, ff_dim: int, dropout: float, max_len: int):
        super().__init__()
        self.embedding = embedding
        self.proj = nn.Linear(embedding.embedding_dim, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, num_heads, ff_dim, dropout,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, embedding.num_embeddings)
        self.hidden_dim = d_model
        self.max_len = max_len

    def forward(self, x: torch.Tensor):
        T = x.size(1)
        assert T <= self.max_len, f"sequence length {T} > max_len {self.max_len}"
        pos = torch.arange(T, device=x.device)
        h = self.drop(self.proj(self.embedding(x)) + self.pos(pos))
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.norm(self.blocks(h, mask=mask, is_causal=True))
        return self.out(self.drop(h)), h


def build_model(cfg: dict, vocab_size: int, pad_idx: int,
                pretrained: torch.Tensor | None = None) -> nn.Module:
    """Factory: cfg['model']['type'] in {rnn, lstm, transformer}."""
    m = cfg["model"]
    freeze = m["embedding"].endswith("_frozen")
    if freeze and pretrained is None:
        raise ValueError(f"embedding={m['embedding']} needs a pretrained matrix")
    emb = build_embedding(vocab_size, m["emb_dim"], pad_idx, pretrained, freeze)
    if m["type"] == "rnn":
        return RNNLM(emb, m["hidden_dim"], m["num_layers"], m["dropout"], nonlinearity="tanh")
    if m["type"] == "lstm":
        return LSTMLM(emb, m["hidden_dim"], m["num_layers"], m["dropout"])
    if m["type"] == "transformer":
        return TransformerLM(emb, m["d_model"], m["num_layers"], m["num_heads"],
                             m["ff_dim"], m["dropout"], max_len=cfg["train"]["seq_len"])
    raise ValueError(m["type"])
