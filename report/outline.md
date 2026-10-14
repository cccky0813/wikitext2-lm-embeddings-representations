# Report outline (≤ 6 pages, excl. appendix & references)

1. **Question & setup** (~0.5 p) — guiding question; WikiText-2 + preprocessing; hardware; seeds.
2. **Part I: model families** (~2 p)
   - Table: test CE / PPL, #params, train time, tokens/sec
   - Fig: train/val loss curves (RNN, LSTM, Transformer)
   - Generated samples (same prompts, top-k=40, T=0.8, 40 tokens) + distinct-n / repetition
   - Observations vs explanations (context, params, budget)
3. **Part II: embedding ablation (LSTM)** (~1.5 p)
   - Table: test PPL, best epoch, val-loss curve, stability (grad norms / variance)
   - GloVe coverage + OOV init; Word2Vec training cost (separate)
   - Interpret: trainability vs extra data in GloVe
4. **Part III: downstream** (~1.5 p)
   - Table: accuracy / macro-F1 for each representation
   - Does LM PPL ranking match downstream ranking?
   - Representative errors
5. **Limitations & lessons** (~0.5 p)

Appendix: full hyperparameters, extra samples, tool-use statement.
