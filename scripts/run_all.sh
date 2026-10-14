#!/usr/bin/env bash
# Reproduce every reported result end to end. Run from the project root:
#   conda activate ais5102 && bash scripts/run_all.sh
set -euo pipefail
export PYTHONHASHSEED=0          # gensim Word2Vec determinism

python scripts/prepare_data.py

# Part I
python scripts/run_ngram.py --config configs/part1_ngram.yaml
python scripts/run_lm.py    --config configs/part1_rnn.yaml
python scripts/run_lm.py    --config configs/part1_lstm.yaml        # = Part II setting (1)
python scripts/run_lm.py    --config configs/part1_transformer.yaml

# Part II
python scripts/train_word2vec.py --config configs/part2_lstm_w2v_frozen.yaml
python scripts/run_lm.py    --config configs/part2_lstm_w2v_frozen.yaml
python scripts/run_lm.py    --config configs/part2_lstm_glove_frozen.yaml

# Part III
python scripts/run_downstream.py --config configs/part3.yaml
python scripts/part3_bootstrap.py

python scripts/make_figures.py
