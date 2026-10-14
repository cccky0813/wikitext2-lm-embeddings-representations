| representation   | LM                            | pooling   |   dim |   test_acc |   test_macro_F1 |   best_C |   LM_test_PPL |
|:-----------------|:------------------------------|:----------|------:|-----------:|----------------:|---------:|--------------:|
| lstm_last        | LSTM                          | last      |   256 |       0.65 |            0.64 |     0.01 |        156.69 |
| lstm_mean        | LSTM                          | mean      |   256 |       0.70 |            0.70 |     0.10 |        156.69 |
| transformer_last | Transformer                   | last      |   256 |       0.72 |            0.72 |     0.01 |        142.96 |
| transformer_mean | Transformer                   | mean      |   256 |       0.77 |            0.77 |     0.10 |        142.96 |
| rnn_mean         | RNN                           | mean      |   256 |       0.73 |            0.73 |     0.01 |        176.45 |
| lstm_w2v_mean    | LSTM + own Skip-gram (frozen) | mean      |   256 |       0.79 |            0.79 |    10.00 |        141.05 |
| lstm_glove_mean  | LSTM + GloVe (frozen)         | mean      |   256 |       0.83 |            0.83 |     0.01 |        153.14 |
| lstm_emb_mean    | LSTM                          | emb_mean  |   100 |       0.67 |            0.67 |     0.10 |        156.69 |