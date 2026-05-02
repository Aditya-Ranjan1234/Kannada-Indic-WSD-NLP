# Final Missing Execution Report (v2)

## Dataset Targets
- Pair count: 640
- Unique words: 45

## Main Metrics
| model   |   threshold |   validation_accuracy |   validation_macro_f1 |   test_accuracy |   test_macro_f1 |
|:--------|------------:|----------------------:|----------------------:|----------------:|----------------:|
| xlmr    |         0.1 |              0.46875  |              0.319149 |        0.489583 |        0.328671 |
| mbert   |         0.9 |              0.479167 |              0.460189 |        0.489583 |        0.47681  |
| muril   |         0.1 |              0.46875  |              0.319149 |        0.489583 |        0.328671 |

## Random Baselines
| baseline           |   accuracy |   macro_f1 |   ari |   mrr |
|:-------------------|-----------:|-----------:|------:|------:|
| uniform_random     |   0.5625   |   0.561739 |     0 |   0.5 |
| class_prior_random |   0.416667 |   0.416413 |     0 |   0.5 |

## Zero/Few-shot
| setting                   |   shots_per_class |   accuracy |   macro_f1 |
|:--------------------------|------------------:|-----------:|-----------:|
| zero_shot_fixed_threshold |                 0 |   0.489583 |   0.328671 |
| few_shot_logreg           |                 8 |   0.479167 |   0.467849 |
| few_shot_logreg           |                16 |   0.520833 |   0.52     |
| few_shot_logreg           |                32 |   0.53125  |   0.530792 |

## ARI
| model   |        ari |
|:--------|-----------:|
| mbert   | -0.0781646 |
| muril   | -0.0751302 |
| xlmr    | -0.027236  |

## MRR
| model   |      mrr |   n |
|:--------|---------:|----:|
| xlmr    | 0.755208 |  96 |
| mbert   | 0.807292 |  96 |
| muril   | 0.8125   |  96 |

## HF Upload
- Uploaded: False
- Repo: 
- Details: HF_DATASET_REPO not set or not in owner/repo format.