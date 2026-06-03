# Evaluation Metrics for Kannada WSD

## Overview

This document defines the main metrics used to evaluate WiC-style word sense disambiguation and clustering-based sense separation.

The four metrics are:
- Accuracy
- Macro F1
- ARI
- MRR

Each metric answers a different question, so they should be reported together rather than in isolation.

---

## 1) Accuracy

### What it measures
Accuracy is the fraction of predictions that are exactly correct.

For WiC, this means the model predicts the correct binary label for each sentence pair:
- `1` = same sense
- `0` = different sense

Formula:

$$
Accuracy = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}}
$$

### Why it is relevant
- It is the simplest overall measure of classification performance.
- It is easy to interpret and compare across models.
- For balanced WiC datasets, it gives a good first summary of performance.

### Important note
Accuracy can be misleading if the dataset is imbalanced. If most examples belong to one class, a model can get high accuracy by predicting only the majority class. That is why macro F1 is also needed.

---

## 2) Macro F1

### What it measures
Macro F1 is the average of the F1 score computed independently for each class.

For binary WiC, compute F1 for:
- class `1` (same sense)
- class `0` (different sense)

Then average the two values equally.

Formula:

$$
F1 = \frac{2 \cdot Precision \cdot Recall}{Precision + Recall}
$$

$$
Macro\ F1 = \frac{F1_{class\ 0} + F1_{class\ 1}}{2}
$$

### Why it is relevant
- It treats both classes equally, regardless of class frequency.
- It is better than accuracy when labels are unbalanced.
- It reflects whether the model performs well on both same-sense and different-sense pairs.

### Important note
Macro F1 is especially useful if you care about avoiding a model that performs well only on one class. It is usually more informative than accuracy for real evaluation.

---

## 3) ARI

### What it measures
ARI stands for Adjusted Rand Index. It measures how well a clustering matches the true grouping, while correcting for agreement that could happen by chance.

ARI is used when evaluating unsupervised clustering of embeddings into sense groups.

Formula intuition:
- Compare all pairs of items.
- Check whether each pair is placed together or separately in both the predicted clustering and the true sense labels.
- Adjust the score so random agreement is not rewarded.

ARI range:
- `1.0` = perfect match
- `0.0` = random labeling expectation
- negative values = worse than random agreement

### Why it is relevant
- It is useful for clustering-based WSD, where embeddings are grouped into sense clusters.
- It evaluates whether the discovered clusters correspond to actual senses.
- It is more principled than raw cluster purity because it corrects for chance.

### Important note
ARI is not a classification metric. It is only appropriate when the model outputs clusters, not direct class labels.

---

## 4) MRR

### What it measures
MRR stands for Mean Reciprocal Rank. It evaluates ranking quality.

For each query, find the rank of the correct answer. The reciprocal rank is:
- `1 / rank` if the correct answer appears at position `rank`

Then average across all queries.

Formula:

$$
MRR = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{rank_i}
$$

where `rank_i` is the position of the first correct result for query `i`.

### Why it is relevant
- It is useful when the system produces ranked candidate senses or glosses.
- It measures how high the correct sense appears in the ranking.
- It is especially relevant for gloss-based WSD baselines, where each sense is ranked by similarity.

### Important note
MRR is most useful when the system does not just predict one label, but produces an ordered list of candidate senses. It rewards placing the correct sense near the top.

---

## When to Use Which Metric

- Use **Accuracy** for a quick overall WiC result.
- Use **Macro F1** to ensure both classes are handled fairly.
- Use **ARI** to evaluate unsupervised sense clustering.
- Use **MRR** to evaluate ranked sense retrieval or gloss-based sense selection.

---

## Recommended Reporting Setup

For the Kannada WSD project, a good evaluation report should include:
- **Accuracy** and **Macro F1** for WiC classification
- **ARI** for clustering-based sense separation
- **MRR** for gloss-ranking or candidate-sense ranking

This gives a complete view of both classification quality and sense-discovery quality.
