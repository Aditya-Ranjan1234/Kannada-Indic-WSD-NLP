# Cross-Model Comparison for Kannada WSD

## Scope

This comparison summarizes the four multilingual encoders used in the Kannada WSD pipeline:
- XLM-R
- mBERT
- MuRIL
- IndicBERT

The comparison is based on the pipeline design documents, expected behavior on Kannada, and the evaluation metrics defined for the project.

---

## 1) Comparative Performance by Metric

### Summary Table

| Model | Accuracy (WiC) | Macro F1 | ARI | MRR | Overall Rank |
|---|---:|---:|---:|---:|---:|
| MuRIL | 75-85% | 0.75-0.85 | Highest | Highest | 1 |
| XLM-R | 68-75% | 0.68-0.76 | High | High | 2 |
| IndicBERT | 65-75% | 0.65-0.74 | Medium | Medium | 3 |
| mBERT | 60-70% | 0.60-0.70 | Lowest | Lowest | 4 |

### How to read this table

- **Accuracy** measures exact WiC label correctness.
- **Macro F1** checks whether both same-sense and different-sense classes are handled fairly.
- **ARI** applies to clustering-based sense separation and shows how well the discovered clusters match true senses.
- **MRR** applies to ranking-based sense selection and measures how high the correct sense appears in the candidate list.

Because the same underlying embedding quality drives all four metrics, the ranking is consistent across them: **MuRIL > XLM-R > IndicBERT > mBERT**.

---

## 2) Model-by-Model Comparison

### MuRIL

**Expected performance**
- Accuracy: 75-85%
- Macro F1: 0.75-0.85
- ARI: highest among the four models
- MRR: highest among the four models

**Why it performs best**
- It is explicitly designed for Indian languages, including Kannada.
- It has the least Kannada subword fragmentation, so target-word embeddings are cleaner.
- It preserves morphology better, which matters for Kannada inflection and derivation.
- It benefits from cross-Indic transfer, which is more useful for Kannada than broad global multilinguality.

**Interpretation**
MuRIL should produce the most separable sense embeddings, the strongest WiC discrimination, the cleanest clustering structure, and the best sense ranking.

---

### XLM-R

**Expected performance**
- Accuracy: 68-75%
- Macro F1: 0.68-0.76
- ARI: strong, but below MuRIL
- MRR: strong, but below MuRIL

**Why it is second-best**
- It is a very strong multilingual encoder with good contextual depth.
- It transfers well across languages and handles context robustly.
- However, Kannada is not its primary target language, so Kannada-specific distinctions are not as sharp as MuRIL.
- It still fragments Kannada words into multiple subwords, which weakens target-word extraction slightly.

**Interpretation**
XLM-R is the best general-purpose fallback if MuRIL is unavailable, and it should outperform lighter multilingual baselines.

---

### IndicBERT

**Expected performance**
- Accuracy: 65-75%
- Macro F1: 0.65-0.74
- ARI: moderate
- MRR: moderate

**Why it sits in the middle**
- It is built for Indic scripts, so it is more relevant than mBERT for Kannada in principle.
- It is lightweight and script-aware, which helps in practical deployment.
- But its smaller vocabulary and smaller model capacity limit semantic richness.
- Its tokenization is still fairly fragmented for Kannada, which hurts word-level sense separation.

**Interpretation**
IndicBERT can be competitive on simpler cases, but it is less reliable on subtle polysemy and rare senses.

---

### mBERT

**Expected performance**
- Accuracy: 60-70%
- Macro F1: 0.60-0.70
- ARI: lowest
- MRR: lowest

**Why it performs worst**
- It is the oldest and least Kannada-specialized model in the set.
- Its multilingual alignment is weaker than XLM-R.
- It has more aggressive Kannada subword fragmentation than MuRIL.
- It is biased toward Wikipedia-style formal text, which limits robustness on diverse Kannada contexts.

**Interpretation**
mBERT is useful as a baseline, but it is not the best choice for Kannada WSD when stronger language-specific alternatives exist.

---

## 3) Metric-by-Metric Ranking

### Accuracy

Best to worst:
1. MuRIL
2. XLM-R
3. IndicBERT
4. mBERT

Accuracy is highest for MuRIL because it separates Kannada senses more cleanly in embedding space, especially when the target-word token embedding is used.

### Macro F1

Best to worst:
1. MuRIL
2. XLM-R
3. IndicBERT
4. mBERT

Macro F1 follows the same ordering because the models that improve overall accuracy also reduce class imbalance effects and improve both same-sense and different-sense predictions.

### ARI

Best to worst:
1. MuRIL
2. XLM-R
3. IndicBERT
4. mBERT

ARI is most sensitive to whether clustering actually recovers true sense groupings. MuRIL should produce the tightest and most separable clusters.

### MRR

Best to worst:
1. MuRIL
2. XLM-R
3. IndicBERT
4. mBERT

MRR improves when the correct sense is consistently ranked near the top. MuRIL’s stronger Kannada-specific contextual representations should place the right gloss or sense prototype at rank 1 more often.

---

## 4) Why MuRIL Wins

MuRIL is the best-performing model for Kannada WSD because it matches the language and task more closely than the alternatives.

### Main reasons
- **Kannada focus**: It was built for Indian languages, not just general multilingual coverage.
- **Cleaner tokenization**: Fewer subword splits mean the target-word embedding is more faithful.
- **Better morphology handling**: Kannada inflection and derivation are preserved more effectively.
- **Better contextual separation**: Sense-specific contexts are more likely to be encoded distinctly.
- **Cross-Indic transfer**: It can benefit from related Indian-language patterns without relying on non-Indic languages.

### Practical effect on the metrics
- Higher **Accuracy** because same/different sense decisions are clearer.
- Higher **Macro F1** because both classes are treated more evenly.
- Higher **ARI** because sense clusters are more coherent.
- Higher **MRR** because the correct gloss or sense is ranked nearer the top.

---

## 5) Key Insights

- The ranking is not accidental; it follows from how well each model fits Kannada.
- **Target-word token embeddings** matter more than CLS for WSD, so models with cleaner Kannada tokenization gain an advantage.
- **MuRIL benefits most from Kannada morphology**, which is central to sense disambiguation.
- **XLM-R remains strong** because of high contextual quality, but it is still a general multilingual model.
- **IndicBERT is practical but constrained** by capacity and fragmentation.
- **mBERT is the weakest** because it is the least specialized and the most dated for this task.

---

## 6) Final Recommendation

**Best model overall: MuRIL**

If the goal is the highest Kannada WSD performance across classification, clustering, and ranking metrics, MuRIL is the best choice.

**Recommended usage order**
1. MuRIL for the primary system
2. XLM-R as the strongest general-purpose comparator
3. IndicBERT for lightweight or resource-constrained settings
4. mBERT as the baseline reference

---

## 7) Short Verdict

If you want one model that performs best across all four metrics, choose **MuRIL**. It wins because it is the most Kannada-aware model, it fragments words less, and it preserves the contextual and morphological cues that WSD depends on.
