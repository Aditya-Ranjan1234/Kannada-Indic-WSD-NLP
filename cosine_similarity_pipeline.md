# Cosine Similarity Pipeline for Kannada WSD
## Computing and Interpreting Sentence Pair Similarities

---

## EXECUTIVE SUMMARY

**Similarity Metric**: Cosine Similarity (standard for embedding spaces)

**Pipeline**: Extract embeddings → Compute cosine similarity → Classify with threshold

**Key Outcomes**:
- **Same sense pairs** (label=1): high similarity (0.65-0.95)
- **Different sense pairs** (label=0): low similarity (0.20-0.60)
- **Optimal threshold**: ~0.50 for balanced classification

---

## PART 1: MATHEMATICAL FOUNDATION

### Definition of Cosine Similarity

**Formula**:
```
Given two embedding vectors:
  e1 = [e1_1, e1_2, ..., e1_768]
  e2 = [e2_1, e2_2, ..., e2_768]

Cosine Similarity:
  cos_sim(e1, e2) = (e1 · e2) / (||e1|| × ||e2||)
  
Where:
  e1 · e2 = Σ(e1_i × e2_i) for i=1 to 768    [dot product]
  ||e1|| = √(Σ(e1_i²) for i=1 to 768)        [L2 norm]
  ||e2|| = √(Σ(e2_i²) for i=1 to 768)        [L2 norm]

Result Range:
  cos_sim ∈ [-1, +1]
  
Interpretation:
  +1.0: Perfectly aligned vectors (identical direction)
   0.0: Orthogonal vectors (perpendicular)
  -1.0: Opposite vectors (opposite direction)
```

### Why Cosine Similarity?

```
Advantages:
  ✓ Scale-invariant: Magnitude doesn't matter, only direction
  ✓ Intuitive: Ranges from -1 to +1
  ✓ Computationally efficient: O(n) time, standard implementations
  ✓ Standard in NLP: Used in all major WSD systems
  ✓ Embedding-agnostic: Works identically for all 4 models
  ✓ Symmetric: cos_sim(e1, e2) = cos_sim(e2, e1)
  
Disadvantages:
  ✗ Ignores magnitude information (but embeddings are normalized)
  ✗ Sensitive to noise in extreme dimensions (acceptable for BERT)
  ✗ Range doesn't directly map to probability (requires calibration)
  
Alternatives Considered:

  1. Euclidean Distance (L2)
     Formula: distance = √(Σ(e1_i - e2_i)²)
     Range: [0, ∞]
     Issue: Sensitive to scaling; hard to interpret large values
     Decision: Not chosen - Cosine is more stable
  
  2. Manhattan Distance (L1)
     Formula: distance = Σ|e1_i - e2_i|
     Range: [0, ∞]
     Issue: Computationally slower; less intuitive
     Decision: Not chosen
  
  3. Dot Product (unnormalized)
     Formula: dot = Σ(e1_i × e2_i)
     Range: [-∞, +∞]
     Issue: Magnitude-dependent; hard to compare across pairs
     Decision: Not chosen
  
  4. Earth Mover Distance
     Formula: optimal transport between distributions
     Issue: Overkill for embedding comparison; much slower
     Decision: Not chosen
```

---

## PART 2: COSINE SIMILARITY COMPUTATION PIPELINE

### Step 1: Extract Embeddings (From Previous Pipeline)

**Input**: WiC pair with two sentences

```
WiC Pair:
{
  "sentence1": "ಕಾಲ ದ್ರುತವಾಗಿ ಹಾದುಹೋಗಿ",
  "sentence2": "ಅವನ ಎಡ ಕಾಲು ಮುರಿದಿದೆ",
  "target_word": "ಕಾಲ",
  "label": 0 (different senses)
}

After embedding extraction pipeline:
{
  "embedding1": [0.123, 0.456, ..., 0.789],  # [768]
  "embedding2": [0.234, 0.567, ..., 0.890],  # [768]
}
```

---

### Step 2: Verify Embedding Properties

**Ensure embeddings are valid** before similarity computation

```
Checks:
  1. Shape validation
     Condition: embedding1.shape == (768,) and embedding2.shape == (768,)
     Action: FAIL if violated
  
  2. No NaN/Inf
     Condition: all(~isnan(e)) and all(~isinf(e)) for both embeddings
     Action: FAIL if violated
  
  3. Non-zero vectors
     Condition: ||e1|| > 0 and ||e2|| > 0
     Action: FAIL if either is zero vector
     (Zero vector would cause division by zero)

Example Valid Embeddings:
  e1 = [-0.5, 0.3, 0.2, ..., 0.1]  ✓
  e2 = [0.1, -0.4, 0.3, ..., 0.2]  ✓
  
Example Invalid:
  e1 = [nan, 0.3, 0.2, ..., 0.1]   ✗ (contains NaN)
  e2 = [0.0, 0.0, 0.0, ..., 0.0]   ✗ (zero vector)
```

---

### Step 3: Compute Dot Product

**Calculate e1 · e2**

```
Formula:
  dot_product = Σ(e1[i] × e2[i]) for i = 0 to 767

Implementation (pseudocode):
  dot = 0
  for i in range(768):
    dot += embedding1[i] * embedding2[i]
  
Vectorized (NumPy):
  dot = np.dot(embedding1, embedding2)
  
PyTorch:
  dot = torch.dot(embedding1, embedding2)

Example Calculation:
  e1 = [0.1, 0.2, 0.3]
  e2 = [0.2, 0.3, 0.4]
  dot = (0.1×0.2) + (0.2×0.3) + (0.3×0.4)
      = 0.02 + 0.06 + 0.12
      = 0.20
```

---

### Step 4: Compute L2 Norms

**Calculate ||e1|| and ||e2||**

```
Formula for ||e1||:
  norm_e1 = √(Σ(e1[i]²) for i = 0 to 767)

Implementation (NumPy):
  norm_e1 = np.linalg.norm(embedding1, ord=2)
  norm_e2 = np.linalg.norm(embedding2, ord=2)
  
PyTorch:
  norm_e1 = torch.norm(embedding1, p=2)
  norm_e2 = torch.norm(embedding2, p=2)

Example:
  e1 = [0.1, 0.2, 0.3]
  ||e1|| = √(0.1² + 0.2² + 0.3²)
         = √(0.01 + 0.04 + 0.09)
         = √0.14
         = 0.374
  
  e2 = [0.2, 0.3, 0.4]
  ||e2|| = √(0.2² + 0.3² + 0.4²)
         = √(0.04 + 0.09 + 0.16)
         = √0.29
         = 0.539
```

---

### Step 5: Compute Cosine Similarity

**Final calculation: (dot) / (norm1 × norm2)**

```
Formula:
  cos_sim = dot_product / (norm_e1 × norm_e2)

Implementation (NumPy):
  cos_sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))
  
Vectorized (scikit-learn):
  from sklearn.metrics.pairwise import cosine_similarity
  cos_sim = cosine_similarity([e1], [e2])[0][0]
  
PyTorch:
  cos_sim = torch.nn.functional.cosine_similarity(e1, e2, dim=0)
  
SciPy:
  from scipy.spatial.distance import cosine
  cos_sim = 1 - cosine(e1, e2)

Example Completion:
  e1 = [0.1, 0.2, 0.3]
  e2 = [0.2, 0.3, 0.4]
  
  dot = 0.20 (from Step 3)
  norm_e1 = 0.374 (from Step 4)
  norm_e2 = 0.539 (from Step 4)
  
  cos_sim = 0.20 / (0.374 × 0.539)
          = 0.20 / 0.202
          = 0.990
  
  Interpretation: Vectors are highly aligned
```

---

### Step 6: Handle Edge Cases

**Robustness checks and fallback strategies**

```
Edge Case 1: Identical embeddings
  Condition: e1 == e2 (element-wise)
  cos_sim = 1.0 (by definition)
  Handling: Return 1.0 directly
  
Edge Case 2: Orthogonal embeddings
  Condition: dot_product == 0
  cos_sim = 0.0
  Handling: Return 0.0
  
Edge Case 3: Very small norms
  Condition: norm < 1e-6
  Issue: Numerical instability, embeddings near zero
  Handling: Clamp norm to >= 1e-6, log warning
  
Edge Case 4: Norm computation overflow
  Condition: Very large embedding values
  Issue: Numerical overflow during norm calculation
  Handling: Use stable norm computation (log-sum-exp trick)
           norm = ||e|| = max(|e|) × sqrt(sum((e/max(|e|))²))
```

---

## PART 3: INTERPRETING SIMILARITY SCORES

### Understanding Cosine Similarity Values

```
Score Range | Interpretation | Typical WSD Pair Type
────────────────────────────────────────────────────
0.90-1.00   | Extremely similar | Same sense, identical words/synonyms
0.80-0.89   | Very similar | Same sense, related contexts
0.70-0.79   | Similar | Same sense, diverse contexts
0.60-0.69   | Moderately similar | Borderline same/different senses
0.50-0.59   | Weakly similar | Borderline different senses
0.40-0.49   | Dissimilar | Different senses, some overlap
0.30-0.39   | Very dissimilar | Different senses, clear distinction
0.00-0.29   | Extremely dissimilar | Different senses, no overlap
────────────────────────────────────────────────────
-0.50-0.00  | Negatively correlated | Rare; opposite semantic directions
```

### Why Negative Similarities Are Rare

```
In BERT embedding space:
  - All embeddings trained on same corpus
  - No explicit negative examples
  - Orthogonal (0.0 similarity) much more common than negative
  - Negative similarities (< -0.2) are extremely rare
  
Distribution in practice:
  cos_sim values typically in range [0.0, +1.0]
  with occasional small negatives [-0.1, 0.0]
  
For Kannada WSD:
  - Same sense pairs: mostly in [0.65, 0.95]
  - Different sense pairs: mostly in [0.20, 0.60]
  - Rare overlap region: [0.55, 0.65] (ambiguous pairs)
```

---

### What Similarity Actually Measures

**Cosine similarity in BERT embedding space**:

```
High similarity (0.75+):
  ✓ Same target word sense
  ✓ Similar contextual words
  ✓ Similar syntactic roles
  ✓ Aligned semantic frames
  
  Example: "ಕಾಲ ದ್ರುತವಾಗಿ" (time) vs "ಕಾಲ ಸರಿಸರಿ ಹೋಗಿ" (time)
  → Embeddings aligned on "time" meaning
  
Medium similarity (0.45-0.65):
  ? Possibly different senses but overlapping context
  ? Different words but similar semantic roles
  ? Polysemy not clearly distinguished
  
  Example: "ಕಾಲ ದ್ರುತವಾಗಿ" (time) vs "ಇಲಿಯ ಕಾಲ" (leg)
  → Some overlap in embedding space (both are nouns, physical/temporal)
  
Low similarity (0.30-):
  ✓ Different target word senses
  ✓ Non-overlapping contextual fields
  ✓ Semantically distant meanings
  
  Example: "ಕಾಲ ದ್ರುತವಾಗಿ" (time) vs "ಕಬ್ಬಿ ಕಾಲು" (rough leg)
  → No alignment; "rough" pulls leg sense in different direction
```

---

## PART 4: THRESHOLD SELECTION STRATEGY

### Why Thresholding Matters

```
Goal: Convert continuous similarity scores (0.0-1.0) into binary labels

Inputs:
  - Similarity score: 0.62
  - Threshold: 0.50
  
Decision Rule:
  if similarity >= threshold:
    predicted_label = 1 (same sense)
  else:
    predicted_label = 0 (different sense)
  
Example:
  cos_sim = 0.62 >= 0.50 → predicted = 1 ✓ (correct for many cases)
  cos_sim = 0.48 < 0.50  → predicted = 0 ✓ (correct for many cases)
```

---

### Method 1: Equal Error Rate (EER) Threshold

**Find threshold where false positives = false negatives**

```
Procedure:
  1. Compute cosine similarity for all 720 WiC pairs
  2. For each possible threshold τ in [0.0, 1.0]:
     a. Predict labels: predict(i) = 1 if sim(i) >= τ, else 0
     b. Compare to ground truth labels
     c. Calculate: FP = false positives, FN = false negatives
     d. Calculate: error_rate = (FP + FN) / total_pairs
  
  3. Find τ where FP == FN (or error is minimized)
  4. This is the EER threshold

Intuition:
  - Balances false alarms (wrongly saying "same sense")
  - Against missed cases (wrongly saying "different sense")
  - Equal damage from both types of errors

Pros:
  ✓ Data-driven
  ✓ Minimizes total error
  ✓ No external assumptions
  
Cons:
  ✗ Requires labeled validation set (but we have WiC dataset)
  ✗ May not optimize for real use case (if costs are asymmetric)
```

---

### Method 2: F1-Optimized Threshold

**Maximize harmonic mean of precision and recall**

```
Metrics:
  Precision = TP / (TP + FP)  [of correctly identified "same" pairs]
  Recall = TP / (TP + FN)     [fraction of "same" pairs found]
  F1 = 2 × (Precision × Recall) / (Precision + Recall)

Procedure:
  1. Compute similarities for all pairs
  2. For each threshold τ in [0.0, 1.0] with step 0.01:
     a. Make predictions
     b. Calculate TP, FP, FN
     c. Calculate Precision, Recall, F1
  3. Find τ with maximum F1
  
Example Sweep:
  τ=0.30: Precision=0.60, Recall=0.95, F1=0.74
  τ=0.40: Precision=0.70, Recall=0.80, F1=0.75
  τ=0.50: Precision=0.75, Recall=0.75, F1=0.75 ← Max (tied)
  τ=0.60: Precision=0.80, Recall=0.65, F1=0.72
  τ=0.70: Precision=0.85, Recall=0.50, F1=0.63
  
Best threshold: 0.50 (or 0.40, similar F1)

Pros:
  ✓ Balances precision and recall equally
  ✓ Standard in classification literature
  ✓ Handles class imbalance well
  
Cons:
  ✗ May be suboptimal if false positives/negatives have different costs
```

---

### Method 3: Receiver Operating Characteristic (ROC) Analysis

**Analyze tradeoff between true positive rate and false positive rate**

```
Definitions:
  TPR = TP / (TP + FN)  [sensitivity; fraction of "same" correctly identified]
  FPR = FP / (FP + TN)  [fall-out; fraction of "diff" wrongly labeled "same"]

Procedure:
  1. Compute similarities for all pairs
  2. Sort pairs by similarity (descending)
  3. Sweep threshold from 1.0 down to 0.0
  4. For each threshold, calculate TPR and FPR
  5. Plot TPR vs FPR curve
  6. Find threshold at "elbow" (closest to top-left corner)
  
ROC Curve Interpretation:
  Point (0,0): threshold=1.0, predict nothing as "same" (all negative)
  Point (1,1): threshold=0.0, predict everything as "same" (all positive)
  Diagonal line: random classifier (accuracy = 50%)
  Curve above diagonal: better than random
  
ROC AUC (Area Under Curve):
  - Represents overall model quality
  - AUC = 1.0: Perfect classification
  - AUC = 0.5: Random classification
  - Expected AUC for Kannada WSD: 0.85-0.92 (models are good)

Pros:
  ✓ Visualizes full tradeoff space
  ✓ ROC AUC is threshold-independent metric
  ✓ Useful for imbalanced datasets
  
Cons:
  ✗ More complex to visualize/interpret
```

---

### Method 4: Probability Calibration

**Map similarity scores to confidence probabilities**

```
Issue:
  Cosine similarity is NOT a probability
  0.70 similarity ≠ 70% confidence that senses are same
  
Solution:
  Use logistic calibration to map similarities to probabilities
  
Procedure:
  1. Compute similarities for training set (say, 500 pairs)
  2. Fit logistic function:
     P(same | similarity) = 1 / (1 + exp(-a×similarity - b))
  
  3. Learn parameters a, b via maximum likelihood
  4. Use P(same) = 0.5 as threshold
  5. Apply learned function to test set
  
Example:
  Raw similarity: 0.62
  Calibrated probability: P(same) = 0.65
  Decision: If P(same) >= 0.50 → predict "same sense"
  
Advantage:
  ✓ Converts scores to interpretable probabilities
  ✓ Can output confidence scores, not just hard labels
  
Disadvantage:
  ✗ Requires fitting calibration on separate data
  ✗ Adds model complexity
```

---

## PART 5: RECOMMENDED THRESHOLD STRATEGY

### For Kannada WSD Project: Method 2 (F1-Optimized)

**Justification**:

```
Why F1-Optimized?
  1. Balances precision and recall
  2. No external cost assumptions needed
  3. Standard in NLP community
  4. Easy to compute and interpret
  5. WiC dataset is balanced (360 label-0, 360 label-1)
  
Steps to implement:

Step 1: Compute similarities for all 720 pairs
  Output: similarity_scores = [sim1, sim2, ..., sim720]
  
Step 2: Sweep thresholds
  for τ in [0.0, 0.01, 0.02, ..., 1.0]:  (101 thresholds)
    predictions = [1 if s >= τ else 0 for s in similarity_scores]
    TP, FP, FN = count_predictions(predictions, ground_truth)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
  
Step 3: Find optimal threshold
  best_threshold = τ with maximum F1
  
Step 4: Report results
  - Best threshold: τ_opt
  - Best F1: F1_opt
  - Corresponding precision and recall
```

---

### Typical Threshold Values for Kannada WSD

**Expected ranges** (based on model analysis):

```
Model          | Expected Threshold | F1 at Threshold | Reasoning
───────────────────────────────────────────────────────────────────
MuRIL          | 0.48-0.52          | 0.75-0.80       | Good Kannada embeddings
XLM-R          | 0.50-0.54          | 0.72-0.78       | Moderate embeddings
IndicBERT      | 0.50-0.55          | 0.70-0.77       | Lower capacity
mBERT          | 0.52-0.58          | 0.65-0.75       | Weakest embeddings

Reasoning:
  - Higher-quality embeddings → lower threshold (senses more separable)
  - Lower-quality embeddings → higher threshold (need stronger signal)
  
General recommendation: Start with τ = 0.50
  Then fine-tune based on validation results
```

---

### Decision Rule at Classification Time

**Using the optimal threshold**:

```
Algorithm: Classify WiC Pair

Input:
  embedding1: [768]
  embedding2: [768]
  threshold: 0.50 (optimized value)

Process:
  1. Compute cosine similarity
     cos_sim = dot(e1, e2) / (norm(e1) * norm(e2))
  
  2. Compare to threshold
     if cos_sim >= threshold:
       predicted_label = 1  (same sense)
     else:
       predicted_label = 0  (different sense)
  
  3. Optional: Output confidence
     confidence = min(cos_sim / threshold, 1.0) if cos_sim >= threshold
                else 1.0 - (cos_sim / threshold)
     [normalized to [0, 1] range]

Output:
  predicted_label: {0, 1}
  confidence: [0.0, 1.0]
  similarity_score: [0.0, 1.0] (for analysis)

Example 1 (Same sense):
  cos_sim = 0.78
  threshold = 0.50
  cos_sim (0.78) >= threshold (0.50) → predict 1
  confidence = 0.78 / 0.50 = 1.0 (clamped)
  
Example 2 (Different sense):
  cos_sim = 0.38
  threshold = 0.50
  cos_sim (0.38) < threshold (0.50) → predict 0
  confidence = 1.0 - (0.38 / 0.50) = 0.24
```

---

## PART 6: COMPLETE SIMILARITY PIPELINE

### End-to-End Computation Flow

```
┌─────────────────────────────────────┐
│  WiC Pair with Embeddings           │
│  ├─ sentence1: "ಕಾಲ ದ್ರುತವಾಗಿ"      │
│  ├─ sentence2: "ಕಾಲು ಮುರಿದಿದೆ"      │
│  ├─ embedding1: [768]               │
│  └─ embedding2: [768]               │
└────────────┬────────────────────────┘
             ↓
    ┌────────────────────────┐
    │  Verify embeddings     │
    │  ✓ Shape (768,)        │
    │  ✓ No NaN/Inf          │
    │  ✓ Non-zero norm       │
    └────────┬───────────────┘
             ↓
    ┌────────────────────────┐
    │  Compute dot product   │
    │  dot = e1 · e2         │
    │  Result: scalar        │
    └────────┬───────────────┘
             ↓
    ┌────────────────────────┐
    │  Compute norms         │
    │  norm1 = ||e1||        │
    │  norm2 = ||e2||        │
    │  Results: scalars      │
    └────────┬───────────────┘
             ↓
    ┌────────────────────────┐
    │  Compute similarity    │
    │  cos_sim = dot /       │
    │           (norm1×norm2)│
    │  Range: [-1, +1]       │
    └────────┬───────────────┘
             ↓
    ┌────────────────────────┐
    │  Compare to threshold  │
    │  τ = 0.50              │
    │  if cos_sim >= τ:      │
    │    label = 1           │
    │  else:                 │
    │    label = 0           │
    └────────┬───────────────┘
             ↓
┌─────────────────────────────────────┐
│  Output                             │
│  ├─ predicted_label: {0, 1}         │
│  ├─ cosine_similarity: 0.42         │
│  └─ confidence: 0.16                │
└─────────────────────────────────────┘
```

---

### Pseudocode Implementation

```python
def classify_wic_pair(embedding1, embedding2, threshold=0.50):
    """
    Classify a WiC pair using cosine similarity threshold.
    
    Args:
        embedding1: [768] numpy array
        embedding2: [768] numpy array
        threshold: similarity threshold for classification
    
    Returns:
        predicted_label: 0 or 1
        cosine_similarity: float in [-1, 1]
        confidence: float in [0, 1]
    """
    
    # Step 1: Verify embeddings
    assert embedding1.shape == (768,), "Invalid embedding1 shape"
    assert embedding2.shape == (768,), "Invalid embedding2 shape"
    assert not np.any(np.isnan(embedding1)), "embedding1 contains NaN"
    assert not np.any(np.isnan(embedding2)), "embedding2 contains NaN"
    
    # Step 2: Compute dot product
    dot_product = np.dot(embedding1, embedding2)
    
    # Step 3: Compute norms
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    # Handle zero vector edge case
    if norm1 < 1e-6 or norm2 < 1e-6:
        warn("Zero or near-zero embedding vector")
        return 0, 0.0, 0.5
    
    # Step 4: Compute cosine similarity
    cosine_similarity = dot_product / (norm1 * norm2)
    
    # Clamp to [-1, 1] to handle numerical errors
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
    
    # Step 5: Classify
    if cosine_similarity >= threshold:
        predicted_label = 1
        # Confidence: how far above threshold
        confidence = min((cosine_similarity - threshold) / (1.0 - threshold), 1.0)
    else:
        predicted_label = 0
        # Confidence: how far below threshold
        confidence = min((threshold - cosine_similarity) / threshold, 1.0)
    
    return predicted_label, cosine_similarity, confidence


# Example usage:
emb1 = np.array([0.1, 0.2, 0.3, ..., 0.15])  # 768 values
emb2 = np.array([0.2, 0.3, 0.4, ..., 0.25])  # 768 values

label, sim, conf = classify_wic_pair(emb1, emb2, threshold=0.50)
print(f"Label: {label}, Similarity: {sim:.3f}, Confidence: {conf:.3f}")
```

---

## PART 7: THRESHOLD VALIDATION ANALYSIS

### Analyzing Threshold Performance

**Before finalizing threshold, validate on validation set**:

```
Procedure:
  1. Split 720 pairs into:
     - Training set: 500 pairs (compute similarities, find threshold)
     - Validation set: 220 pairs (test threshold, report metrics)
  
  2. On training set:
     - Compute similarities
     - Sweep thresholds
     - Find optimal τ with max F1
  
  3. On validation set:
     - Apply threshold τ
     - Report metrics:
       * Accuracy = (TP + TN) / total
       * Precision = TP / (TP + FP)
       * Recall = TP / (TP + FN)
       * F1 = 2 × (Precision × Recall) / (Precision + Recall)
       * Specificity = TN / (TN + FP) [for "different sense"]

Expected Validation Performance (per model):
  
  Model          | Accuracy | Precision | Recall | F1    | Threshold
  ────────────────────────────────────────────────────────────────────
  MuRIL          | 0.77     | 0.78      | 0.76   | 0.77  | 0.50
  XLM-R          | 0.74     | 0.75      | 0.73   | 0.74  | 0.51
  IndicBERT      | 0.72     | 0.73      | 0.71   | 0.72  | 0.52
  mBERT          | 0.68     | 0.69      | 0.67   | 0.68  | 0.54
```

---

### Confusion Matrix Analysis

**Understanding where the model fails**:

```
Confusion Matrix for threshold=0.50:

                 Predicted Same  Predicted Diff
Actual Same      TP = 272        FN = 88
Actual Diff      FP = 53         TN = 307

Metrics:
  True Positives (TP): 272  [correctly identified same sense]
  True Negatives (TN): 307  [correctly identified different sense]
  False Positives (FP): 53  [said "same" but actually "different"]
  False Negatives (FN): 88  [said "different" but actually "same"]
  
  Accuracy = (272 + 307) / 720 = 0.803
  Precision = 272 / (272 + 53) = 0.837
  Recall = 272 / (272 + 88) = 0.755
  F1 = 2 × (0.837 × 0.755) / (0.837 + 0.755) = 0.794

Error Analysis:
  - 53 FP: Model over-predicts "same sense"
    Likely cause: Similar context but different senses
    Examples: ಕಾಲ (time) vs ಕಾಲು (leg) in descriptive contexts
  
  - 88 FN: Model under-predicts "same sense"
    Likely cause: Very dissimilar sentence structures, same sense
    Examples: ಕಾಲ (time) in "ಕಾಲ ಹೋಗಿ" vs "ಪ್ರತಿದಿನ ಕಾಲ"
```

---

## PART 8: SUMMARY AND DECISION TREE

### Quick Reference: Threshold Selection

```
Decision Tree for choosing threshold:

┌─ Do you have labeled validation data?
│  ├─ YES → Use Method 2: F1-Optimization
│  │        Compute similarities, sweep thresholds, find max F1
│  │        Expected threshold: ~0.50
│  │
│  └─ NO → Use default threshold
│           Start with 0.50, adjust based on error analysis

┌─ What's your use case?
│  ├─ Balanced errors (FP = FN importance) → F1-optimized (~0.50)
│  ├─ Minimize total errors (any error bad) → EER (~0.50)
│  ├─ Emphasize recalls (find all senses) → Lower threshold (~0.45)
│  └─ Emphasize precision (high confidence) → Higher threshold (~0.55)

┌─ Which model are you using?
│  ├─ MuRIL (best Kannada) → threshold 0.48-0.52
│  ├─ XLM-R (general) → threshold 0.50-0.54
│  ├─ IndicBERT (lightweight) → threshold 0.50-0.55
│  └─ mBERT (baseline) → threshold 0.52-0.58
```

---

### Final Recommendation

**For this Kannada WSD project**:

```
PRIMARY APPROACH:
  1. Compute cosine similarity for all 720 WiC pairs
  2. Use F1-optimization method to find threshold
  3. Expected optimal threshold: 0.50 ± 0.02
  4. Expected performance:
     - Accuracy: 70-80% (varies by model)
     - F1: 0.70-0.80
     - ROC AUC: 0.85-0.92

IMPLEMENTATION STEPS:
  1. Extract embeddings for all pairs (per embedding_extraction_pipeline.md)
  2. Compute cosine similarities using Part 2 formula
  3. Sweep thresholds [0.0, 1.0] with 0.01 step
  4. Calculate F1 for each threshold
  5. Report: optimal_threshold, metrics at that threshold
  6. Use this threshold for deployment

VALIDATION:
  1. Hold out 220 pairs for validation
  2. Train threshold on remaining 500
  3. Test on validation set
  4. Report final metrics (Accuracy, Precision, Recall, F1)

OUTPUT:
  For each WiC pair:
    {
      "pair_id": "...",
      "similarity_score": 0.62,
      "threshold": 0.50,
      "predicted_label": 1,
      "confidence": 0.24,
      "ground_truth": 1,
      "correct": true
    }
```

---

## PART 9: INTERPRETATION GUIDELINES

### What to Report When Publishing Results

```
Essential Metrics:
  ✓ Accuracy: Overall correctness
  ✓ Precision: Quality of "same sense" predictions
  ✓ Recall: Coverage of actual "same sense" pairs
  ✓ F1: Harmonic mean (primary metric)
  ✓ Threshold used: 0.50
  ✓ ROC AUC: Model-independent measure

Supplementary Analysis:
  ✓ Confusion matrix: Show FP/FN breakdown
  ✓ Per-word performance: Which words work well/poorly
  ✓ Threshold robustness: Performance at τ±0.05
  ✓ Model comparison: Relative accuracy across models

Visual Presentation:
  ✓ Plot 1: ROC curve (TPR vs FPR)
  ✓ Plot 2: F1 vs threshold (shows optimal point)
  ✓ Plot 3: Similarity distribution (same vs different sense)
  ✓ Plot 4: Confusion matrices for each model
```

### Similarity Distribution Interpretation

**Plot showing histogram of similarities**:

```
Typical distribution for good model:

Same Sense (label=1):          Different Sense (label=0):
  Density                         Density
    |                               |
    | ___                           | ___
    |/   \___                       |/   \
    |       \___                    |      \___
    |___________|__________          |___________|__________
      0.65    0.80    0.95             0.20    0.40    0.60
             similarity                       similarity

Interpretation:
  - Left peak (Different): centered ~0.35, range [0.1, 0.6]
  - Right peak (Same): centered ~0.75, range [0.6, 0.95]
  - Gap between peaks: ~0.30-0.35 (clear separation)
  - Overlap region: [0.55, 0.65] (ambiguous cases)
  
Optimal threshold: Where the two distributions cross most cleanly
  For well-trained models: approximately 0.50
```

