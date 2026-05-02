# t-SNE Visualization for Word Sense Embeddings
## Understanding Cluster Separation and Sense Distribution

---

## EXECUTIVE SUMMARY

**Purpose**: Visualize high-dimensional embeddings (768-D) in 2D/3D to inspect sense clusters

**Method**: t-Stochastic Neighbor Embedding (t-SNE)

**Key Insight**: "Points that are far apart in 768-D should be far in 2-D visualization"

**What We Visualize**:
- Token embeddings (one point per sentence occurrence)
- Color-coded by true sense (for validation)
- Clustered by k-means labels (to evaluate clustering)

**What Good Visualization Shows**:
- Clear spatial separation between senses
- Minimal overlap between clusters
- Within-cluster cohesion (points grouped together)
- Outliers visible and identifiable

**Example Output**:
```
Ideal "ಕಾಲ" (time/leg/death):
  
  Red cluster (Time)         Blue cluster (Leg)         Green cluster (Death)
       *  *                        •  •                       ◦  ◦
      * * * ===== SEPARATION ===== • • • ===== SEPARATION ===== ◦ ◦ ◦
       *  *                        •  •                       ◦  ◦
  
  Interpretation: Senses well-separated → good clustering → high accuracy expected
```

---

## PART 1: t-SNE FUNDAMENTALS

### What is t-SNE?

```
Acronym: t-Stochastic Neighbor Embedding

Purpose: Reduce high-dimensional data to 2D/3D while preserving local structure

Input: High-dimensional points (embeddings) [n, 768]
Output: 2D/3D coordinates [n, 2] or [n, 3]

Core Principle:
  "If two points are neighbors in high-D space,
   they should be neighbors in low-D visualization"

How it Works (Simplified):

Step 1: Compute Pairwise Similarities in High-D
  For each pair (xi, xj):
    similarity_high = exp(-||xi - xj||² / (2σ²))
    (Gaussian kernel: closer points = higher similarity)

Step 2: Compute Random Similarities in Low-D
  Initialize random 2D positions: [y1, y2, ..., yn]
  For each pair (yi, yj):
    similarity_low = 1 / (1 + ||yi - yj||²)
    (t-distribution: different from Gaussian)

Step 3: Minimize Divergence
  Use gradient descent to minimize KL divergence:
    KL = Σ_ij p_ij * log(p_ij / q_ij)
  
  Goal: Make low-D similarities match high-D similarities
  Iterate until convergence (typically 1000-5000 iterations)

Result: 2D visualization that preserves relationships
```

### Why t-SNE (not PCA)?

```
Comparison: t-SNE vs. PCA

Method    | Preserves | Computation | Interpretability | Best For
──────────────────────────────────────────────────────────────────
PCA       | Global    | Fast (sec)  | Linear direction | Overview
          | variance  |             |                  |
t-SNE     | Local     | Slow (min)  | Cluster shape    | Cluster
          | neighbors |             |                  | inspection
UMAP      | Local +   | Medium      | Clusters +       | General
          | global    | (fast)      | transitions      | exploration

For WSD Clustering Inspection:
  → t-SNE is best (shows exact cluster separation)
  → PCA for quick overview only
  → UMAP if you need faster computation on large datasets

Why t-SNE Better for Clusters:
  ✓ Emphasizes local neighborhoods (clusters stay together)
  ✓ Maximizes separation between clusters
  ✓ Makes outliers visible
  ✗ But: computationally expensive, hyperparameter-sensitive
```

---

## PART 2: DATA PREPARATION FOR t-SNE

### What to Visualize: Token Embeddings

```
Decision: What Data Points to Include?

Option 1: All Occurrences
  One point per sentence with target word
  
  Data:
    Sentence 1 with "ಕಾಲ" → Embedding 1 → Point 1 (red if Sense 1)
    Sentence 2 with "ಕಾಲ" → Embedding 2 → Point 2 (blue if Sense 2)
    ...
    Sentence 250 with "ಕಾಲ" → Embedding 250 → Point 250
  
  Visualization:
    250 points in 2D space, colored by sense or cluster
  
  Pros: Complete picture, shows all variations
  Cons: Can be crowded if > 500 points

Option 2: Sampled Subset
  If too many points (> 1000), sample 500-800 representative points
  
  Sampling strategy:
    - Stratified: equal samples per sense
    - Random: random 500 from 2000
    - Diverse: use clustering to get diverse samples
  
  Pros: Cleaner visualization, faster computation
  Cons: May miss outliers or rare senses
  
  Recommended for WSD: Sample 300-500 if > 1000 points

Option 3: Cluster Centers + Samples
  1. Run k-means clustering
  2. Plot cluster centroids
  3. Sample points from each cluster
  
  Visualization:
    Cluster centers as large dots
    Sample points as small dots
    Shows both aggregate structure and individual variation
  
  Pros: Informative for cluster inspection
  Cons: More complex interpretation

For Kannada WSD:
  PRIMARY: All occurrences (250-300 per word)
  BACKUP: Stratified sample if > 500
  ADVANCED: Centroid + samples for publication
```

### Data Format Preparation

```
Step 1: Collect Embeddings

Input:
  Dataset of 250 sentences with "ಕಾಲ"
  Each: {sentence, target_word, position, true_sense, embedding}

Code:
  embeddings = []
  labels = []  # true senses (1, 2, or 3)
  
  for item in dataset:
    emb = extract_embedding(item["sentence"], item["target_word"])
    embeddings.append(emb)
    labels.append(item["true_sense"])
  
  X = np.array(embeddings)  # [250, 768]
  y = np.array(labels)      # [250]

Step 2: Optionally Normalize
  Why normalize?
    - t-SNE sometimes sensitive to scale
    - Embeddings already ~normalized but ensure consistency
  
  Option A: L2 normalization (most common)
    from sklearn.preprocessing import normalize
    X_normalized = normalize(X, norm='l2')
  
  Option B: StandardScaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
  
  Recommendation: Use L2 (embeddings already normalized by models)

Step 3: Prepare Metadata
  metadata = {
    "word": "ಕಾಲ",
    "n_samples": 250,
    "n_senses": 3,
    "n_features": 768,
    "model": "MuRIL",
    "timestamp": "2026-05-02"
  }
  
  For each point, store:
    - Original sentence
    - Sense label
    - Cluster assignment (from k-means)
    - Any special flags (outlier, ambiguous, etc.)
```

---

## PART 3: t-SNE COMPUTATION

### Running t-SNE

```python
from sklearn.manifold import TSNE
import numpy as np

# Input: embeddings X [n, 768] and labels y [n]

# Step 1: Initialize t-SNE
tsne = TSNE(
    n_components=2,              # 2D output (or 3 for 3D)
    perplexity=30,               # Typical: 5-50, default 30
    learning_rate=200,           # Typical: 100-1000
    n_iter=1000,                 # At least 1000, usually 1000-5000
    random_state=42,             # For reproducibility
    verbose=1                     # Print progress
)

# Step 2: Compute t-SNE projection
X_2d = tsne.fit_transform(X)    # [n, 2]

print(f"t-SNE completed in {tsne.n_iter_} iterations")
print(f"KL divergence: {tsne.kl_divergence_:.4f}")

# Output:
#   X_2d has coordinates for each point in 2D space
#   Can now visualize: plot(X_2d[:, 0], X_2d[:, 1])
```

### t-SNE Hyperparameters

```
Parameter 1: Perplexity
  Definition: Balance between local and global structure
  Range: 5-50 (typical); 5-100 for large datasets
  
  Impact:
    Low perplexity (5-10):
      - Emphasizes local structure (very tight clusters)
      - Can create artificial fragments
      - Good for: small datasets (< 100 points)
    
    Medium perplexity (20-40):
      - Balanced local/global (recommended)
      - Good for: most WSD tasks (100-500 points)
    
    High perplexity (50+):
      - Emphasizes global structure
      - Clusters merge together more
      - Good for: large datasets (> 1000 points)
  
  Formula: perplexity ≈ n / 30 (heuristic)
  
  For Kannada WSD:
    n=250 (per word) → perplexity = 250/30 ≈ 8-10? No, use default 30
    n=1000 (combined) → perplexity = 30-40
    n=5000 (all words) → perplexity = 50

Parameter 2: Learning Rate
  Definition: Step size for gradient descent
  Range: 100-1000 (sometimes auto-scales)
  
  Impact:
    Too low (50):
      - Slow convergence, may not reach good solution
    
    Optimal (200-500):
      - Good convergence, stable results
    
    Too high (1000+):
      - Can oscillate, diverge
  
  Recommendation: Use default or 200

Parameter 3: Number of Iterations
  Definition: How many gradient descent steps
  Range: 1000-5000 minimum
  
  Guidance:
    1000 iterations: Usually sufficient
    5000 iterations: More refined (slower)
  
  How to check if converged:
    Watch KL divergence during fitting
    Should decrease monotonically
    Convergence when decreasing slowly
  
  For WSD: Use n_iter=1000 (standard)

Parameter 4: Random State
  Definition: Random seed for reproducibility
  Set to 42 (or any constant) to get same results on rerun
  
  Best Practice:
    - Always set random_state for reproducibility
    - Change to see sensitivity to initialization
```

---

## PART 4: VISUALIZATION INTERPRETATION

### Ideal Cluster Patterns

```
Pattern 1: GOOD SEPARATION (Ideal)

Visualization:
  
  Cluster 1 (Red)        Cluster 2 (Blue)       Cluster 3 (Green)
       * * *                   • • •                    ◦ ◦ ◦
      * * * *                 • • • •                  ◦ ◦ ◦ ◦
       * * *                   • • •                    ◦ ◦ ◦
  
  ↓↓↓ CLEAR GAP ↓↓↓      ↓↓↓ CLEAR GAP ↓↓↓
  
  Cluster 1 (Red)        Cluster 2 (Blue)       Cluster 3 (Green)
       * * *                   • • •                    ◦ ◦ ◦

Interpretation:
  ✓ Senses clearly separated
  ✓ Within-cluster cohesion high
  ✓ Between-cluster distance large
  ✓ Few/no outliers
  ✓ No ambiguous middle zones
  
Expected WSD Performance:
  - Purity: 80-90%
  - Silhouette score: 0.65-0.85
  - Accuracy: 75-85%
  - Recommendation: Use this for WSD

Real Example ("ಕಾಲ" - Time/Leg/Death):
  Time cluster (Time contextual words):
    - "ದ್ರುತವಾಗಿ" (quickly), "ಹೋಗಿ" (went), "ದಿವಸ" (day)
    - Forms tight group on left
  
  Leg cluster (Body-related words):
    - "ಮುರಿದೆ" (broken), "ನಿಜ" (true), "ಪ್ರಚಲಿತ" (active)
    - Forms tight group on right
  
  Death cluster (Rare senses):
    - "ಅಂತ್ಯ" (end), "ಮರಣ" (death), "ಕೃತ್ಯತೆ" (task)
    - Smaller group in center-top
```

### Problematic Cluster Patterns

```
Pattern 2: POOR SEPARATION (Problem 1)

Visualization:

  Cluster 1 (Red)     Cluster 2 (Blue)
       * * *              • • •
      * • * •            * * • •
       * * * •            * * •
  
  ↓↓↓ SIGNIFICANT OVERLAP ↓↓↓

Interpretation:
  ✗ Red and blue points mixed
  ✗ Unclear boundaries
  ✗ Senses overlap in embedding space
  ✗ Hard to separate

Causes:
  1. Senses are contextually similar
     Example: "ಹಾಡು" (song/sing) - similar contexts
  2. Embeddings not discriminative enough
     Solution: Try better model (MuRIL > XLM-R)
  3. Polysemy not true senses
     Solution: Reconsider sense granularity

Expected WSD Performance:
  - Purity: 55-70%
  - Silhouette: 0.30-0.50
  - Accuracy: 60-70%
  - Recommendation: Combine with gloss-based or supervised learning

What to Do:
  → Check if senses truly distinct
  → Use larger corpus (more examples)
  → Try different embedding model
  → Reduce k if too many senses
```

### More Problematic Patterns

```
Pattern 3: DEGENERATE CLUSTERING (Problem 2)

Visualization:

      * • ◦ * • ◦
     * • ◦ * • ◦ *
    * • ◦ * • ◦ * •
   * • ◦ * • ◦ * • ◦
  
  ↓↓↓ RANDOM SCATTERING, NO CLUSTERS ↓↓↓

Interpretation:
  ✗ All colors mixed evenly
  ✗ No apparent structure
  ✗ Clustering completely failed

Causes:
  1. k wrong (too large)
     Solution: Reduce k to 2 or 3
  2. Embedding quality poor
     Solution: Check embedding model
  3. Data too homogeneous
     Solution: Verify data contains multiple senses
  4. t-SNE parameters bad
     Solution: Adjust perplexity or random state

What to Do:
  → First: Verify k matches true number of senses
  → Second: Check embedding quality (visualize all words together)
  → Third: Inspect raw data (confirm senses present)
  → Finally: Adjust t-SNE perplexity and rerun

Pattern 4: ONE DOMINANT CLUSTER (Problem 3)

Visualization:

      Cluster 1 (Red) - 190 points
        * * * * * *
       * * * * * * *
        * * * * * *
       * * * * * * *
  
      Scattered points (Blue/Green) - 5 points each
        • ◦ ◦ • ◦ ◦

Interpretation:
  ✗ One sense dominates (190/200 = 95%)
  ✗ Other senses rare
  ✗ k-means collapsed to one cluster

Causes:
  1. Highly skewed sense distribution
     Common for polysemy (one sense 90%, others 10%)
  2. Rare senses under-represented
     Need more data to validate

Expected WSD Performance:
  - Purity: 95% (trivial - just predict dominant sense)
  - F1 on rare senses: Very low
  - Accuracy misleading (class imbalance)

What to Do:
  → Check sense distribution (balanced?)
  → Oversample rare senses if possible
  → Use weighted loss during training
  → Evaluate per-sense metrics (not just accuracy)

Pattern 5: CLEAR OUTLIERS (Information-Rich)

Visualization:

  Normal Cluster (Red)      Outliers (Blue)
       * * *                    •
      * * * *                         •
       * * *
                              •
  
  ↓↓↓ WELL-DEFINED CORE + CLEAR OUTLIERS ↓↓↓

Interpretation:
  ✓ Main cluster cohesive
  ✓ Outliers far from core
  ✓ Can identify and analyze individually

Causes:
  1. Ambiguous sentences (genuinely hard to classify)
  2. Boundary cases (multiple valid senses)
  3. Annotation errors (mislabeled data)
  4. True errors in model/clustering

What to Do:
  → INSPECT outliers manually
  → Determine if errors or legitimate ambiguity
  → Re-annotate if needed
  → Track for error analysis

Pattern 6: NESTED CLUSTERS (Sub-Structure)

Visualization:

  Large Red Cluster (Sense 1) with sub-structure:
       * * - - - - * *
      * - (sub-cluster A) - *
       * - - - - - - *
  
      * * - - - - * *
     * - (sub-cluster B) - *
      * - - - - - - *

Interpretation:
  ◐ One sense has multiple senses itself (rare)
  ◐ Or: sense has sub-types
  ◐ Or: polysemy within polysemy

Example:
  "ಹಾಡು" (Song):
    - Sub-sense 1: Classical Kannada song
    - Sub-sense 2: Modern film song
    - Sub-sense 3: Lullaby/folk song
    All are "songs" but different contexts

What to Do:
  → Check if fine-grained sense distinctions exist
  → Consider using finer sense granularity (k+1 or k+2)
  → Or accept as variation within sense
```

---

## PART 5: QUANTITATIVE INTERPRETATION GUIDELINES

### Metrics from Visualization

```
Metric 1: Visual Separation Score (Not Formal, But Useful)

Procedure:
  1. Draw boundary around each cluster
  2. Estimate percentage of overlap
  3. Rate on scale

Rating Scale:
  
  Score 5 (Excellent):
    - No overlap between clusters
    - Clear, distinct regions
    - Visual similarity: 100% distinct
    Expected clustering accuracy: 80%+
  
  Score 4 (Good):
    - Minimal overlap (< 5%)
    - Mostly distinct
    - Visual similarity: 95% distinct
    Expected clustering accuracy: 75%+
  
  Score 3 (Fair):
    - Some overlap (10-20%)
    - Distinguishable but fuzzy
    - Visual similarity: 80% distinct
    Expected clustering accuracy: 65%+
  
  Score 2 (Poor):
    - Significant overlap (30-50%)
    - Hard to distinguish
    - Visual similarity: 50% distinct
    Expected clustering accuracy: 55%+
  
  Score 1 (Very Poor):
    - Extensive overlap or no structure
    - Indistinguishable
    - Visual similarity: < 50% distinct
    Expected clustering accuracy: 50% (random)

For Kannada WSD:
  Target: Score 4-5 (good/excellent)
  Acceptable: Score 3+ (fair or better)
  Problematic: Score 2 or less

Metric 2: Cluster Compactness (Empirical)

Procedure:
  1. For each cluster, measure radius (max distance from center to any point)
  2. Small radius = compact, tight cluster
  3. Large radius = spread out cluster

Visual Guide:
  Compact (Good):    Radius ≈ 1-2 units, points close together
  Moderate (OK):     Radius ≈ 3-5 units, some spread
  Loose (Bad):       Radius ≈ 10+ units, very spread out

Expected relationship:
  Compact clusters → high purity (90%+)
  Loose clusters → lower purity (60-70%)

Metric 3: Inter-Cluster Gaps (Empirical)

Procedure:
  1. Measure closest distance between any two clusters
  2. Large gap = clear separation
  3. Small gap = clusters close together

Visual Guide:
  Large gap (Good):       Clusters separated by > 5 units
  Medium gap (OK):        Clusters separated by 3-5 units
  Small gap (Bad):        Clusters separated by < 2 units or touching

Expected relationship:
  Large gap → low false positives (better for WSD)
  Small gap → high false positives (errors in classification)

Metric 4: Outlier Percentage (Empirical)

Procedure:
  1. Identify points far from cluster center
  2. Count percentage of outliers
  3. Typical threshold: points > 3x cluster radius away

Outlier Rate Interpretation:
  < 2%:     Excellent (very cohesive clusters)
  2-5%:     Good (some outliers but acceptable)
  5-10%:    Fair (notable outliers)
  > 10%:    Poor (many outliers, clustering unstable)

For WSD:
  < 5%: Target
  Expected: 5-15% (depends on data quality)
```

### Translating Visual Insights to Metrics

```
Visual Pattern → Predicted Metric

Pattern                      | Silhouette | Purity | Expected Accuracy
─────────────────────────────────────────────────────────────────────
Perfect separation (Score 5) | 0.70-0.85  | 85-95% | 80-90%
Good separation (Score 4)    | 0.55-0.70  | 75-85% | 70-80%
Fair overlap (Score 3)       | 0.40-0.55  | 65-75% | 60-75%
Poor overlap (Score 2)       | 0.20-0.40  | 50-65% | 50-65%
No structure (Score 1)       | < 0.20     | < 50%  | ~50% (random)

Use this to PREDICT performance without running full evaluation
```

---

## PART 6: PRACTICAL VISUALIZATION GUIDE

### Creating Visualizations

```python
import matplotlib.pyplot as plt
import numpy as np

# Input: X_2d [n, 2] from t-SNE, labels y [n], cluster_labels pred [n]

# Visualization 1: Ground Truth Sense Labels
fig, ax = plt.subplots(figsize=(10, 8))

# Color map for senses
colors = {0: 'red', 1: 'blue', 2: 'green'}
sense_names = {0: 'Time', 1: 'Leg', 2: 'Death'}

for sense in range(3):
    mask = (y == sense)
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1], 
               c=colors[sense], label=sense_names[sense], 
               s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.set_title('Word "ಕಾಲ" - t-SNE with Ground Truth Senses')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('tsne_groundtruth.png', dpi=150)
plt.show()

print("Saved: tsne_groundtruth.png")

# Visualization 2: K-Means Cluster Labels
fig, ax = plt.subplots(figsize=(10, 8))

colors_cluster = {0: 'red', 1: 'blue', 2: 'green'}
cluster_names = {0: 'Cluster 1', 1: 'Cluster 2', 2: 'Cluster 3'}

for cluster in range(3):
    mask = (pred == cluster)
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1], 
               c=colors_cluster[cluster], 
               label=cluster_names[cluster],
               s=100, alpha=0.6, edgecolors='black', linewidth=0.5)

ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.set_title('Word "ಕಾಲ" - t-SNE with K-Means Clusters')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('tsne_clusters.png', dpi=150)
plt.show()

print("Saved: tsne_clusters.png")

# Visualization 3: Side-by-Side Comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Ground truth
for sense in range(3):
    mask = (y == sense)
    ax1.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                c=colors[sense], label=sense_names[sense],
                s=100, alpha=0.6, edgecolors='black')

ax1.set_xlabel('t-SNE Dimension 1')
ax1.set_ylabel('t-SNE Dimension 2')
ax1.set_title('Ground Truth Senses')
ax1.legend()
ax1.grid(True, alpha=0.3)

# K-means clusters
for cluster in range(3):
    mask = (pred == cluster)
    ax2.scatter(X_2d[mask, 0], X_2d[mask, 1],
                c=colors_cluster[cluster], 
                label=cluster_names[cluster],
                s=100, alpha=0.6, edgecolors='black')

ax2.set_xlabel('t-SNE Dimension 1')
ax2.set_ylabel('t-SNE Dimension 2')
ax2.set_title('K-Means Clusters')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tsne_comparison.png', dpi=150)
plt.show()

print("Saved: tsne_comparison.png")

# Visualization 4: Interactive with Annotations
# For important points, add text labels
fig, ax = plt.subplots(figsize=(12, 10))

for sense in range(3):
    mask = (y == sense)
    ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
               c=colors[sense], label=sense_names[sense],
               s=100, alpha=0.6, edgecolors='black')

# Annotate cluster centers
from scipy.spatial.distance import cdist
centers = []
for sense in range(3):
    mask = (y == sense)
    center = np.mean(X_2d[mask], axis=0)
    centers.append(center)
    ax.plot(center[0], center[1], 'k*', markersize=20, markeredgewidth=2)
    ax.annotate(f'Sense {sense+1}', (center[0], center[1]), 
                fontsize=12, fontweight='bold',
                xytext=(5, 5), textcoords='offset points')

ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.set_title('Word "ಕಾಲ" - Annotated with Sense Centers')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('tsne_annotated.png', dpi=150)
plt.show()

print("Saved: tsne_annotated.png")
```

### 3D Visualization (Optional)

```python
from mpl_toolkits.mplot3d import Axes3D

# Compute 3D t-SNE
tsne_3d = TSNE(n_components=3, perplexity=30, n_iter=1000)
X_3d = tsne_3d.fit_transform(X)  # [n, 3]

# Plot 3D
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

for sense in range(3):
    mask = (y == sense)
    ax.scatter(X_3d[mask, 0], X_3d[mask, 1], X_3d[mask, 2],
               c=colors[sense], label=sense_names[sense],
               s=100, alpha=0.6, edgecolors='black')

ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.set_zlabel('t-SNE Dimension 3')
ax.set_title('Word "ಕಾಲ" - 3D t-SNE Visualization')
ax.legend()
plt.tight_layout()
plt.savefig('tsne_3d.png', dpi=150)
plt.show()

print("Saved: tsne_3d.png (rotate with mouse to inspect clusters)")
```

---

## PART 7: INTERPRETATION CHECKLIST

### Step-by-Step Interpretation Guide

```
Step 1: Visual Inspection (5 minutes)

□ Generate t-SNE visualization
□ Visually identify clusters (colored regions)
□ Count number of distinct clusters
  Expected: Should see k distinct clusters (k = number of senses)

Step 2: Cluster Structure Check (5 minutes)

For each cluster:
  □ Cohesive? (points close together)
  □ Circular or elongated? (shape indicates homogeneity)
  □ Any outliers? (isolated points)
  
Expectation:
  ✓ Tight, circular clusters = good clustering
  ✗ Elongated or fragmented = poor clustering

Step 3: Separation Assessment (5 minutes)

□ Are clusters touching?
□ Is there clear white space between clusters?
□ Estimate overlap percentage

Rating:
  < 5% overlap:   Score 5 (Excellent)
  5-15% overlap:  Score 4 (Good)
  15-30% overlap: Score 3 (Fair)
  > 30% overlap:  Score 2-1 (Poor)

Step 4: Outlier Analysis (10 minutes)

□ Identify outliers (isolated points far from clusters)
□ Count outlier percentage
□ For each outlier, check:
  - Is it misclassified? (wrong color in true sense view)
  - Is it ambiguous? (could belong to multiple senses)
  - Is it an error? (mislabeled in ground truth)

Step 5: Cross-Validation

□ Generate visualization with cluster labels (from k-means)
□ Compare to ground truth visualization
□ Check if cluster colors align with sense colors

Expected:
  - Cluster 1 should mostly contain Sense 1 color
  - Cluster 2 should mostly contain Sense 2 color
  - Etc.

If colors don't align:
  → Clusters may be permuted (relabel and retry)
  → Or: Clustering failed

Step 6: Quantitative Check

□ Estimate silhouette score from visual appearance
  Silhouette formula: (distance_to_nearest_cluster - distance_within_cluster) / max(...)
  
□ Predict purity percentage
  Example: If 90% of red points in Cluster 1, purity ≈ 90%

□ Estimate expected WSD accuracy
  Use mapping table from Part 5

Step 7: Final Assessment

Conclusion: Cluster quality
  Excellent (Score 5):
    ✓ Use for WSD directly
    ✓ Expect 80%+ accuracy
  
  Good (Score 4):
    ✓ Use for WSD with validation
    ✓ Expect 70-80% accuracy
  
  Fair (Score 3):
    ◐ Combine with other methods
    ◐ Expect 60-75% accuracy
  
  Poor (Score 2-1):
    ✗ Do not use for standalone WSD
    ✗ Needs investigation or alternative approach
```

---

## PART 8: TROUBLESHOOTING POOR VISUALIZATIONS

### Problem: Clusters Not Separated

```
Symptom:
  t-SNE shows overlapping clusters, no clear structure

Diagnosis:

Cause 1: Wrong Perplexity
  Solution:
    - Try lower perplexity (10-15)
    - Or higher perplexity (40-50)
    - Rerun and compare
  
  Implementation:
    tsne_low = TSNE(perplexity=15, n_iter=2000)
    X_2d_low = tsne_low.fit_transform(X)
    
    tsne_high = TSNE(perplexity=50, n_iter=2000)
    X_2d_high = tsne_high.fit_transform(X)
    
    # Visualize both, choose better one

Cause 2: Too Few Iterations
  Solution:
    - Increase iterations to 2000-5000
    - t-SNE may not have converged
  
  Implementation:
    tsne = TSNE(n_iter=5000, verbose=1)
    X_2d = tsne.fit_transform(X)
    # Check if KL divergence still decreasing

Cause 3: Wrong k (K-Means)
  Solution:
    - If using k-means, verify k = number of senses
    - If k wrong, clustering will fail
    - Check: Does ground truth have exactly k senses?
  
  Verification:
    unique_senses = np.unique(y)
    n_senses_true = len(unique_senses)
    if n_senses_true != k:
      print(f"Error: k={k} but data has {n_senses_true} senses")

Cause 4: Poor Embedding Quality
  Solution:
    - Check if embeddings are good
    - Try different embedding model
    - Example: MuRIL (best) vs mBERT (worst)
  
  Test:
    Visualize similar words' embeddings
    If all words show poor separation → embedding problem
    If only this word poor → sense ambiguity (not embedding)

Cause 5: Senses Genuinely Similar
  Solution:
    - Some words have overlapping senses (OK)
    - Not all clustering can achieve perfect separation
    - This is expected for difficult words
  
  Acceptance:
    Score 3 (Fair) is acceptable
    Use ensemble methods (gloss-based + supervised)

Debugging Code:

def diagnose_tsne_quality(X, y, perplexities=[10, 30, 50]):
    """Try different perplexities and visualize"""
    
    fig, axes = plt.subplots(1, len(perplexities), figsize=(15, 5))
    
    for idx, perp in enumerate(perplexities):
        tsne = TSNE(perplexity=perp, n_iter=2000, random_state=42)
        X_2d = tsne.fit_transform(X)
        
        ax = axes[idx]
        for sense in range(len(np.unique(y))):
            mask = (y == sense)
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], s=50, alpha=0.6)
        
        ax.set_title(f'Perplexity={perp}')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
    
    plt.tight_layout()
    plt.savefig('tsne_perplexity_comparison.png')
    plt.show()

# Usage
diagnose_tsne_quality(X, y, [10, 30, 50])
```

### Problem: Clusters Too Tight (Not Showing Structure)

```
Symptom:
  All points clustered into tiny region, hard to see internal structure

Cause:
  Perplexity too low or data scale issue

Solution:
  - Increase perplexity
  - Normalize data if not already normalized
  - Use exaggeration=4.0 to magnify differences

Implementation:
  tsne = TSNE(
      perplexity=50,
      exaggeration=4.0,
      n_iter=1000
  )
```

### Problem: Random Noise Instead of Clusters

```
Symptom:
  t-SNE shows random scattering, no visible clusters

Causes:
  1. Embeddings are not good (all points equally distant)
  2. k-means with wrong k
  3. Data has no sense distinctions

Diagnostics:
  
  Step 1: Check raw embeddings
    distance_matrix = pairwise_distances(X)
    print(np.mean(distance_matrix))  # Should be ~1.5-2.0
    
    If all distances ≈ same value:
      → Embeddings uniformly distributed (bad)
      → Try different model or preprocessing

  Step 2: Check if k correct
    n_unique_labels = len(np.unique(y))
    print(f"Data has {n_unique_labels} senses")
    print(f"Using k={k_means_k}")
    
    If mismatch:
      → Use correct k

  Step 3: Check ground truth labels
    for sense in np.unique(y):
        count = np.sum(y == sense)
        print(f"Sense {sense}: {count} examples")
    
    If one sense dominates (e.g., 95%):
      → Class imbalance, not model issue
```

---

## PART 9: PRACTICAL WORKFLOW

### Complete Visualization Workflow

```
Start: Target word "ಕಾಲ" with 3 senses

Step 1: Data Preparation (5 min)
  ✓ Load embeddings (250 sentences)
  ✓ Load ground truth labels (sense 1/2/3)
  ✓ Load k-means cluster assignments

Step 2: t-SNE Computation (2-5 min)
  ✓ Run t-SNE with perplexity=30, n_iter=1000
  ✓ Get 2D coordinates X_2d [250, 2]

Step 3: Visualization Generation (5 min)
  ✓ Plot ground truth (colored by sense)
  ✓ Plot clusters (colored by k-means label)
  ✓ Generate side-by-side comparison

Step 4: Interpretation (15 min)
  ✓ Check cluster separation (visual score 1-5)
  ✓ Identify outliers (mark for inspection)
  ✓ Cross-validate ground truth vs. clusters
  ✓ Estimate metrics (silhouette, purity)

Step 5: Analysis Summary (5 min)
  ✓ Document findings (good/fair/poor)
  ✓ Recommend next steps
  ✓ Save plots for report

Step 6: Decision Making
  IF clusters well-separated (Score 4-5):
    → Use k-means for WSD
    → Expected accuracy 75-85%
  
  IF clusters fair (Score 3):
    → Combine with gloss-based/supervised
    → Expected accuracy 70-80%
  
  IF clusters poor (Score 1-2):
    → Investigate cause
    → Try different model or preprocessing
    → Or use pure supervised approach

Total Time: ~30 minutes per word × 45 words = 22.5 hours (parallel: 30 min)
```

---

## PART 10: SUMMARY TABLE

### t-SNE Visualization Guide

```
Aspect                  | Details
────────────────────────────────────────────────────────
Dimensionality          | 768-D embeddings → 2D/3D
Method                  | t-Stochastic Neighbor Embedding
Typical Parameters      | perplexity=30, n_iter=1000
Computation Time        | 1-5 minutes per word
Key Hyperparameter      | Perplexity (typically 30)
Distance Metric         | Usually Euclidean
Visualization Tool      | matplotlib, plotly, or tensorboard
Good Pattern            | Distinct, separated clusters
Bad Pattern             | Overlapping, mixed, or random points
Score 5 (Excellent)     | < 5% overlap, separation clear
Score 4 (Good)          | 5-15% overlap, mostly distinct
Score 3 (Fair)          | 15-30% overlap, distinguishable
Score 2 (Poor)          | > 30% overlap, hard to separate
Expected Silhouette     | 0.50-0.85 (good) or < 0.20 (poor)
Expected Purity         | 70-90% (good) or < 60% (poor)
Expected Accuracy       | 75-85% (good) or < 60% (poor)
For WSD Task            | Use Score 3+ (combine with other methods)
Interpretation Time     | 15-20 minutes per word
Next Step               | Verify with confusion matrix or metrics
```

---

## CONCLUSION

**t-SNE for WSD Visualization**:

1. **Purpose**: See if embeddings naturally cluster by sense
2. **Process**: Extract embeddings → Run t-SNE → Visualize with sense colors and cluster colors
3. **Interpretation**: Look for clear spatial separation (good) vs overlap (bad)
4. **Patterns**: 5-point scale (Score 5=excellent, Score 1=very poor)
5. **Decision**: Use visual score to predict WSD performance (75-85% if Score 4-5)

**Key Insight**: "Good visualization → Good clustering → High WSD accuracy"

**For Kannada WSD**:
- Visualize each of 45 words
- Target Score 4-5 (good separation)
- Use visualization to guide model selection
- Combine with gloss-based and supervised for final accuracy

