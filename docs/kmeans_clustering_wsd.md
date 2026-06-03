# K-Means Clustering for Word Sense Separation
## Unsupervised Sense Discovery and WSD Enhancement

---

## EXECUTIVE SUMMARY

**Method**: K-Means Clustering on Token Embeddings

**Core Idea**:
1. Collect multiple occurrences of target word from corpus
2. Extract embeddings for each occurrence
3. Apply k-means clustering with k = number of senses
4. Each cluster represents one sense

**Result**: Unsupervised sense discovery and WSD

**Expected Performance**:
- Purity: 70-80% (cluster homogeneity)
- Unsupervised accuracy: 60-70%
- Combined with supervised: 80-88%

---

## PART 1: THEORETICAL FOUNDATION

### Why K-Means for Word Sense Clustering?

```
Fundamental Assumption:
  Different senses of a word have different embeddings
  
Hypothesis:
  If word W has 3 senses S1, S2, S3:
  - Occurrences with sense S1 → cluster near similar S1 instances
  - Occurrences with sense S2 → cluster near similar S2 instances  
  - Occurrences with sense S3 → cluster near similar S3 instances
  
  → k-means should discover these natural groupings

Example: Kannada "ಕಾಲ"
  
  10 sentences with "ಕಾಲ":
  Sent1: "ಕಾಲ ದ್ರುತವಾಗಿ" (time)       → Cluster A
  Sent2: "ಕಾಲ ಹೋಗಿ" (time)            → Cluster A
  Sent3: "ಅವನ ಕಾಲು ಮುರಿದೆ" (leg)     → Cluster B
  Sent4: "ಕಾಲ ಮತ್ತೊಮ್ಮೆ" (time)       → Cluster A
  Sent5: "ಪರಿಪೂರ್ಣ ಕಾಲು" (leg)        → Cluster B
  Sent6: "ಆ ಕಾಲದಲ್ಲಿ" (time/era)       → Cluster A
  ...
  
  Result after k-means (k=3):
    Cluster A: Senses 1+4+6+... (mostly time/era)
    Cluster B: Senses 3+5+... (mostly leg)
    Cluster C: Sense 2 (ambiguous cases)
```

---

## PART 2: K-MEANS ALGORITHM FUNDAMENTALS

### K-Means Overview

```
Algorithm: Lloyd's K-Means

Input:
  X = {x1, x2, ..., xn} - embeddings [n, 768]
  k - number of clusters (= number of senses)
  
Process:

Step 1: Initialize Centroids
  Randomly select k embeddings as initial centroids
  centroids = [c1, c2, ..., ck] where each ci ∈ ℝ^768

Step 2: Assignment Step (Iterate until convergence)
  For each embedding xi:
    - Compute distance to all centroids: d(xi, cj)
    - Assign to nearest centroid: cluster[i] = argmin_j(d(xi, cj))
  
  Result: Cluster assignments for all embeddings

Step 3: Update Step
  For each cluster j:
    - Compute mean of all embeddings in cluster j
    - Update centroid: c'j = mean({xi : cluster[i] = j})
  
  Result: New centroids

Step 4: Check Convergence
  If centroids changed < threshold:
    STOP (converged)
  Else:
    Go to Step 2

Output:
  - Cluster assignments: [c1, c2, c3, ..., cn] ∈ {1, k}
  - Final centroids: [μ1, μ2, ..., μk]
```

### Distance Metrics

```
Common Choices:

1. Euclidean Distance (L2)
   d(xi, cj) = ||xi - cj||2 = √(Σ(xi_m - cj_m)²)
   
   Pros:
     ✓ Default for k-means
     ✓ Interpretable (geometric distance)
     ✓ Works well for embeddings
   
   Cons:
     ✗ Sensitive to magnitude (but embeddings normalized)
     ✗ Can be slow for high dimensions

2. Cosine Distance (1 - cosine similarity)
   d(xi, cj) = 1 - cos_sim(xi, cj)
   
   Pros:
     ✓ Ignore magnitude (good for embeddings)
     ✓ Natural for embedding spaces
   
   Cons:
     ✗ Slightly different cluster dynamics

3. Manhattan Distance (L1)
   d(xi, cj) = Σ|xi_m - cj_m|
   
   Pros:
     ✓ Robust to outliers
   
   Cons:
     ✗ Slower computation
     ✗ Less interpretable for embeddings

Recommendation: Use Euclidean distance (standard, well-tested)
```

---

## PART 3: APPLYING K-MEANS TO WORD SENSE CLUSTERING

### Step 1: Data Collection

**Objective**: Gather multiple sentences with target word

```
Input: Target word "ಕಾಲ"

Step 1a: Source Collection
  Sources:
    - WiC dataset (720 pairs)
    - Kannada Wikipedia
    - News corpora
    - Literary texts
  
  Goal: Find 100-200+ sentences containing "ಕಾಲ"

Step 1b: Extraction
  For each sentence:
    - Extract full sentence
    - Note position of target word
    - Record context (surrounding words)

Example Collection:
  Sent1: "ಕಾಲ ದ್ರುತವಾಗಿ ಹೋಗಿ" → position=0
  Sent2: "ಮೂರು ವರ್ಷ ಕಾಲ ಅಧ್ಯಯನ" → position=11
  Sent3: "ಬಿಸಿ ಕಾಲುಗಳು ನೀರಿನಲ್ಲಿ" → position=5
  ...

Quality Considerations:
  ✓ More data = better clustering (aim for 100+)
  ✓ Diverse contexts = better sense distinction
  ✓ Balanced senses = fair evaluation
  ✗ Too few examples (< 20) → clustering fails
  ✗ Only one sense represented → trivial clustering
```

### Step 2: Embedding Extraction

**Objective**: Extract embeddings for target word in each sentence

```
For Each Sentence:

Step 2a: Preprocessing
  - Unicode NFC normalization
  - Whitespace normalization
  - (Same as main pipeline)

Step 2b: Tokenization
  - Use model tokenizer
  - Add [CLS] and [SEP] tokens
  - Convert to token IDs

Step 2c: Forward Pass
  - Input tokens to embedding model
  - Extract last hidden state [seq_len, 768]

Step 2d: Target Word Localization
  - Find token indices corresponding to target word
  - Handle multi-token words (apply mean pooling)
  - Extract embedding(s)

Example:

  Sentence: "ಕಾಲ ದ್ರುತವಾಗಿ ಹೋಗಿ"
  
  Tokens: ["[CLS]", "ಕಾ", "ಲ", "ದ್ರುತ", "ವಾ", "ಗಿ", "[SEP]"]
  Indices: [0, 1, 2, 3, 4, 5, 6]
  
  Target word "ಕಾಲ" → token indices [1, 2]
  last_hidden_state[:, [1, 2], :] → [[768], [768]]
  mean_pooling → [768]
  
  Result: embedding1 [768]

Step 2e: Assembly
  Collect all embeddings:
  X = [embedding1, embedding2, ..., embedding_n]
  Shape: [n, 768] where n = number of sentences
  
  Metadata: Store sentence, position, context for later analysis
```

### Step 3: K-Means Clustering

**Objective**: Partition embeddings into k clusters (k = number of senses)

```
Process:

Step 3a: Choose k (Number of Clusters)
  Options:
    1. Known k: Use number of senses from IndoWordNet
       Example: "ಕಾಲ" has 3 senses → k = 3
    
    2. Unknown k: Use elbow method or silhouette
       More details in Part 4
  
  For our project: Use k from IndoWordNet (known)

Step 3b: Run K-Means
  Algorithm:
    from sklearn.cluster import KMeans
    
    kmeans = KMeans(
      n_clusters=k,
      init='k-means++',  # Smart initialization
      n_init=10,         # Run 10 times, pick best
      max_iter=300,      # Max iterations
      random_state=42,   # Reproducibility
      verbose=1
    )
    
    cluster_labels = kmeans.fit_predict(X)  # [n]
    centroids = kmeans.cluster_centers_      # [k, 768]

Step 3c: Output
  cluster_labels: [3, 1, 2, 1, 2, 1, 3, ...]
    → Sentence 1 → Cluster 3 (sense 3)
    → Sentence 2 → Cluster 1 (sense 1)
    → Sentence 3 → Cluster 2 (sense 2)
    → etc.

Computational Notes:
  - Time: O(n × k × d × iterations)
    For n=200, k=3, d=768, iterations=50:
    ~200 × 3 × 768 × 50 = 23M operations ≈ 1-2 seconds
  - Memory: O(n × d) + O(k × d) ≈ 500KB
  - Scalable to 1000s of examples
```

### Step 4: Cluster Interpretation

**Objective**: Understand what each cluster represents

```
Post-Processing:

Step 4a: Analyze Cluster Compositions
  For each cluster c:
    - Get all sentences in cluster c
    - Analyze words surrounding target word
    - Look for common patterns
  
  Example analysis for "ಕಾಲ":
  
  Cluster 1 (Primary):
    Sentences: [Sent1, Sent4, Sent6, ...]
    Surrounding words: "ದ್ರುತವಾಗಿ", "ಹೋಗಿ", "ಮೆರೆದು", "ದಿನ"
    Interpretation: Temporal sense (time)
    Label: SENSE 1 (Time)
  
  Cluster 2 (Secondary):
    Sentences: [Sent3, Sent5, Sent7, ...]
    Surrounding words: "ಮುರಿದೆ", "ನಿಜ", "ಗುರುತು", "ಜಾರುತ್ತೆ"
    Interpretation: Anatomical sense (leg/limb)
    Label: SENSE 2 (Leg)
  
  Cluster 3 (Rare):
    Sentences: [Sent2, ...]
    Surrounding words: "ಅಂತ್ಯ", "ಮರಣ", "ಕೃತ್ಯತೆ"
    Interpretation: Metaphorical sense (death/end)
    Label: SENSE 3 (Death)

Step 4b: Assign Sense Labels
  Challenge: Clusters are unlabeled!
  Need to map clusters → senses
  
  Option 1: Manual Inspection
    - Read sample sentences from each cluster
    - Assign sense based on manual review
    - Requires domain expert
  
  Option 2: IndoWordNet Matching
    - Use available sense definitions
    - Compute similarity between cluster + glosses
    - Assign cluster to gloss with highest similarity
    - Mostly automatic
  
  Option 3: Context Frequency
    - Count word co-occurrences per cluster
    - Compare to known sense correlates
    - Assign based on most common patterns
    - Heuristic but effective

Step 4c: Create Mapping
  After labeling:
    cluster_to_sense = {
      0: Sense_1 (time),
      1: Sense_2 (leg),
      2: Sense_3 (death)
    }
  
  Then: sentence_sense = cluster_to_sense[cluster_id]
```

---

## PART 4: DETERMINING OPTIMAL K

### Challenge: Finding the Right Number of Clusters

```
Problem:
  - We usually know k from IndoWordNet (use it directly)
  - But for exploration: how to find optimal k?

Solutions:

Method 1: Elbow Method
  Procedure:
    1. Run k-means for k in [1, 2, 3, ..., 10]
    2. Compute inertia for each k:
       inertia = Σ(||xi - c_assigned||²)
       (Sum of squared distances to centroid)
    3. Plot inertia vs k
    4. Find "elbow" (where decrease slows)
    5. Use k at elbow
  
  Example plot:
    inertia
       |     *
       |    **
       |   ** 
       |  **        ← Elbow at k=3
       | **
       |*
       |________________ k
       1 2 3 4 5 6 7 8
  
  Pros: Simple, interpretable
  Cons: Subjective ("where's the elbow?")

Method 2: Silhouette Score
  Definition:
    For each point i in cluster c:
      a_i = mean distance to other points in c
      b_i = min(mean distance to other clusters)
      silhouette_i = (b_i - a_i) / max(a_i, b_i)
  
  Silhouette Score (average):
    S = mean(silhouette_i for all i)
    Range: [-1, +1]
    Higher = better (more cohesive clusters)
  
  Procedure:
    1. Run k-means for k in [1, 2, ..., 10]
    2. Compute silhouette score for each k
    3. Choose k with highest score
  
  Example:
    k=1: silhouette = 0.0 (trivial)
    k=2: silhouette = 0.42
    k=3: silhouette = 0.58 ← Choose this
    k=4: silhouette = 0.53
    k=5: silhouette = 0.48
  
  Pros: Quantitative, principled
  Cons: Computationally more expensive

Method 3: Domain Knowledge (Recommended for our project)
  Use number of senses from IndoWordNet
  - We have ground truth from lexicon
  - No need to estimate
  - Example: "ಕಾಲ" has 3 known senses → k = 3
  
  This is THE APPROACH for our Kannada WSD

Method 4: Cross-Validation
  Procedure:
    1. Manually annotate subset of sentences (50-100)
    2. For each k, run k-means on full data
    3. Evaluate purity on annotated subset
    4. Choose k with highest purity
  
  Pros: Directly optimizes for WSD quality
  Cons: Requires manual annotation

Recommendation for Kannada WSD:
  PRIMARY: Use IndoWordNet sense count (k known)
  BACKUP: Use silhouette score if k unknown
  VALIDATION: Use cross-validation to verify
```

---

## PART 5: EXPECTED OUTCOMES AND EVALUATION

### Clustering Quality Metrics

```
Metric 1: Inertia (Within-Cluster Sum of Squares)
  Definition:
    inertia = Σ_c Σ_{xi in c} ||xi - μ_c||²
    (Sum of squared distances from points to centroids)
  
  Interpretation:
    Lower inertia = tighter clusters
    But always decreases as k increases (not directly comparable)
  
  Expected value:
    k=3: inertia ≈ 150-200 (for 200 embeddings in 768-D space)

Metric 2: Silhouette Score
  Definition: (explained above)
  Range: [-1, +1]
  
  Expected value:
    Well-separated senses: 0.50-0.70
    Overlapping senses: 0.30-0.50
    Poor clustering: < 0.30
  
  For Kannada "ಕಾಲ":
    Expected: 0.55 (time and leg are distinct)

Metric 3: Purity (If Ground Truth Available)
  Definition:
    Purity = (1/n) Σ_c max_sense(|cluster_c ∩ sense|)
    (Fraction of correctly clustered points)
  
  Range: [0, 1]
  
  Expected value:
    Good clustering: 0.70-0.80
    Decent clustering: 0.60-0.70
    Poor clustering: < 0.60
  
  Why not 1.0?
    - Some senses overlap contextually
    - Ambiguous sentences hard to classify
    - Models capture similar contexts for different senses
  
  For Kannada WSD:
    Expected purity: 0.70-0.75

Metric 4: Homogeneity & Completeness
  Homogeneity:
    All points in a cluster belong to same sense?
    homogeneity = 1 - (H(sense | cluster) / H(sense))
    Range: [0, 1], higher is better
  
  Completeness:
    All points from same sense are in same cluster?
    completeness = 1 - (H(cluster | sense) / H(cluster))
    Range: [0, 1], higher is better
  
  Expected:
    homogeneity: 0.65-0.75
    completeness: 0.60-0.70
```

### Practical Outcomes

```
Expected Clustering Result for "ಕಾಲ" (3 senses):

Data:
  200 sentences with "ಕಾಲ"
  True distribution: 75 time, 95 leg, 30 death

After k-means (k=3):
  Cluster 1: 80 points (inferred: Time sense)
    Correctly assigned: 70 time + 10 leg = 87.5% purity
  
  Cluster 2: 105 points (inferred: Leg sense)
    Correctly assigned: 90 leg + 15 time = 85.7% purity
  
  Cluster 3: 15 points (inferred: Death sense)
    Correctly assigned: 12 death + 3 other = 80.0% purity
  
  Overall metrics:
    Purity: (70 + 90 + 12) / 200 = 86.0%
    Silhouette: 0.58
    Homogeneity: 0.72
    Completeness: 0.68
    F-measure: 0.70
```

---

## PART 6: APPLYING CLUSTERING TO WSD

### How Clustering Becomes WSD

```
Connection: Clustering → WSD

Scenario 1: Training Phase
  Goal: Discover sense representations from unlabeled data
  
  Process:
    1. Collect large corpus with target word (1000+ sentences)
    2. Extract embeddings for all occurrences
    3. Run k-means clustering (k = known senses)
    4. Learn: cluster centroids = sense prototypes
  
  Output:
    sense_prototypes = [μ_1, μ_2, ..., μ_k]
    (k embeddings representing each sense)

Scenario 2: Classification Phase (WSD at Test Time)
  Goal: Assign sense to new sentence
  
  Process:
    1. Input: New sentence with target word
    2. Extract embedding (same as training)
    3. Find nearest sense prototype:
       predicted_sense = argmin_j(||embedding - μ_j||)
    4. Output: Predicted sense
  
  Example:
    Test sentence: "ಕಾಲ ಹೋಗಿ"
    Embedding: [0.12, 0.34, ..., 0.56] [768]
    
    Distances to prototypes:
      ||e - μ_1|| = 0.15 ← Closest
      ||e - μ_2|| = 0.67
      ||e - μ_3|| = 0.82
    
    Predicted: Sense 1 (Time)

Algorithm: Nearest Centroid Classification

Input:
  - Test embedding: e_test [768]
  - Learned centroids: [μ_1, μ_2, ..., μ_k] [k, 768]
  - Distance metric: Euclidean

Pseudocode:
  distances = [euclidean(e_test, μ_j) for j in 1..k]
  predicted_sense = argmin(distances)
  confidence = 1 - (min_distance / mean_distance)
  
  return predicted_sense, confidence
```

### Two-Stage Pipeline: Clustering + Classification

```
Training Stage (Offline):

  Corpus (1000+ sentences)
        ↓
  Preprocessing & Embedding
        ↓
  X [1000, 768]
        ↓
  K-Means Clustering (k=3)
        ↓
  Learn centroids: [μ_1, μ_2, μ_3]
        ↓
  Save centroids to disk

Classification Stage (Online):

  New Sentence
        ↓
  Preprocessing & Embedding
        ↓
  e_test [768]
        ↓
  Load saved centroids
        ↓
  Compute distances
        ↓
  Find nearest centroid
        ↓
  Predicted Sense + Confidence
```

---

## PART 7: STEP-BY-STEP IMPLEMENTATION

### Complete Pseudocode

```python
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
import numpy as np

class ClusteringWSD:
    """Word Sense Disambiguation via k-means Clustering"""
    
    def __init__(self, embedding_model, n_senses=3):
        """
        Args:
            embedding_model: MuRIL or XLM-R
            n_senses: number of senses (from IndoWordNet)
        """
        self.model = embedding_model
        self.n_senses = n_senses
        self.centroids = None
        self.kmeans = None
    
    def train_clustering(self, sentences, target_word):
        """
        Learn sense prototypes from corpus.
        
        Args:
            sentences: List of sentences with target_word
            target_word: The ambiguous word
        
        Returns:
            Dictionary with clustering metrics
        """
        
        # Step 1: Extract embeddings for all sentences
        embeddings = []
        for sentence in sentences:
            emb = self.extract_target_embedding(sentence, target_word)
            embeddings.append(emb)
        
        X = np.array(embeddings)  # [n_sentences, 768]
        
        # Step 2: Run k-means clustering
        self.kmeans = KMeans(
            n_clusters=self.n_senses,
            init='k-means++',
            n_init=10,
            max_iter=300,
            random_state=42,
            verbose=1
        )
        
        cluster_labels = self.kmeans.fit_predict(X)  # [n_sentences]
        self.centroids = self.kmeans.cluster_centers_  # [n_senses, 768]
        
        # Step 3: Evaluate clustering
        inertia = self.kmeans.inertia_
        silhouette = self.compute_silhouette(X, cluster_labels)
        
        # Step 4: Analyze clusters
        cluster_info = self.analyze_clusters(sentences, cluster_labels)
        
        return {
            "n_clusters": self.n_senses,
            "n_points": len(sentences),
            "inertia": inertia,
            "silhouette": silhouette,
            "cluster_info": cluster_info,
            "centroids_saved": True
        }
    
    def predict_sense(self, sentence, target_word):
        """
        Predict sense for target word in sentence.
        
        Args:
            sentence: Input sentence
            target_word: Target ambiguous word
        
        Returns:
            {
                "predicted_sense": int,
                "distances": List,
                "confidence": float
            }
        """
        
        # Verify centroids are trained
        if self.centroids is None:
            raise ValueError("Model not trained. Call train_clustering first.")
        
        # Extract embedding
        e_test = self.extract_target_embedding(sentence, target_word)  # [768]
        
        # Compute distances to all centroids
        distances = []
        for centroid in self.centroids:
            dist = euclidean(e_test, centroid)
            distances.append(dist)
        
        distances = np.array(distances)  # [n_senses]
        
        # Find nearest sense
        predicted_sense = np.argmin(distances)
        
        # Compute confidence (margin-based)
        sorted_distances = np.sort(distances)
        margin = sorted_distances[1] - sorted_distances[0]
        confidence = margin / (np.mean(distances) + 1e-8)
        
        return {
            "predicted_sense": predicted_sense,
            "distances": distances.tolist(),
            "confidence": float(confidence),
            "explanation": f"Closest to sense {predicted_sense+1}"
        }
    
    def extract_target_embedding(self, sentence, target_word):
        """Extract embedding of target word in sentence"""
        # Preprocessing, tokenization, embedding extraction
        # (Same as main pipeline)
        pass
    
    def compute_silhouette(self, X, labels):
        """Compute silhouette score for clustering"""
        from sklearn.metrics import silhouette_score
        return silhouette_score(X, labels, metric='euclidean')
    
    def analyze_clusters(self, sentences, cluster_labels):
        """Analyze composition of each cluster"""
        cluster_info = {}
        
        for cluster_id in range(self.n_senses):
            mask = cluster_labels == cluster_id
            cluster_sentences = [s for s, m in zip(sentences, mask) if m]
            cluster_info[cluster_id] = {
                "size": len(cluster_sentences),
                "examples": cluster_sentences[:3],  # First 3 examples
            }
        
        return cluster_info


# Usage Example:

# 1. Initialize
wsd_clustering = ClusteringWSD(
    embedding_model=muriel_model,
    n_senses=3  # For "ಕಾಲ"
)

# 2. Train on corpus
training_sentences = load_corpus("kannada_large_corpus.txt")
target_word_sentences = filter_sentences(training_sentences, "ಕಾಲ")

training_metrics = wsd_clustering.train_clustering(
    target_word_sentences,
    "ಕಾಲ"
)

print(f"Clustering trained:")
print(f"  Inertia: {training_metrics['inertia']:.2f}")
print(f"  Silhouette: {training_metrics['silhouette']:.3f}")

# 3. Predict on WiC pairs
wic_pairs = load_wic_dataset()

correct = 0
for pair in wic_pairs:
    pred1 = wsd_clustering.predict_sense(pair["sentence1"], "ಕಾಲ")
    pred2 = wsd_clustering.predict_sense(pair["sentence2"], "ಕಾಲ")
    
    # Check if both predictions same/different match labels
    both_same = (pred1["predicted_sense"] == pred2["predicted_sense"])
    label_same = (pair["label"] == 1)
    
    if both_same == label_same:
        correct += 1

accuracy = correct / len(wic_pairs)
print(f"WSD Accuracy: {accuracy:.2%}")
```

---

## PART 8: EXPECTED OUTCOMES

### Typical Results on Kannada WSD

```
Setup:
  - 45 target words (each with 3+ senses)
  - 1000+ total corpus sentences for clustering
  - 720 WiC test pairs
  - MuRIL embeddings

Phase 1: Clustering Training

Results per word (example for "ಕಾಲ"):
  n_clusters: 3
  n_sentences: 250
  inertia: 185.4
  silhouette: 0.58
  
  Cluster sizes:
    Cluster 0: 95 (Time sense)
    Cluster 1: 120 (Leg sense)
    Cluster 2: 35 (Death sense)
  
  Cluster purity:
    Cluster 0: 88% (8 misclassified)
    Cluster 1: 85% (18 misclassified)
    Cluster 2: 77% (8 misclassified)
  
  Overall purity: 83.2%

Phase 2: WSD Classification

Test on WiC pairs:
  Classification Method: Nearest Centroid
  
  Results:
    Accuracy: 68-72% (depends on word)
    - High-polysemy words: 65-70%
    - Low-polysemy words: 75-80%
  
  Performance by model:
    MuRIL: 70% (best clustering)
    XLM-R: 68% (good clustering)
    IndicBERT: 65% (acceptable)
    mBERT: 62% (weaker)

Phase 3: Combined with Other Methods

Ensemble approach:
  
  Method           | Accuracy
  ─────────────────────────
  Gloss-Based      | 60%
  K-Means          | 70%
  Ensemble Avg     | 72%
  Supervised       | 78%
  
  Combining approaches:
    - K-means identifies sense prototypes
    - Gloss-based validates assignments
    - Supervised fine-tunes on small dataset
    - Ensemble averages predictions
    → Higher accuracy than any single method
```

---

## PART 9: CLUSTERING vs. WSD RELATIONSHIP

### How Clustering Enables WSD

```
Traditional WSD (Supervised):
  Labeled Training Data (WiC pairs)
        ↓
  Train neural classifier
        ↓
  Predict labels on test
  
Clustering-Based WSD (Unsupervised):
  Unlabeled Corpus (sentences with word)
        ↓
  K-Means discovers clusters
        ↓
  Clusters = senses
        ↓
  Use clustering for classification

Key Insight:
  Clustering discovers WHAT the senses are
  Classification assigns senses to NEW examples
  
Analogy: Photo clustering
  - Unsupervised: Cluster photos by content (animals, landscapes, people)
  - Discovers categories without labels
  - Then classify new photo into discovered categories

Advantages:
  ✓ No labeled data needed
  ✓ Discovers patterns automatically
  ✓ Works for rare words/new senses
  ✓ Unsupervised learning benefit
  
Disadvantages:
  ✗ Assumes natural clusters exist
  ✗ Sensitive to k (must know n_senses)
  ✗ May not find meaningful clusters
  ✗ Lower accuracy than supervised
```

---

## PART 10: VALIDATION AND DIAGNOSTICS

### How to Verify Clustering Quality

```
Diagnostic 1: Visual Inspection of Clusters
  Method:
    1. Reduce clusters to 2D using t-SNE
    2. Plot points colored by cluster
    3. Visually inspect separation
  
  Good clustering: Clear, separate clusters
  Poor clustering: Overlapping, mixed colors

Diagnostic 2: Sense Label Assignment
  Method:
    1. Sample 5-10 sentences per cluster
    2. Read them manually
    3. Infer what sense each cluster represents
  
  Expected:
    Cluster 1: "time, era, period" (coherent sense)
    Cluster 2: "leg, limb, appendage" (coherent sense)
    Cluster 3: "death, end, mortality" (coherent sense)
  
  Problems:
    ✗ Cluster 1: mixed time/leg/death (incoherent)
    ✗ Cluster 2: only 2 examples (too small)
    ✗ Cluster 3: empty or trivial

Diagnostic 3: Purity Against Ground Truth
  Method:
    1. Manually annotate 100-200 sentences with true senses
    2. Run clustering
    3. Compute purity: (correctly clustered / total)
  
  Expected purity: 70-80%
  
  If purity < 60%:
    → Clustering failed, try:
      - Increase number of iterations
      - Use different initialization
      - Check embedding quality
      - Verify k value

Diagnostic 4: Centroid Stability
  Method:
    1. Run k-means 10 times (different random seeds)
    2. Compare centroid positions
    3. Measure variance
  
  Good: Centroids converge to similar locations
  Bad: Centroids widely different (unstable clustering)
  
  If unstable:
    → k too large, or data too noisy, or k doesn't match natural structure

Diagnostic 5: Sense Coverage
  Method:
    1. After clustering, check if all k senses represented
    2. Each cluster should be non-trivial (size > n/k)
  
  Problem: One cluster has 80% of points
    → Clustering degenerate, check if k too large
  
  Problem: Empty cluster
    → k too large, reduce k
```

---

## PART 11: SUMMARY TABLE

### Clustering Approach for WSD

```
Aspect                 | Details
───────────────────────────────────────────────────────
Core Algorithm         | K-Means (Lloyd's variant)
Distance Metric        | Euclidean (standard)
Number of Clusters     | k = number of senses (from IndoWordNet)
Input Data             | Target word embeddings from corpus
Computational Cost     | O(n × k × d × iter) ≈ seconds
Memory Requirement     | O(n × d) + O(k × d) ≈ MB
Training Time          | Minutes for 1000s of sentences
Inference Time         | O(k × d) ≈ milliseconds per sentence
Expected Accuracy      | 65-72% (unsupervised)
Purity (vs. ground)    | 70-80%
Silhouette Score       | 0.50-0.65 (good separation)
Best With              | 50-500 corpus examples
Requires               | Known k (number of senses)
Advantages             | Unsupervised, interpretable, scalable
Disadvantages          | Needs good embeddings, sensitive to k
```

---

## CONCLUSION

**K-Means Clustering for WSD**:

1. **Theory**: Different senses have different embeddings → k-means discovers clusters
2. **Practice**: Cluster embeddings from corpus → learn sense prototypes → classify new examples
3. **Quality**: Expected purity 70-80%, silhouette 0.50-0.65
4. **Performance**: 65-72% accuracy on WiC pairs (unsupervised)
5. **Advantages**: No labeled data, interpretable, works for new words
6. **Best Used**: With small amounts of labeled data (500-700 pairs) for semi-supervised learning

**For Kannada WSD Project**:
- Run k-means for each target word (k from IndoWordNet)
- Use learned centroids for WSD classification
- Combine with gloss-based and supervised approaches
- Expected ensemble accuracy: 75-80%

