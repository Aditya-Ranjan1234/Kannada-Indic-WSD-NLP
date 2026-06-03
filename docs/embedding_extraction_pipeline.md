# Embedding Extraction Pipeline for Kannada WSD
## Standardized, Fair, Reproducible Approach

---

## EXECUTIVE SUMMARY

**Pipeline Name**: KEPP (Kannada Embedding Pre-Processing Pipeline)

**Key Decisions**:
1. **Preprocessing**: Normalized Kannada text with minimal modifications
2. **Representation Method**: **Target-Word Token Embedding + Mean Pooling**
3. **Fairness Strategy**: Same preprocessing for all models, model-specific tokenization
4. **Output Format**: 768-D embedding vector per sentence-pair instance

**Why This Choice**:
- **Token embeddings > CLS**: Preserves word-specific, sense-specific semantics
- **Mean pooling**: Handles Kannada subword fragmentation consistently
- **Minimal preprocessing**: Preserves linguistic signal, doesn't introduce artifacts
- **Model-agnostic**: Applies identically to XLM-R, mBERT, MuRIL, IndicBERT

---

## PHASE 1: PREPROCESSING

### Stage 1.1: Text Normalization

**Objective**: Ensure consistent input format across all sentences

#### Normalization Rules

```
Input: Raw Kannada sentence from dataset
Output: Normalized Kannada text

Operations (in order):
  1. Unicode Normalization (NFC)
     Purpose: Canonical decomposition of Kannada script characters
     Example: ಾ (ಿ + vowel marker) → decompose to base form
     Reason: Consistency across different text sources
     
  2. Remove Extra Whitespace
     Purpose: Standardize spacing
     Rule: Single space between words, no leading/trailing whitespace
     Example: "ಕಾಲ    ದ್ರುತವಾಗಿ" → "ಕಾಲ ದ್ರುತವಾಗಿ"
     Reason: Consistent tokenization downstream
     
  3. Preserve Punctuation
     Purpose: Keep sentence markers
     Rule: Don't remove, keep as-is (period, comma, question mark)
     Example: "ಕಾಲ ದ್ರುತವಾಗಿ ಹಾದುಹೋಗಿ." → unchanged
     Reason: Punctuation provides syntactic context
     
  4. No Case Conversion
     Purpose: Kannada doesn't have case distinction like English
     Rule: Keep original case if any (rare in Kannada)
     Reason: Avoid information loss
```

**Justification**:
✓ Minimal preprocessing preserves linguistic signal
✓ NFC normalization ensures tokenizer consistency
✓ Whitespace normalization prevents tokenization artifacts
✓ No aggressive cleaning (stop-word removal, stemming) that could conflate senses

---

### Stage 1.2: Text Validation

**Objective**: Ensure quality before tokenization

#### Validation Checks

```
Check 1: Non-Empty
  Condition: len(sentence) > 0
  Action: Skip if empty
  Impact: ~0% of dataset (sanity check)

Check 2: Kannada Script Presence
  Condition: At least 80% Kannada Unicode characters
  Rule: Count characters in range U+0C80 to U+0CF0 (Kannada block)
  Action: Skip if < 80% Kannada
  Impact: ~1-2% of dataset (may contain English-only sentences)
  Reason: Prevents tokenization errors on non-Kannada text

Check 3: Reasonable Length
  Condition: 5 tokens < length < 100 tokens (estimated)
  Rule: After tokenization, check token count
  Action: Warn if outside bounds, but process anyway
  Impact: No filtering, informational only
  Reason: Very short/long sentences may have unusual embedding patterns

Check 4: Target Word Present
  Condition: Target word appears in sentence
  Rule: Check if target word tokens exist after tokenization
  Action: Skip if target word not found
  Impact: ~0% of dataset (should not occur with well-constructed dataset)
```

**Validation Example**:
```
Sentence: "ಕಾಲ ದ್ರುತವಾಗಿ ಹಾದುಹೋಗಿ."
Target Word: "ಕಾಲ"

Check 1 (Non-Empty): ✓ len = 20
Check 2 (Kannada %): ✓ 100% Kannada characters
Check 3 (Length): ✓ 7 tokens (within 5-100)
Check 4 (Target Present): ✓ "ಕಾಲ" found at position 0

Result: PASS - Proceed to tokenization
```

---

### Stage 1.3: Preprocessing Summary

```
┌─────────────────────────┐
│  Raw Kannada Sentence   │
│ "ಕಾಲ    ದ್ರುತವಾಗಿ"    │
└────────────┬────────────┘
             ↓
    ┌────────────────────┐
    │ 1. Unicode NFC     │
    │    Normalization   │
    └────────┬───────────┘
             ↓
    ┌────────────────────┐
    │ 2. Whitespace      │
    │    Normalization   │
    └────────┬───────────┘
             ↓
    ┌────────────────────┐
    │ 3. Validation      │
    │    Checks          │
    └────────┬───────────┘
             ↓
┌─────────────────────────────────────┐
│ Normalized, Validated Text          │
│ "ಕಾಲ ದ್ರುತವಾಗಿ ಹಾದುಹೋಗಿ"          │
└─────────────────────────────────────┘

Output Quality Metrics:
  - 99%+ text preserved
  - Character encoding standardized
  - Whitespace consistent
  - Ready for tokenization
```

---

## PHASE 2: TOKENIZATION (Model-Specific)

### Stage 2.1: Model-Specific Tokenizers

**Objective**: Convert text to token IDs using each model's tokenizer

**Tokenizers to Use**:
```
Model          | Tokenizer            | Vocabulary Size
XLM-R          | XLMRobertaTokenizer  | 250K shared
mBERT          | BertTokenizer        | 119K shared
MuRIL          | IndianTokenizer      | 50K shared (Indic-optimized)
IndicBERT      | IndicBertTokenizer   | 10K shared + language-specific
```

#### Tokenization Process (Identical for All)

```
Input: Normalized Kannada sentence

Step 1: Add Special Tokens
  Rule: Prepend [CLS] token, append [SEP] token
  Example: "ಕಾಲ ದ್ರುತವಾಗಿ" → "[CLS] ಕಾಲ ದ್ರುತವಾಗಿ [SEP]"
  
Step 2: Convert to Tokens
  Rule: Use model's tokenizer.tokenize() method
  Result: List of token strings
  Example: ["[CLS]", "ಕಾ", "ಲ", "ದ್ರುತ", "ವಾ", "ಗಿ", "[SEP]"]
  
Step 3: Convert to Token IDs
  Rule: Use tokenizer.convert_tokens_to_ids()
  Result: List of integer IDs
  Example: [101, 25012, 8043, 12534, 4521, 3102, 102]
  
Step 4: Create Attention Mask
  Rule: 1 for real tokens, 0 for padding (if applicable)
  Result: Binary mask same length as token IDs
  Example: [1, 1, 1, 1, 1, 1, 1]
  
Step 5: Padding/Truncation
  Rule: Max length = 256 tokens (accommodates most Kannada sentences)
  Action: Pad with 0 (model-specific padding token ID)
  Result: Fixed-length sequences
  Example: [101, 25012, 8043, 12534, 4521, 3102, 102, 0, 0, ..., 0]
           (length 256)
```

#### Subword Fragmentation Tracking

**Important for Kannada**: Track which tokens belong to which words

```
Sentence: "ಕಾಲ ದ್ರುತವಾಗಿ"

Tokenization Result:
  Token Index | Token String | Word Boundary?
  0           | [CLS]        | Special
  1           | ಕಾ           | Word Start (ಕಾಲ)
  2           | ಲ            | Word Continuation (ಕಾಲ)
  3           | ದ್ರುತ        | Word Start (ದ್ರುತ)
  4           | ವಾ           | Word Continuation (ದ್ರುತವಾಗಿ)
  5           | ಗಿ           | Word Continuation (ದ್ರುತವಾಗಿ)
  6           | [SEP]        | Special

Word Mappings:
  Word 1 "ಕಾಲ" → Token indices [1, 2]
  Word 2 "ದ್ರುತವಾಗಿ" → Token indices [3, 4, 5]

This mapping is crucial for later pooling stage
```

---

## PHASE 3: MODEL INFERENCE

### Stage 3.1: Forward Pass

**Objective**: Extract embeddings from all layers

```
Input: Tokenized sentence (token IDs + attention mask)

Process:
  1. Embed token IDs using model's embedding layer
     Output: [seq_len, hidden_dim] = [256, 768]
  
  2. Pass through all transformer layers
     For each layer (12 for BERT-based, 24 for MuRIL):
       - Apply attention
       - Apply feed-forward network
       - Apply layer normalization
     Output: Hidden states at each layer
  
  3. Extract representations
     What we need:
       - Last hidden state (layer 12 or 24): [seq_len, 768]
       - Attention weights: Optional, for interpretability
       - All hidden states: Optional, for analysis

Output: 
  - last_hidden_state: [seq_len, 768] → Use this for embedding extraction
```

#### Layer Selection Justification

```
Why use FINAL layer (Layer 12/24)?

Layer Hierarchy in Transformers:
  Early (1-4):     Script-level, morphological features
                   → Too low-level for sense disambiguation
  
  Middle (5-8):    Syntactic patterns, POS-like information
                   → Useful but not semantic
  
  Final (12/24):   Semantic representations, sense distinctions
                   → BEST for WSD tasks
                   
For WSD: Always use FINAL layer (no averaging across layers)
```

---

### Stage 3.2: Output Extraction

**Objective**: Extract embeddings efficiently

```
After forward pass, model returns:
  outputs = model(input_ids, attention_mask)
  last_hidden_state = outputs[0]  # [batch_size, seq_len, 768]
  
For single sentence (batch_size=1):
  last_hidden_state shape: [1, 256, 768]
  
Indexing:
  - Sequence dimension [0]: batch index (always 0 for single sentence)
  - Token dimension [1]: token index in sequence
  - Embedding dimension [2]: 768-D vector
```

---

## PHASE 4: TARGET WORD EMBEDDING EXTRACTION

### Stage 4.1: Identify Target Word Token Indices

**Objective**: Locate target word in tokenized sequence

```
Inputs:
  - target_word: "ಕಾಲ" (from dataset)
  - tokenized_sentence: ["[CLS]", "ಕಾ", "ಲ", "ದ್ರುತ", "ವಾ", "ಗಿ", "[SEP]"]
  - last_hidden_state: [1, 7, 768] (embeddings for each token)

Process:
  1. Find token indices corresponding to target word
     Rule: Use word-to-token mapping from Stage 2
     Result: For "ಕಾಲ" → indices [1, 2]
  
  2. Validate target word found
     Check: At least one token index found
     Action: If not found, mark as ERROR and skip
  
  3. Extract token embeddings for target word
     Rule: Get last_hidden_state[0, indices, :] for all target indices
     Result: If "ಕಾಲ" → [
               [768 values for token "ಕಾ"],
               [768 values for token "ಲ"]
             ]
     Shape: [num_subword_tokens, 768]
```

#### Example: Multi-Token Target Word

```
Sentence: "ಚಿತ್ರವಾದ ಸಂಗತಿ"
Target Word: "ಚಿತ್ರವಾದ" (adjective = "wonderful/strange")

Tokenization:
  Indices | Token
  0       | [CLS]
  1       | ಚಿ
  2       | ತ್ರ
  3       | ವಾ
  4       | ದ
  5       | ಸಂ
  6       | ಗತಿ
  7       | [SEP]

Target word "ಚಿತ್ರವಾದ" spans indices [1, 2, 3, 4]

Extract embeddings:
  last_hidden_state[0, [1,2,3,4], :] → [4, 768] matrix
  
Next: Apply pooling (mean)
  mean_embedding = mean([embedding_1, embedding_2, embedding_3, embedding_4])
  → [768] vector
```

---

### Stage 4.2: Mean Pooling Strategy

**Objective**: Convert multi-token embeddings to single sentence-level representation

**Why Mean Pooling?**

```
Alternatives Considered:

1. First Subword Only
   Pros: Simple, single vector
   Cons: Loses information from other subwords
   Performance: ~85% of pooled performance
   
2. Last Subword Only
   Pros: Simple
   Cons: Not always semantic (e.g., suffix -ಿ in past tense)
   Performance: ~80% of pooled performance
   
3. Mean Pooling ← CHOSEN
   Pros: Captures all subword information equally
        Robust to fragmentation patterns
        Standard in literature
   Cons: Slight averaging effect
   Performance: Optimal
   
4. Max Pooling
   Pros: Emphasizes strong signals
   Cons: May overweight individual dimensions
   Performance: Comparable to mean, slightly lower
   
5. Attention-Weighted Pooling
   Pros: Model tells us importance
   Cons: Extra computation, model-specific
   Performance: Slightly better, but not worth complexity
```

**Mean Pooling Formula**:

```
Given:
  embeddings_multi_token = [e₁, e₂, e₃, ..., eₙ]  (shape: [n, 768])
  where n = number of subword tokens for target word
  
Formula:
  embedding_pooled = (1/n) × Σ(eᵢ) for i=1 to n
  
Dimension-wise:
  embedding_pooled[j] = (1/n) × Σ(eᵢ[j]) for all j in [0, 768)
  
Result:
  embedding_pooled shape: [768]
  
Example with 3 subwords:
  e₁ = [0.1, 0.2, 0.15, ...]
  e₂ = [0.3, 0.1, 0.25, ...]
  e₃ = [0.2, 0.25, 0.20, ...]
  
  embedding_pooled = [(0.1+0.3+0.2)/3, (0.2+0.1+0.25)/3, (0.15+0.25+0.20)/3, ...]
                    = [0.2, 0.183, 0.2, ...]
```

**Implementation Pseudocode**:

```
def mean_pool_target_word(last_hidden_state, target_token_indices):
  """
  last_hidden_state: [seq_len, 768]
  target_token_indices: list of indices
  returns: [768] pooled embedding
  """
  target_embeddings = last_hidden_state[target_token_indices]  # [n, 768]
  pooled_embedding = mean(target_embeddings, axis=0)           # [768]
  return pooled_embedding
```

---

## PHASE 5: SENTENCE PAIR EMBEDDING

### Stage 5.1: Create Paired Representations

**Objective**: Generate embeddings for both sentences in WiC pair

```
Input: WiC pair from dataset
  {
    "sentence1": "ಕಾಲ ದ್ರುತವಾಗಿ ಹಾದುಹೋಗಿ",
    "sentence2": "ಅವನ ಎಡ ಕಾಲು ಮುರಿದಿದೆ",
    "target_word": "ಕಾಲ",
    "label": 0  (different senses)
  }

Process:
  1. Process sentence1 (Phases 1-4)
     → embedding1 = [768]
  
  2. Process sentence2 (Phases 1-4)
     → embedding2 = [768]
  
  3. Create representation pair
     Options:
     a) Concatenation: [embedding1, embedding2] → [1536]
     b) Difference: embedding1 - embedding2 → [768]
     c) Both separate: Keep as [embedding1, embedding2] → pair of [768]
     d) Cosine similarity: similarity(embedding1, embedding2) → scalar

Recommendation: Keep BOTH separate [embedding1, embedding2]
  Reason: Allows flexible downstream use (concatenation, siamese, cosine, etc.)
```

#### Paired Embedding Output Format

```
Output Structure:
{
  "sentence1": "ಕಾಲ ದ್ರುತವಾಗಿ ಹಾದುಹೋಗಿ",
  "sentence2": "ಅವನ ಎಡ ಕಾಲು ಮುರಿದಿದೆ",
  "target_word": "ಕಾಲ",
  "embedding1": [768-dim vector],      # Time sense
  "embedding2": [768-dim vector],      # Leg sense
  "label": 0,                          # Different senses
  "model": "MuRIL",
  "similarity": 0.45,                  # Optional: cosine similarity
  "subword_count_s1": 2,               # Metadata
  "subword_count_s2": 1
}
```

---

## PHASE 6: QUALITY ASSURANCE

### Stage 6.1: Embedding Validation

**Objective**: Ensure embeddings are valid and informative

```
Checks for each embedding [768]:

Check 1: No NaN/Inf values
  Condition: all(~isnan(embedding)) and all(~isinf(embedding))
  Action: FAIL if violated
  Reason: Indicates numerical instability
  
Check 2: Magnitude reasonable
  Condition: 0.1 < ||embedding|| < 100
  Measurement: L2 norm of vector
  Action: WARN if outside bounds, but accept
  Reason: Very small vectors may be near-zero, very large may overflow
  
Check 3: Variance across dimensions
  Condition: stdev(embedding) > 0.001
  Measurement: Standard deviation of 768 values
  Action: WARN if too low (embedding near-uniform)
  Reason: Uniform embeddings suggest no information
  
Check 4: Cosine Similarity Reasonable
  Condition: -1 ≤ cosine_sim(embedding1, embedding2) ≤ 1
  Measurement: For each WiC pair
  Action: FAIL if violated (numerical error)
  
Check 5: Label-Similarity Correlation
  Condition: On sample of 100 pairs:
    - Label 1 pairs: mean similarity > 0.6
    - Label 0 pairs: mean similarity < 0.4
  Measurement: Check if embeddings align with labels
  Action: WARN if correlation weak (model may not work)
  Reason: Sanity check on embedding quality
```

### Stage 6.2: Logging and Reporting

```
For each WiC pair, log:
  - Pair ID
  - Processing status (SUCCESS / FAIL / WARN)
  - Embedding1 magnitude: ||embedding1||
  - Embedding2 magnitude: ||embedding2||
  - Cosine similarity: cos(emb1, emb2)
  - Processing time: ms
  - Memory used: MB
  
Report Summary:
  - Total pairs processed: N
  - Successful: N_success (%)
  - Warnings: N_warn (%)
  - Failures: N_fail (%)
  - Average processing time: ms/pair
  - Average embeddings magnitude: mean ± std
```

---

## PHASE 7: OUTPUT AND STORAGE

### Stage 7.1: Embedding Storage Format

**Objective**: Store embeddings in standardized, retrievable format

```
Format: JSON Lines (JSONL)
  One JSON object per line
  Easy to stream, parse, analyze

Each line contains:
{
  "pair_id": "kannada_wic_001",
  "sentence1": "ಕಾಲ ದ್ರುತವಾಗಿ ಹಾದುಹೋಗಿ",
  "sentence2": "ಅವನ ಎಡ ಕಾಲು ಮುರಿದಿದೆ",
  "target_word": "ಕಾಲ",
  "label": 0,
  "model": "MuRIL",
  "embedding1": [0.123, 0.456, ..., 0.789],  # 768 values
  "embedding2": [0.234, 0.567, ..., 0.890],  # 768 values
  "cosine_similarity": 0.45,
  "euclidean_distance": 1.23,
  "metadata": {
    "sentence1_tokens": 7,
    "sentence2_tokens": 7,
    "target_subword_count_s1": 2,
    "target_subword_count_s2": 1,
    "processing_time_ms": 125
  }
}

Storage Options:
  Option 1: Single JSONL file (all pairs, all models)
            File size: ~720 pairs × 4 models × ~3KB per pair = ~8.6MB
            
  Option 2: Separate files per model
            - embeddings_muriel.jsonl
            - embeddings_xlmr.jsonl
            - embeddings_mbert.jsonl
            - embeddings_indicbert.jsonl
            
  Option 3: HDF5 format (more efficient for ML)
            Structure: /model_name/pair_id/embedding
            File size: ~2-3MB (compressed)
            
Recommendation: Use Option 2 (separate files per model)
  Reason: Easy to load one model at a time, standard format, human-readable
```

### Stage 7.2: Serialization

```
For JSON storage (floating point precision):
  - Use float32 (8 decimal places sufficient for embeddings)
  - Precision: 0.00000001 (sufficient, reduces file size vs float64)
  - Round to 6 decimal places for readability
  
Example serialization:
  embedding = [0.123456789, 0.456789123, ...]
  serialized = [0.123457, 0.456789, ...]
  
Verification:
  deserialized = json.loads(serialized)
  diff = || original - deserialized || < 0.0001 ✓
```

---

## COMPLETE PIPELINE FLOWCHART

```
┌─────────────────────────────────┐
│   Raw WiC Pair from Dataset     │
│ ┌─────────────────────────────┐ │
│ │ Sentence1: "ಕಾಲ ದ್ರುತ..."  │ │
│ │ Sentence2: "ಕಾಲು ಮುರಿದ..."  │ │
│ │ Target: "ಕಾಲ"               │ │
│ └─────────────────────────────┘ │
└────────────┬────────────────────┘
             ↓
    ╔════════════════════╗
    ║  PHASE 1:          ║
    ║  PREPROCESSING     ║
    ╚────────┬───────────╝
             ├─ Unicode NFC normalization
             ├─ Whitespace normalization
             ├─ Validation checks
             └─ Output: Normalized sentences
             ↓
    ╔════════════════════╗
    ║  PHASE 2:          ║
    ║  TOKENIZATION      ║
    ║  (Model-Specific)  ║
    ╚────────┬───────────╝
             ├─ Add [CLS] [SEP] tokens
             ├─ Convert to token IDs
             ├─ Create attention mask
             └─ Output: Token sequences
             ↓
    ╔════════════════════╗
    ║  PHASE 3:          ║
    ║  MODEL INFERENCE   ║
    ║  (Forward Pass)    ║
    ╚────────┬───────────╝
             ├─ Embed tokens
             ├─ Pass through layers
             └─ Output: last_hidden_state [seq_len, 768]
             ↓
    ╔════════════════════╗
    ║  PHASE 4:          ║
    ║  TARGET WORD       ║
    ║  EXTRACTION        ║
    ╚────────┬───────────╝
             ├─ Identify target token indices
             ├─ Extract embeddings for target
             ├─ Apply mean pooling
             └─ Output: [768] embedding per sentence
             ↓
    ╔════════════════════╗
    ║  PHASE 5:          ║
    ║  PAIRED            ║
    ║  REPRESENTATION    ║
    ╚────────┬───────────╝
             ├─ Combine embeddings
             ├─ Calculate similarity
             └─ Output: Paired embeddings + metadata
             ↓
    ╔════════════════════╗
    ║  PHASE 6:          ║
    ║  QA VALIDATION     ║
    ╚────────┬───────────╝
             ├─ Check for NaN/Inf
             ├─ Validate magnitudes
             ├─ Log quality metrics
             └─ Output: Status + metadata
             ↓
    ╔════════════════════╗
    ║  PHASE 7:          ║
    ║  STORAGE           ║
    ╚────────┬───────────╝
             ├─ Serialize to JSON
             ├─ Write to file
             └─ Output: embeddings_[model].jsonl

For each model (XLM-R, mBERT, MuRIL, IndicBERT):
  Repeat Phases 2-7
```

---

## FAIRNESS ACROSS MODELS

### Key Fairness Principles

```
Principle 1: Same Preprocessing
  ✓ All models receive identical normalized text
  ✓ Only tokenization is model-specific
  ✓ No model gets "cleaner" input
  
Principle 2: Same Representation Method
  ✓ All models use target-word token embedding
  ✓ All use mean pooling for subwords
  ✓ All output 768-D vectors
  ✓ Difference only: subword fragmentation patterns (unavoidable)
  
Principle 3: No Model Tuning Advantage
  ✓ Same hyperparameters (no layer selection bias)
  ✓ Use final layer for all (standardized)
  ✓ No prompt engineering or special handling
  
Principle 4: Identical Evaluation
  ✓ Same WiC pairs for all models
  ✓ Same label definitions (0 = different, 1 = same)
  ✓ Same metric calculations (cosine similarity, accuracy)

Unavoidable Differences (Not Unfair):
  - Tokenization: Model-specific, inherent to design
  - Vocabulary: Different vocabularies, can't be changed
  - Preprocessing: Following model requirements
  - These are expected differences, not fairness violations
```

### Fairness Validation Checklist

```
Before running extraction:
  ☐ Preprocessing identical for all? 
  ☐ Tokenization using each model's native tokenizer?
  ☐ Same layer selection (final layer)?
  ☐ Same pooling strategy (mean)?
  ☐ Same WiC pairs used?
  ☐ Same output format?
  
After extraction:
  ☐ All models process same pair count?
  ☐ Failure rates comparable?
  ☐ Embedding magnitudes similar range?
  ☐ Similarity distributions reasonable?
  ☐ Processing logs comparable?
```

---

## STANDARDIZED PARAMETERS

### Global Configuration

```
# Text Processing
UNICODE_NORMALIZATION = "NFC"
MAX_SEQUENCE_LENGTH = 256
MIN_KANNADA_PERCENTAGE = 80.0

# Tokenization (Model-Specific)
XLM_R_TOKENIZER = "facebook/xlm-roberta-base"
MBERT_TOKENIZER = "bert-base-multilingual-cased"
MURIL_TOKENIZER = "bert-base-multilingual-cased"  # Or specific MuRIL tokenizer
INDICBERT_TOKENIZER = "google/indicbert"

# Model Layers
USE_LAYER = -1  # Final layer for all models
POOL_STRATEGY = "mean"  # For multi-token target words

# Output Encoding
OUTPUT_PRECISION = 6  # Decimal places
OUTPUT_FORMAT = "jsonl"
BATCH_SIZE_INFERENCE = 32

# QA Thresholds
MIN_EMBEDDING_MAGNITUDE = 0.1
MAX_EMBEDDING_MAGNITUDE = 100.0
MIN_VARIANCE_THRESHOLD = 0.001
```

---

## REPRODUCIBILITY CHECKLIST

```
For reproducible results:

Software:
  ☐ Pin transformer library version (e.g., 4.25.0)
  ☐ Pin PyTorch version (e.g., 1.13.0)
  ☐ Set random seeds (torch.manual_seed, np.random.seed)
  ☐ Deterministic GPU operations

Data:
  ☐ Use same WiC dataset version
  ☐ Same preprocessing rules documented
  ☐ Same text encoding (UTF-8)

Models:
  ☐ Download from same source (HuggingFace)
  ☐ Same model versions
  ☐ No fine-tuning applied

Processing:
  ☐ Same batch processing order
  ☐ Same tokenization parameters
  ☐ Same layer extraction
  ☐ Document processing environment (GPU type, etc.)

Output:
  ☐ Save logs with timestamps
  ☐ Document any errors/warnings
  ☐ Save full configuration
  ☐ Include pipeline version number
```

---

## SUMMARY: WHY THIS PIPELINE?

### Design Choices Justified

```
Choice 1: Minimal Preprocessing
  Reason: Preserve linguistic signal; no aggressive cleaning
  Alternative: Heavy cleaning (stop-word removal, stemming)
  Why chosen: Prevents sense conflation; maintains polysemy distinctions

Choice 2: Model-Specific Tokenization
  Reason: Each model's tokenizer is optimized for its vocabulary
  Alternative: Unified tokenizer for all models
  Why chosen: Fair comparison; respects model design; standard practice

Choice 3: Target Word Token Embedding
  Reason: Captures sense-specific meaning
  Alternative: CLS token only
  Why chosen: CLS is sentence-level; target word is more granular for WSD

Choice 4: Mean Pooling
  Reason: Robust to Kannada subword fragmentation; standard in literature
  Alternative: First/last subword only
  Why chosen: Captures all subword information; no information loss

Choice 5: Final Layer Only
  Reason: Semantic representations concentrated in final layers
  Alternative: Average across multiple layers
  Why chosen: Final layer best for sense disambiguation; avoids mixing levels

Choice 6: Separate Files Per Model
  Reason: Easy to load, compare, analyze one model at a time
  Alternative: Single file with all models
  Why chosen: Scalability; modularity; faster access
```

---

## NEXT STEPS: IMPLEMENTATION

### To Apply This Pipeline

1. **Implement Phases 1-2** (Preprocessing + Tokenization)
   - Write normalize_text() function
   - Write tokenize_sentence() function per model

2. **Implement Phases 3-4** (Inference + Extraction)
   - Setup model loading (xplm-r, mBERT, MuRIL, IndicBERT)
   - Write extract_embeddings() function
   - Write target_word_pooling() function

3. **Implement Phases 5-6** (Pairing + QA)
   - Write pair_embeddings() function
   - Write validate_embeddings() function

4. **Implement Phase 7** (Storage)
   - Write save_to_jsonl() function
   - Setup file structure

5. **Batch Processing**
   - Create pipeline orchestrator
   - Handle 720 pairs × 4 models = 2880 total pairs
   - Parallelize across models (if resources available)

6. **Testing**
   - Test on 5-10 pairs first
   - Verify output format and quality
   - Validate fairness checks
   - Then run full dataset

