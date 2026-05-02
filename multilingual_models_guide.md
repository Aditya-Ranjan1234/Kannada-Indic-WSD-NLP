# Multilingual Models for Kannada Sentence Representation
## Conceptual Guide for WSD Applications

---

## OVERVIEW OF MODELS

### 1. **XLM-R (Cross-lingual Language Model - Robustly Multilingual)**

#### Model Characteristics
- **Pretrained on**: 100+ languages (including Kannada)
- **Architecture**: RoBERTa-based transformer
- **Vocabulary**: 250K shared subword tokens across all languages
- **Parameters**: 550M (base), 1B+ (large)
- **Training Data**: CC100 corpus (diverse web data)

#### Embedding Extraction
**CLS Token (Recommended for Sentence Representation)**:
- Represents entire sentence meaning
- Extracted from final layer `[CLS]` token
- Shape: `[1, hidden_size]` (typically 768 for base)
- Best for: Sentence-level classification, similarity tasks
- Why: CLS token is explicitly trained via MLM to aggregate sentence semantics

**Token-Level Embeddings**:
- Extract hidden states from all tokens
- Shape: `[sequence_length, hidden_size]`
- Best for: Word-level analysis, sense-specific representations
- Use for Kannada WSD: Can extract embeddings for target word across different contexts

**Attention Weights** (Advanced):
- Extract attention patterns from transformer heads
- Reveals how model aligns multilingual semantics
- Limited use for standard WSD but useful for interpretability

#### Strengths for Cross-Lingual Tasks
✓ **Broad Language Coverage**: Handles Kannada, English, Hindi without special tuning
✓ **Robust Subword Tokenization**: Kannada words decomposed into meaningful subwords
✓ **Cross-Lingual Transfer**: Can leverage English sense definitions to improve Kannada WSD
✓ **Fine-tuning Friendly**: Good starting point for downstream tasks
✓ **Proven Performance**: State-of-the-art on many multilingual benchmarks

#### Weaknesses for Cross-Lingual Tasks
✗ **Language Imbalance**: High-resource languages (EN, ZH) may dominate representations
✗ **Kannada Underrepresentation**: Low-resource Kannada data in pretraining = weaker Kannada embeddings
✗ **Subword Fragmentation**: Kannada scripts (Indic script) may fragment more than Latin scripts
✗ **Limited Kannada Context**: Pretraining data for Kannada is relatively small compared to EN
✗ **Cross-Script Limitations**: Cannot leverage structural similarities across Indic scripts

#### Expected Behavior on Kannada
- **CLS Embeddings**: Captures broad semantic meaning but may conflate rare Kannada senses
- **Polysemy Handling**: Can distinguish senses reasonably well for common words (ಕಾಲ, ಮಾಟು), but struggles with rare senses
- **Subword Tokenization**: Kannada words often split into 2-4 subword pieces
- **Sense Stability**: Embeddings may shift across different context types (formal vs colloquial)
- **Performance Baseline**: Expect ~65-75% accuracy on Kannada WSD tasks without fine-tuning

#### Architecture Diagram (Conceptual)
```
Input: "ಕಾಲ ದ್ರುತವಾಗಿ ಹಾದುಹೋಗಿ"
    ↓
Tokenization: ["[CLS]", "ಕಾ", "ಲ", "ದ್ರುತ", "ವಾಗಿ", "ಹಾದು", "ಹೋಗಿ", "[SEP]"]
    ↓
Token Embeddings: [768-dim vectors for each token]
    ↓
12 Transformer Layers (XLM-R base)
    ↓
Output Layer:
  - CLS Token: [768] → Sentence representation
  - Token i: [768] → Word representation
  - Attention Heads: 12 × 12 matrices → Alignment patterns
```

---

### 2. **mBERT (Multilingual BERT)**

#### Model Characteristics
- **Pretrained on**: 104 languages (including Kannada)
- **Architecture**: Original BERT-base architecture
- **Vocabulary**: 119K shared subword tokens
- **Parameters**: 110M (relatively lightweight)
- **Training Data**: Wikipedia + BookCorpus

#### Embedding Extraction
**CLS Token**:
- Standard approach for classification tasks
- Extracted from final layer `[CLS]`
- Shape: `[1, 768]`
- Best for: Quick sentence representations, multilingual transfer

**Token-Level Embeddings**:
- All token hidden states available
- Useful for understanding individual word contributions
- For Kannada WSD: Can pinpoint exact sense contributions

**Pooling Strategies** (Post-processing):
- **Mean pooling**: Average of all token embeddings (more robust to noise)
- **Max pooling**: Maximum activation across tokens
- **Weighted pooling**: Using attention weights as masks

#### Strengths for Cross-Lingual Tasks
✓ **Lightweight**: Smaller model size → faster inference, lower memory
✓ **Wide Language Support**: 104 languages ensure Kannada coverage
✓ **Well-Established**: Extensive research and community resources
✓ **Zero-Shot Transfer**: Works reasonably well across language pairs
✓ **Interpretable Embeddings**: BERT embeddings are well-studied and interpretable

#### Weaknesses for Cross-Lingual Tasks
✗ **Older Architecture**: Predates contextual improvements in XLM-R
✗ **Smaller Vocabulary**: 119K tokens → more aggressive Kannada subword splitting
✗ **Wikipedia Bias**: Overrepresents formal language, underrepresents colloquial Kannada
✗ **Weaker Multilingual Alignment**: Not explicitly trained for cross-lingual transfer
✗ **Limited Indic Script Support**: Relative to models designed for Indian languages

#### Expected Behavior on Kannada
- **CLS Embeddings**: Captures semantic content but less robust than XLM-R
- **Polysemy Handling**: Moderate performance on sense disambiguation (~60-70%)
- **Subword Tokenization**: Kannada words split into 2-5 subword pieces (more fragmentation than XLM-R)
- **Domain Generalization**: Better on Wikipedia-like formal text, struggles with colloquial Kannada
- **Performance Baseline**: Expect ~60-70% accuracy on Kannada WSD tasks

#### Comparison to XLM-R
```
Feature                | mBERT          | XLM-R
Model Size             | 110M           | 550M
Language Coverage      | 104            | 100+
Kannada Pretraining    | Wikipedia      | CC100 (more diverse)
Subword Vocab          | 119K           | 250K
Cross-Lingual Align    | Weak           | Strong
Inference Speed        | Fast           | Slower
Performance (Kannada)  | 60-70%         | 65-75%
```

---

### 3. **MuRIL (Multilingual Representations for Indian Languages)**

#### Model Characteristics
- **Designed for**: Indian languages (Hindi, Tamil, Telugu, Kannada, Marathi, Bengali, Gujarati, Urdu)
- **Architecture**: RoBERTa-based (similar to XLM-R)
- **Vocabulary**: Specific focus on Indic scripts
- **Parameters**: 560M (comparable to XLM-R base)
- **Training Data**: Wikipedia + OSCAR corpus (Indian language focused)

#### Embedding Extraction
**CLS Token** (Best for Kannada):
- Optimized for Indian language semantics
- Extracted from final layer
- Shape: `[1, 768]`
- **Advantage**: Kannada linguistic properties better preserved

**Token-Level Embeddings**:
- Individual token representations
- Particularly useful for Kannada morphology
- Can capture verb conjugations, noun declensions

**Subword-Level Analysis**:
- Kannada-specific subword vocabulary enables finer-grained analysis
- Useful for understanding morphologically complex words

#### Strengths for Cross-Lingual Tasks (Indian Context)
✓ **Kannada-Optimized**: Explicitly trained on diverse Kannada texts
✓ **Indic Script Awareness**: Vocabulary designed for Kannada, Telugu, Tamil, etc.
✓ **Cross-Indic Transfer**: Can leverage Hindi/Tamil to improve Kannada WSD
✓ **Morphological Sensitivity**: Better captures Kannada word structure
✓ **Regional Language Focus**: Understands cultural/regional nuances

#### Weaknesses for Cross-Lingual Tasks
✗ **Limited Language Coverage**: Only 8 Indian languages (no English, Chinese, etc.)
✗ **Less Multilingual Transfer**: Cannot leverage non-Indic languages
✗ **Smaller Pretraining Corpus**: Less data than XLM-R overall
✗ **Niche Adoption**: Fewer community resources and research papers
✗ **Model Size**: Medium-sized, not as lightweight as mBERT

#### Expected Behavior on Kannada
- **CLS Embeddings**: **Best performance for Kannada** - captures nuanced Kannada semantics
- **Polysemy Handling**: **Excellent** for Kannada sense disambiguation (~75-85%)
- **Subword Tokenization**: Kannada-optimized tokens (1-2 pieces per word average)
- **Morphological Awareness**: Handles conjugations, declensions, derivations well
- **Performance Baseline**: Expect **75-85% accuracy on Kannada WSD tasks** (best for Kannada-only)

#### Unique Advantage: Cross-Indic Transfer
```
Example: Kannada-English Polysemy
Word: ಕಾಲ (kāla)
  
Senses in Hindi (मौसम, समय): 
  - Season (ऋतु)
  - Time (समय)
  
Senses in Kannada (Same word):
  - Time (ಕಾಲ)
  - Leg (ಕಾಲ)
  - Death (ಕಾಲ - poetic)

MuRIL can leverage Hindi-Kannada alignment to disambiguate
```

---

### 4. **IndicBERT (BERT for Indic Languages)**

#### Model Characteristics
- **Designed for**: 12 Indian scripts + English
- **Architecture**: BERT-base architecture
- **Vocabulary**: 10K shared + language-specific tokens
- **Parameters**: 110M (lightweight)
- **Training Data**: Wikipedia + OSCAR in Indic languages + English

#### Embedding Extraction
**CLS Token** (Recommended):
- Simple, fast sentence representation
- Shape: `[1, 768]`
- Best for: Quick inference, real-time applications

**Token-Level Embeddings**:
- Useful for script-aware analysis
- Captures Kannada script-specific features

**Language Tag Embedding** (Advanced):
- IndicBERT includes language tags (e.g., `[Kn]` for Kannada)
- Can separate language-specific representations

#### Strengths for Cross-Lingual Tasks
✓ **Lightweight**: 110M parameters, fastest inference
✓ **Script-Aware**: Handles Kannada script properties explicitly
✓ **English Integration**: Includes English for comparative analysis
✓ **Low-Resource Friendly**: Can work on devices with limited memory
✓ **Cross-Script Learning**: Can leverage Tamil, Telugu for similar script properties

#### Weaknesses for Cross-Lingual Tasks
✗ **Smaller Vocabulary**: 10K shared tokens → more subword fragmentation
✗ **Limited Pretraining Data**: Less diverse than XLM-R or MuRIL
✗ **Older Architecture**: BERT-base (vs RoBERTa in XLM-R/MuRIL)
✗ **Less Semantic Depth**: Smaller model may miss nuanced Kannada semantics
✗ **Niche Adoption**: Limited community adoption for advanced NLP tasks

#### Expected Behavior on Kannada
- **CLS Embeddings**: Basic semantic content, suitable for classification
- **Polysemy Handling**: Moderate (~65-75% on Kannada WSD)
- **Subword Tokenization**: High fragmentation (3-4 pieces per Kannada word)
- **Script Awareness**: Good for script-level tasks
- **Performance Baseline**: Expect **65-75% accuracy on Kannada WSD tasks** (good for lightweight applications)

---

## COMPARATIVE ANALYSIS

### Performance on Kannada WSD (Estimated Benchmarks)

```
Model          | Accuracy | Speed | Memory | Kannada Focus | Best For
XLM-R          | 68-75%   | 300ms| 2.1GB | Moderate      | General-purpose, high accuracy
mBERT          | 60-70%   | 150ms| 0.8GB | Low           | Fast baseline, multilingual
MuRIL          | 75-85%   | 350ms| 2.2GB | HIGH          | Kannada-optimized, best Kannada
IndicBERT      | 65-75%   | 120ms| 0.7GB | High          | Lightweight, real-time
```

### Embedding Characteristics by Model

| Aspect | XLM-R | mBERT | MuRIL | IndicBERT |
|---|---|---|---|---|
| **CLS Dimension** | 768 | 768 | 768 | 768 |
| **Token Sequence** | Yes | Yes | Yes | Yes |
| **Attention Heads** | 12 | 12 | 12 | 12 |
| **Layers** | 12 | 12 | 24 | 12 |
| **Kannada Subwords/Token** | 2-3 | 2-4 | 1-2 | 3-4 |
| **Contextual Depth** | Very High | Medium | Very High | Medium |
| **Code-Switch Support** | Excellent | Good | Excellent | Good |

---

## USAGE PATTERNS FOR KANNADA WSD

### Pattern 1: CLS Token Extraction (Standard)

**When to use**: Sentence-level classification, sentence similarity
**Process**:
1. Input: Kannada sentence with target word
2. Tokenize using model's tokenizer
3. Pass through all transformer layers
4. Extract `[CLS]` token from last hidden state
5. Use as sentence representation vector

**For Kannada**: All four models support this. **Recommendation: Use MuRIL or XLM-R**

**Example conceptual flow**:
```
Input Sentence: "ಕಾಲ ದ್ರುತವಾಗಿ ಹಾದುಹೋಗಿ"

Model Processing:
  1. Tokenize → ["[CLS]", "ಕಾ", "ಲ", "ದ್ರುತ", "ವಾ", "ಗಿ", "ಹಾದು", "ಹೋ", "ಗಿ", "[SEP]"]
  2. Token embeddings → 10 vectors of dim 768
  3. Transformer attention → contextualize each token
  4. Extract CLS from last layer → [768]

Output: CLS embedding represents "time passed quickly" meaning
```

---

### Pattern 2: Target Word Embedding (Sense-Specific)

**When to use**: Word-level sense disambiguation, sense-specific analysis
**Process**:
1. Identify target word position in tokenized sequence
2. Extract hidden state at that position
3. Use as target-word-specific representation

**For Kannada WSD**: **Ideal approach for this task**

**Advantages**:
- Captures sense-specific context
- More directly relevant than CLS for WSD
- Can compare embeddings of same word in different sentences

**Example**:
```
Sentence 1: "ಅವನ ಎಡ ಕಾಲು ಮುರಿದಿದೆ" (leg - anatomical sense)
Sentence 2: "ಕಾಲ ದ್ರುತವಾಗಿ ಹಾದುಹೋಗಿ" (time - temporal sense)

Extract embeddings for "ಕಾ" + "ಲ" tokens at their positions
Compare: These embeddings should differ substantially
Distance in embedding space ↔ Semantic distance
```

---

### Pattern 3: Pooled Token Embeddings (Robust)

**When to use**: Robust representations, noise reduction
**Process**:
1. Extract all token embeddings (excluding special tokens)
2. Apply pooling strategy:
   - **Mean**: Average all token vectors
   - **Max**: Element-wise maximum across tokens
   - **Weighted**: Use attention weights as masks

**For Kannada WSD**: Good for handling subword fragmentation

**Example for Kannada**:
```
Word: ಕಾಲ tokenizes to ["ಕಾ", "ಲ"]

Option 1 (Mean Pooling):
  representation = (embed["ಕಾ"] + embed["ಲ"]) / 2
  
Option 2 (Max Pooling):
  representation[i] = max(embed["ಕಾ"][i], embed["ಲ"][i]) for each dimension i

Option 3 (Attention Weighted):
  representation = attention_weight["ಕಾ"] × embed["ಕಾ"] + 
                   attention_weight["ಲ"] × embed["ಲ"]
```

---

### Pattern 4: Layered Representations

**When to use**: Analyzing model internals, ensemble methods
**Process**:
1. Extract embeddings from multiple layers (not just last)
2. Earlier layers → more syntactic, language-general
3. Later layers → more semantic, task-specific

**For Kannada**: Useful for understanding what models learn

**Conceptual layer hierarchy**:
```
Layer 1 (Early):   Kannada script features, basic morphology
Layer 6 (Middle):  Part-of-speech, syntax-like patterns
Layer 12 (Final):  Semantics, sense distinctions, context integration

For Kannada WSD, use Layer 12 (final) by default
```

---

## KEY CONCEPTS FOR SENTENCE REPRESENTATION

### Concept 1: Contextualization

**What it means**: Word embeddings change based on context (unlike static Word2Vec)

**For Kannada**:
- Word "ಕಾಲ" has different embeddings in:
  - "ಕಾಲ ದ್ರುತವಾಗಿ ಹಾದುಹೋಗಿ" (time) → embedding represents TIME semantics
  - "ಅವನ ಎಡ ಕಾಲು ಮುರಿದಿದೆ" (leg) → embedding represents LEG semantics

**Why important**: Enables WSD - different senses produce different embeddings

---

### Concept 2: Multilingual Space

**What it means**: All languages share embedding space (different language representations in same 768-D space)

**For Kannada**:
- English "time" and Kannada "ಕಾಲ" (time sense) appear near each other
- Can do cross-lingual sense matching
- **But**: Language-imbalance means English embeddings dominate

**Implication**: XLM-R/mBERT can transfer English sense knowledge to Kannada

---

### Concept 3: Subword Tokenization Impact

**How it affects Kannada**:
- Kannada words split into multiple tokens
- Each subword gets separate embedding
- Final representation is either:
  - First subword only (`[CLS]` position)
  - Concatenation of all subwords
  - Average/Max pooling across subwords

**Model comparison for Kannada word ಕಾಲ**:
```
XLM-R:      ["ಕಾ", "ಲ"]                      → 2 tokens (good)
mBERT:      ["ಕಾ", "ಲ"]                      → 2 tokens (good)
MuRIL:      ["ಕಾಲ"]                         → 1 token (best)
IndicBERT:  ["ಕ", "ಾ", "ಲ"]                → 3 tokens (fragmented)
```

---

### Concept 4: CLS vs Token Embeddings

**CLS Token**:
- Special "classification" token prepended to input
- Trained via MLM to aggregate sentence meaning
- Pros: Simple, fast, aggregates entire context
- Cons: May lose word-specific information
- **Best for**: Sentence classification, quick representations

**Token Embeddings**:
- Individual word representations
- Preserves word-specific context
- Pros: Fine-grained, sense-specific
- Cons: Multiple tokens per word, noisier for Kannada
- **Best for**: Word-level tasks like WSD

**For Kannada WSD**: Use **Token embedding of target word**

---

## EXPECTED BEHAVIOR SUMMARY

### XLM-R on Kannada
```
Strengths:
  - General-purpose, proven multilingual support
  - Can leverage cross-lingual similarities
  - Good for rare Kannada words (via English transfer)
  
Weaknesses:
  - Kannada underrepresented in pretraining
  - Subword fragmentation (2-3 tokens/word)
  - May conflate Kannada with other Indic languages
  
Best Use: General WSD, cross-lingual tasks
Accuracy: 68-75%
```

### mBERT on Kannada
```
Strengths:
  - Lightweight, fast
  - Good as baseline
  - Established research
  
Weaknesses:
  - Smaller model = less semantic capacity
  - More subword fragmentation
  - Wikipedia-biased (formal language)
  
Best Use: Lightweight deployment, quick baselines
Accuracy: 60-70%
```

### MuRIL on Kannada ⭐ RECOMMENDED
```
Strengths:
  - Explicitly designed for Kannada & Indic languages
  - Minimal subword fragmentation (1-2 tokens/word)
  - Can leverage Hindi/Tamil for transfer
  - Best embeddings for Kannada polysemy
  
Weaknesses:
  - Cannot leverage non-Indic languages
  - Limited English transfer
  - Niche adoption (fewer research papers)
  
Best Use: Kannada-focused WSD, Indic language tasks
Accuracy: 75-85% ⭐ BEST FOR KANNADA
```

### IndicBERT on Kannada
```
Strengths:
  - Lightweight, low-memory
  - Real-time inference capability
  - Script-aware
  
Weaknesses:
  - Smaller model capacity
  - Older architecture (BERT-base)
  - High subword fragmentation (3-4 tokens/word)
  
Best Use: Mobile/edge deployment, real-time WSD
Accuracy: 65-75%
```

---

## RECOMMENDATIONS FOR KANNADA WSD PROJECT

### Tier 1 (Best Performance)
**Use MuRIL**:
- Extract **target word token embeddings** (not CLS)
- Apply **mean pooling** across subword fragments
- Fine-tune on Kannada WSD dataset
- Expected accuracy: **75-85%**

### Tier 2 (General Purpose)
**Use XLM-R**:
- Extract **target word token embeddings**
- Option to use English definitions for sense enrichment
- Can transfer from English SemEval WSD datasets
- Expected accuracy: **68-75%**

### Tier 3 (Fast Baseline)
**Use IndicBERT**:
- Extract **CLS token** for speed
- Use for quick prototyping
- Deploy on resource-constrained devices
- Expected accuracy: **65-75%**

### Tier 4 (Baseline)
**Use mBERT**:
- Use for comparison only
- Useful for understanding impact of Kannada-specific modeling
- Expected accuracy: **60-70%**

---

## PRACTICAL CONSIDERATIONS

### Memory & Speed Trade-offs
```
Real-time (< 100ms):     IndicBERT (120ms)
Fast (100-200ms):        mBERT (150ms)
Standard (200-400ms):    XLM-R (300ms), MuRIL (350ms)
Batch inference:         MuRIL (best accuracy), XLM-R (good balance)
```

### Pretraining Data Quality
```
Model          | Kannada Data Quality  | Relevance to WSD
XLM-R          | CC100 (web data)      | Mixed quality
mBERT          | Wikipedia             | Formal, encyclopedic
MuRIL          | Wikipedia + OSCAR     | Diverse, formal + informal
IndicBERT      | Wikipedia + OSCAR     | Diverse
```

### Fine-tuning Recommendations
```
MuRIL:        Few-shot learning OK (good Kannada basis)
XLM-R:        Needs moderate data (100+ examples)
IndicBERT:    Needs moderate data
mBERT:        Needs more data for Kannada
```

---

## SUMMARY TABLE

| Aspect | XLM-R | mBERT | MuRIL | IndicBERT |
|---|---|---|---|---|
| **Kannada Performance** | 🟨 Good | 🟧 Fair | 🟩 Excellent | 🟨 Good |
| **Speed** | 300ms | 150ms | 350ms | 120ms |
| **Memory** | 2.1GB | 0.8GB | 2.2GB | 0.7GB |
| **Kannada Optimization** | No | No | Yes | Partial |
| **Subword Fragmentation** | Low | Low | Very Low | High |
| **Cross-Lingual Transfer** | Excellent | Good | Good (Indic) | Moderate |
| **Best Embedding Strategy** | Token + Pooling | CLS | Token + Pooling | Token + Pooling |
| **Recommended for WSD** | ⭐⭐ | ⭐ | ⭐⭐⭐ | ⭐⭐ |

---

## CONCEPTUAL TAKEAWAYS

### Key Insight 1: CLS vs Token Trade-off
- **CLS**: Fast, aggregated, but loses word-specific detail
- **Token**: Precise, sense-specific, but multi-token for Kannada
- **Decision**: For WSD, prioritize **token embeddings** of target word

### Key Insight 2: Language-Specific Models Win
- **General multilingual** (XLM-R) better for cross-lingual, worse for Kannada alone
- **Kannada-specific** (MuRIL) better for Kannada-only tasks
- **Tradeoff**: Kannada-specific models can't leverage English, Chinese, etc.

### Key Insight 3: Subword Fragmentation Matters for Kannada
- Kannada splits more than English due to script complexity
- **Pooling strategies** (mean/max) help mitigate fragmentation
- **MuRIL** minimizes fragmentation (best for Kannada morphology)

### Key Insight 4: Contextual Embeddings for Polysemy
- Different senses produce **different embedding vectors**
- Embedding **distance** in latent space ≈ semantic distance
- This enables WSD via nearest-neighbor or clustering approaches

