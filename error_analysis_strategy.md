# Error Analysis Strategy for Kannada WSD

## Purpose

Error analysis helps explain not just how well a model performs, but *where* it fails and *why*.

For this Kannada WSD project, the goal is to diagnose mistakes in WiC classification, clustering-based sense separation, and gloss-based ranking.

---

## 1) Structured Error Analysis Approach

### Step 1: Collect all errors

Create a table of all incorrect predictions with the following fields:
- target word
- sentence 1
- sentence 2
- gold label
- predicted label
- model confidence or similarity score
- part of speech
- sense IDs if available
- source model

This gives a single place to inspect and compare errors.

### Step 2: Group errors by type

Organize mistakes into categories such as:
- false positive: predicted same sense, but gold label is different
- false negative: predicted different sense, but gold label is same
- ambiguous case: even humans may disagree on the label
- annotation error: gold label appears incorrect
- low-context case: not enough context to support a decision
- rare-sense case: the correct sense is underrepresented in training or corpus data

### Step 3: Group errors by linguistic property

Slice the errors by:
- part of speech
- target word
- sense pair type
- sentence length
- context window size
- model used
- confidence level

This helps identify whether the model fails systematically or only on certain subsets.

### Step 4: Compare correct vs incorrect examples

For each failure category, inspect both:
- a few false predictions
- a few correct predictions for the same word or POS

This reveals what distinguishes easy cases from hard ones.

### Step 5: Summarize recurring patterns

Look for repeated reasons behind mistakes, such as:
- sense overlap
- weak contextual cues
- polysemy with fine-grained senses
- morphology or tokenization issues
- rare senses
- metaphorical usage
- noisy or unnatural sentences

---

## 2) Analyze by Part of Speech

### Why POS analysis matters

Different parts of speech behave differently in WSD:
- nouns often have stable sense clusters
- verbs often depend strongly on surrounding argument structure
- adjectives may have subtle semantic shifts and be harder to separate

### Suggested POS buckets

- noun
- verb
- adjective
- adverb, if present
- mixed or uncertain cases

### What to measure by POS

For each POS category, report:
- number of examples
- accuracy
- macro F1
- error rate
- most common error types
- most confusing sense pairs

### Questions to ask by POS

- Are nouns easier than verbs?
- Do adjectives fail more often because senses are close in meaning?
- Do verbs rely more on long-range context than the current embedding captures?
- Are some POS categories dominated by rare senses?
- Does tokenization hurt certain POS more than others?

---

## 3) Identify Patterns in Model Mistakes

### Pattern A: Confusing close senses

The model may fail when two senses are semantically similar.

Examples of signs:
- same clusters for related senses
- high similarity for incorrect pairs
- repeated confusion between near-synonyms or closely related meanings

### Pattern B: Rare sense failures

The model may ignore senses that appear infrequently.

Signs:
- rare sense almost never predicted
- one cluster absorbs most examples
- low recall for a minority sense

### Pattern C: Context weakness

Some sentences do not provide enough surrounding information.

Signs:
- short sentences
- generic context words
- very low confidence scores
- unstable predictions across models

### Pattern D: Tokenization problems

Kannada words may split into multiple subwords and distort the representation.

Signs:
- long subword chains
- inconsistent embeddings for the same target word
- errors concentrated on morphologically complex forms

### Pattern E: POS-specific ambiguity

Certain POS categories may be inherently harder.

Signs:
- verbs or adjectives show lower scores than nouns
- systematic confusion within one POS group
- stronger dependence on syntactic context

### Pattern F: Annotation noise

Sometimes the model is correct but the label is wrong.

Signs:
- the sentence clearly supports a different sense than the gold label
- multiple annotators would likely disagree
- the same word pair appears contradictory across the dataset

---

## 4) Suggested Error Analysis Table

A practical template:

| target word | POS | sentence 1 | sentence 2 | gold | pred | confidence | error type | notes |
|---|---|---|---|---|---|---|---|---|
| ಕಾಲ | noun | ... | ... | 1 | 0 | 0.42 | false negative | weak context |
| ಗಟ್ಟ | adjective | ... | ... | 0 | 1 | 0.81 | false positive | sense overlap |

This table can be filtered by word, POS, model, or error type.

---

## 5) Key Questions to Answer

### Overall performance
- Where does the model fail most often?
- Are errors concentrated in a few words or spread evenly?
- Are errors due to model weakness or dataset noise?

### POS-focused questions
- Which part of speech has the highest error rate?
- Which POS has the lowest macro F1?
- Do nouns, verbs, and adjectives show different confusion patterns?

### Sense-focused questions
- Which sense pairs are most often confused?
- Are rare senses being ignored?
- Are errors mostly between semantically close senses?

### Context-focused questions
- Does sentence length affect accuracy?
- Do short or generic contexts produce more mistakes?
- Does the model fail when the target word has few contextual clues?

### Model-behavior questions
- Are errors correlated with low confidence?
- Do different models fail on the same examples?
- Does one model make fewer rare-sense mistakes than others?

### Data-quality questions
- Are some labels likely incorrect?
- Are the generated sentences natural and idiomatic?
- Do ambiguous cases need manual review?

---

## 6) Recommended Reporting Format

A strong error analysis report should include:
- overall metric summary
- error breakdown by POS
- error breakdown by target word
- top confusion pairs
- representative failure cases
- representative correct cases
- observations about recurring model weaknesses
- dataset issues discovered during analysis
- recommended fixes or next steps

---

## 7) Final Takeaway

Good error analysis should answer three things:
1. **What failed?**
2. **Where did it fail?**
3. **Why did it fail?**

If those answers are clear, the next model iteration becomes much easier to design.
