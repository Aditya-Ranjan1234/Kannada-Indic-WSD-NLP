# Progress Log

Project: Cross-lingual Sense Transfer with Low-Resource Indic Languages

## Roadmap
1. Phase 1: Dataset creation and cleaning
2. Phase 2: Embedding extraction and similarity scoring
3. Phase 3: K-means clustering and visualization
4. Phase 4: Gloss-based baseline and evaluation
5. Phase 5: Reporting and final synthesis

## Current Status
- All phases completed.
- External dependencies resolved: Hugging Face authentication + tokenizer/reporting dependencies.
- Gloss baseline source used doc-derived glosses from workspace markdown files (no local IndoWordNet export available).
- Final outputs are present in data, outputs, results, and reports.

## Phase 1: Dataset creation and cleaning
- What was done: Parsed explicit WiC pairs from markdown and augmented from curated sense-tagged sentence inventory.
- Issues faced: Raw explicit pairs were below target size.
- Fixes applied: Added balanced synthetic pair generation and order augmentation to produce 627 cleaned pairs (within 500-800 target) plus 20% validation sample.

## Phase 2: Embedding extraction and similarity scoring
- What was done: Computed model embeddings for unique sentence occurrences and wrote pair-level cosine similarities and tuned thresholds.
- Issues faced: Initial run failed on gated IndicBERT access.
- Fixes applied: Re-ran with Hugging Face token and kept one common extraction/similarity pipeline across XLM-R, mBERT, MuRIL, and IndicBERT.

## Phase 3: K-means clustering and visualization
- What was done: Clustered unique sentence occurrences per word and computed ARI summaries.
- Issues faced: Crash on NaN sense IDs during integer conversion.
- Fixes applied: Added NaN filtering in occurrence builder before clustering.

## Phase 4: Gloss baseline and evaluation
- What was done: Ranked gloss candidates for each sense and computed MRR summaries.
- Issues faced: Gloss coverage is derived from the markdown docs rather than a live IndoWordNet export.
- Fixes applied: Built a doc-based gloss inventory and used the same embedding pipeline for sentence and gloss scoring.

## Phase 5: Reporting and final synthesis
- What was done: Generated comparison plots, t-SNE outputs, and the final markdown report.
- Issues faced: Missing runtime dependencies (`protobuf`, `tiktoken`, `tabulate`) during iterative runs.
- Fixes applied: Installed dependencies and completed report generation.
