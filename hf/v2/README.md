---
language:
- kn
task_categories:
- text-classification
task_ids:
- word-sense-disambiguation
pretty_name: Kannada WiC Benchmark v2
license: cc-by-4.0
---

# Kannada WiC Benchmark v2

## Dataset Summary
WiC benchmark synthesized from live Kannada IndoWordNet (pyiwn) and Kannada Wikipedia context.

## Dataset Structure
- Total pairs: 640
- Unique words: 45
- Label 1 count: 320
- Label 0 count: 320
- Splits: train / validation / test

## Data Sources
- IndoWordNet Kannada synsets and examples via pyiwn
- Kannada Wikipedia summaries and search snippets

## Quality Control
- 20% sample validation report generated in reports/v2/data_quality_report.md

## Evaluation
- Metrics: Accuracy, Macro F1, ARI, MRR
- Includes random baselines, zero-shot, and few-shot tracks

## Limitations
- Wikipedia snippets may contain noisy sentence fragments
- Some words have sparse real-context examples

## Ethical Considerations
- Lexical ambiguity may reflect socio-cultural bias in source corpora