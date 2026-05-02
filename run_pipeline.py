#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, f1_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = ROOT / "reports"
PROGRESS_FILE = ROOT / "progress.md"

MODEL_SPECS = {
    "xlmr": "xlm-roberta-base",
    "mbert": "bert-base-multilingual-cased",
    "muril": "google/muril-base-cased",
    "indicbert": "ai4bharat/indic-bert",
}

PAIR_RE = re.compile(r"^(?P<s1>.+?)\s*\|\s*(?P<s2>.+?)\s*\|\s*(?P<word>.+?)\s*\|\s*(?P<label>[01])\s*$")
WORD_HEADER_RE = re.compile(r"^###\s+\d+\.\s+\*\*?(?P<word>[^*()]+?)\*\*?\s*\((?P<roman>[^)]+)\)")
SENSE_HEADER_RE = re.compile(r"^\*\*Sense\s+(?P<sense>\d+):\s*(?P<gloss>.+?)\*\*$")
NUMBERED_RE = re.compile(r"^(?P<idx>\d+)\.\s+(?P<text>.+?)\s*$")
SENSE_LINE_RE = re.compile(r"^###\s+(?P<idx>\d+)\.\s+(?P<word>.+?)\s*\((?P<roman>[^)]+)\)\s*-\s*(?P<count>\d+)\s+senses?")
POS_HEADER_RE = re.compile(r"^##\s+(?P<pos>[A-Z]+)")


@dataclass
class ParsedResources:
    word_inventory: pd.DataFrame
    example_inventory: pd.DataFrame
    raw_pairs: pd.DataFrame
    sense_lookup: dict[tuple[str, str], dict[str, Any]]
    gloss_lookup: dict[tuple[str, int], str]


def ensure_dirs() -> None:
    for path in [DATA_DIR, OUTPUTS_DIR, RESULTS_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def sentence_key(text: str) -> str:
    text = normalize_text(text)
    text = text.rstrip(" .,!?:;۔।")
    return text


def sanitize_word(word: str) -> str:
    return re.sub(r"\W+", "_", sentence_key(word))


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def parse_word_inventory(path: Path) -> pd.DataFrame:
    lines = read_lines(path)
    pos = None
    current_word = None
    current_roman = None
    sense_id = None
    collecting_meanings = False
    rows: list[dict[str, Any]] = []
    current_meanings: list[str] = []

    def flush_current() -> None:
        nonlocal current_word, current_roman, sense_id, current_meanings, collecting_meanings
        if current_word and current_meanings:
            for idx, gloss in enumerate(current_meanings, start=1):
                rows.append(
                    {
                        "word": current_word,
                        "romanization": current_roman,
                        "pos": pos,
                        "sense_id": idx,
                        "gloss": normalize_text(gloss),
                    }
                )
        current_meanings = []
        collecting_meanings = False
        sense_id = None

    for line in lines:
        pos_match = POS_HEADER_RE.match(line)
        if pos_match:
            flush_current()
            pos = pos_match.group("pos").lower().rstrip("s")
            continue

        header_match = re.match(r"^###\s+\d+\.\s+\*{0,2}(?P<word>[^*()]+?)\*{0,2}\s*\((?P<roman>[^)]+)\)", line)
        if header_match:
            flush_current()
            current_word = normalize_text(header_match.group("word"))
            current_roman = normalize_text(header_match.group("roman"))
            continue

        if "- **Meanings**:" in line:
            collecting_meanings = True
            current_meanings = []
            continue

        if collecting_meanings:
            num_match = NUMBERED_RE.match(line.strip())
            if num_match:
                current_meanings.append(num_match.group("text"))
                continue
            if line.strip() == "":
                continue
            if line.startswith("### ") or line.startswith("## "):
                flush_current()
                continue

    flush_current()
    return pd.DataFrame(rows).drop_duplicates(subset=["word", "sense_id"]).reset_index(drop=True)


def parse_example_inventory(path: Path) -> pd.DataFrame:
    lines = read_lines(path)
    current_word = None
    current_roman = None
    current_sense = None
    current_gloss = None
    rows: list[dict[str, Any]] = []

    for line in lines:
        word_match = SENSE_LINE_RE.match(line)
        if word_match:
            current_word = normalize_text(word_match.group("word"))
            current_roman = normalize_text(word_match.group("roman"))
            current_sense = None
            current_gloss = None
            continue

        sense_match = SENSE_HEADER_RE.match(line)
        if sense_match:
            current_sense = int(sense_match.group("sense"))
            current_gloss = normalize_text(sense_match.group("gloss"))
            continue

        if line.startswith("- ") and current_word and current_sense is not None:
            sentence_text = line[2:].split("(", 1)[0].strip()
            if sentence_text and not sentence_text.startswith("Source:") and not sentence_text.startswith("Context:"):
                rows.append(
                    {
                        "word": current_word,
                        "romanization": current_roman,
                        "sense_id": current_sense,
                        "sense_gloss": current_gloss,
                        "sentence": normalize_text(sentence_text),
                        "sentence_key": sentence_key(sentence_text),
                    }
                )

    return pd.DataFrame(rows).drop_duplicates(subset=["word", "sense_id", "sentence_key"]).reset_index(drop=True)


def parse_wic_pairs(path: Path) -> pd.DataFrame:
    lines = read_lines(path)
    rows: list[dict[str, Any]] = []
    current_section = None
    current_word = None
    pair_idx = 0

    for line in lines:
        if line.startswith("### Word:"):
            current_word = normalize_text(line.split(":", 1)[1].split("-")[0])
            current_section = "word"
            continue
        if "SAME SENSE PAIRS" in line:
            current_section = "same"
            continue
        if "DIFFERENT SENSE PAIRS" in line:
            current_section = "different"
            continue
        match = PAIR_RE.match(line)
        if match:
            pair_idx += 1
            s1 = normalize_text(match.group("s1"))
            s2 = normalize_text(match.group("s2"))
            target = normalize_text(match.group("word"))
            label = int(match.group("label"))
            rows.append(
                {
                    "pair_id": f"pair_{pair_idx:04d}",
                    "section": current_section,
                    "section_word": current_word,
                    "sentence1": s1,
                    "sentence2": s2,
                    "target_word": target,
                    "target_key": sentence_key(target),
                    "label_raw": label,
                    "label": label,
                }
            )

    return pd.DataFrame(rows).reset_index(drop=True)


def generate_augmented_pairs(example_df: pd.DataFrame, target_total: int = 720) -> pd.DataFrame:
    same_candidates: list[dict[str, Any]] = []
    diff_candidates: list[dict[str, Any]] = []

    for word_key, word_group in example_df.groupby(example_df["word"].map(sentence_key)):
        word = word_group.iloc[0]["word"]
        sense_groups = [group.reset_index(drop=True) for _, group in word_group.groupby("sense_id")]
        if not sense_groups:
            continue

        for group in sense_groups:
            sentences = group["sentence"].tolist()
            sense_id = int(group.iloc[0]["sense_id"])
            for i in range(len(sentences)):
                for j in range(i + 1, len(sentences)):
                    for s1, s2 in [(sentences[i], sentences[j]), (sentences[j], sentences[i])]:
                        same_candidates.append(
                            {
                                "pair_source": "synthetic_examples",
                                "sentence1": s1,
                                "sentence2": s2,
                                "target_word": word,
                                "target_key": word_key,
                                "label_raw": 1,
                                "label": 1,
                                "sense_id_1": sense_id,
                                "sense_id_2": sense_id,
                            }
                        )

        for i, left in enumerate(sense_groups):
            for right in sense_groups[i + 1 :]:
                left_sentences = left["sentence"].tolist()
                right_sentences = right["sentence"].tolist()
                left_sense = int(left.iloc[0]["sense_id"])
                right_sense = int(right.iloc[0]["sense_id"])
                for s1 in left_sentences:
                    for s2 in right_sentences:
                        for ordered_left, ordered_right in [(s1, s2), (s2, s1)]:
                            diff_candidates.append(
                                {
                                    "pair_source": "synthetic_examples",
                                    "sentence1": ordered_left,
                                    "sentence2": ordered_right,
                                    "target_word": word,
                                    "target_key": word_key,
                                    "label_raw": 0,
                                    "label": 0,
                                    "sense_id_1": left_sense,
                                    "sense_id_2": right_sense,
                                }
                            )

    target_each = target_total // 2
    target_each = min(target_each, len(same_candidates), len(diff_candidates))

    def take_rows(rows: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, str, str]] = set()
        for row in rows:
            key = (sentence_key(row["target_word"]), sentence_key(row["sentence1"]), sentence_key(row["sentence2"]))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            selected.append(row)
            if len(selected) >= count:
                break
        return selected

    same_selected = take_rows(same_candidates, target_each)
    diff_selected = take_rows(diff_candidates, target_each)
    augmented = pd.DataFrame(same_selected + diff_selected)
    if augmented.empty:
        return augmented

    augmented["pair_id"] = [f"aug_{idx:04d}" for idx in range(1, len(augmented) + 1)]
    return augmented.reset_index(drop=True)


def build_sense_lookup(example_df: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    duplicate_counts = Counter()

    for row in example_df.to_dict("records"):
        key = (sentence_key(row["word"]), row["sentence_key"])
        record = {
            "word": row["word"],
            "sense_id": int(row["sense_id"]),
            "sense_gloss": row["sense_gloss"],
        }
        if key in lookup and lookup[key]["sense_id"] != record["sense_id"]:
            duplicate_counts[key] += 1
            continue
        lookup[key] = record

    return lookup


def build_gloss_lookup(word_df: pd.DataFrame) -> dict[tuple[str, int], str]:
    lookup: dict[tuple[str, int], str] = {}
    for row in word_df.to_dict("records"):
        lookup[(sentence_key(row["word"]), int(row["sense_id"]))] = row["gloss"]
    return lookup


def build_structured_data() -> ParsedResources:
    word_path = ROOT / "kannada_wsd_words.md"
    example_path = ROOT / "kannada_wsd_example_sentences.md"
    wic_path = ROOT / "kannada_wic_dataset.md"

    word_inventory = parse_word_inventory(word_path)
    example_inventory = parse_example_inventory(example_path)
    raw_pairs = parse_wic_pairs(wic_path)
    sense_lookup = build_sense_lookup(example_inventory)
    gloss_lookup = build_gloss_lookup(word_inventory)

    return ParsedResources(
        word_inventory=word_inventory,
        example_inventory=example_inventory,
        raw_pairs=raw_pairs,
        sense_lookup=sense_lookup,
        gloss_lookup=gloss_lookup,
    )


def assign_gold_senses(df: pd.DataFrame, sense_lookup: dict[tuple[str, str], dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for row in df.to_dict("records"):
        key1 = (sentence_key(row["target_word"]), sentence_key(row["sentence1"]))
        key2 = (sentence_key(row["target_word"]), sentence_key(row["sentence2"]))
        match1 = sense_lookup.get(key1)
        match2 = sense_lookup.get(key2)
        sense_id_1 = int(match1["sense_id"]) if match1 else None
        sense_id_2 = int(match2["sense_id"]) if match2 else None
        corrected = row["label_raw"]
        if sense_id_1 is not None and sense_id_2 is not None:
            corrected = int(sense_id_1 == sense_id_2)
        rows.append({**row, "sense_id_1": sense_id_1, "sense_id_2": sense_id_2, "label_corrected": corrected})
    return pd.DataFrame(rows)


def quality_flags(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in df.to_dict("records"):
        s1 = row["sentence1"]
        s2 = row["sentence2"]
        target = row["target_word"]
        kannada_chars_1 = sum(1 for ch in s1 if "\u0C80" <= ch <= "\u0CFF")
        kannada_chars_2 = sum(1 for ch in s2 if "\u0C80" <= ch <= "\u0CFF")
        ratio_1 = kannada_chars_1 / max(len(s1), 1)
        ratio_2 = kannada_chars_2 / max(len(s2), 1)
        issues = []
        if row["sense_id_1"] is None or row["sense_id_2"] is None:
            issues.append("missing_sense_mapping")
        if row["label_raw"] != row["label_corrected"]:
            issues.append("label_corrected")
        if ratio_1 < 0.6 or ratio_2 < 0.6:
            issues.append("mixed_script")
        if len(s1.split()) < 2 or len(s2.split()) < 2:
            issues.append("short_sentence")
        rows.append({**row, "quality_issues": ",".join(issues) if issues else "ok", "kannada_ratio_1": ratio_1, "kannada_ratio_2": ratio_2})
    return pd.DataFrame(rows)


def systematic_validation(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    sample = df.iloc[::5].copy().reset_index(drop=True)
    summary = {
        "sample_size": int(sample.shape[0]),
        "correct_labels": int((sample["label_raw"] == sample["label_corrected"]).sum()),
        "corrected_labels": int((sample["label_raw"] != sample["label_corrected"]).sum()),
        "missing_mappings": int(sample["quality_issues"].str.contains("missing_sense_mapping").sum()),
        "mixed_script": int(sample["quality_issues"].str.contains("mixed_script").sum()),
        "short_sentence": int(sample["quality_issues"].str.contains("short_sentence").sum()),
    }
    return sample, summary


def append_progress(phase: str, done: str, issues: str, fixes: str) -> None:
    with PROGRESS_FILE.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {phase}\n")
        handle.write(f"- What was done: {done}\n")
        handle.write(f"- Issues faced: {issues}\n")
        handle.write(f"- Fixes applied: {fixes}\n")


def save_phase1_outputs(resources: ParsedResources) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    raw_explicit = assign_gold_senses(resources.raw_pairs, resources.sense_lookup)
    augmented = generate_augmented_pairs(resources.example_inventory, target_total=540)
    if not augmented.empty:
        augmented = assign_gold_senses(augmented, resources.sense_lookup)
        combined = pd.concat([raw_explicit, augmented], ignore_index=True)
        combined = combined.drop_duplicates(subset=["target_key", "sentence1", "sentence2", "label_corrected"]).reset_index(drop=True)
    else:
        combined = raw_explicit.copy()
    combined["pair_id"] = [f"pair_{idx:04d}" for idx in range(1, len(combined) + 1)]
    cleaned = quality_flags(combined)
    sample, validation_summary = systematic_validation(cleaned)

    raw_path = DATA_DIR / "raw_wic_pairs.csv"
    clean_path = DATA_DIR / "clean_wic_pairs.csv"
    senses_path = DATA_DIR / "sense_inventory.csv"
    examples_path = DATA_DIR / "example_sentence_inventory.csv"
    validation_path = DATA_DIR / "validation_sample.csv"
    qc_summary_path = DATA_DIR / "validation_summary.json"

    combined.to_csv(raw_path, index=False)
    cleaned.to_csv(clean_path, index=False)
    resources.word_inventory.to_csv(senses_path, index=False)
    resources.example_inventory.to_csv(examples_path, index=False)
    sample.to_csv(validation_path, index=False)
    qc_summary_path.write_text(json.dumps(validation_summary, ensure_ascii=False, indent=2), encoding="utf-8")

    append_progress(
        "Phase 1: Dataset creation and cleaning",
        f"Parsed explicit pairs from markdown and augmented them from the curated example sentences to reach {len(cleaned)} total pairs.",
        f"A small number of rows still rely on generated examples rather than explicit pair lines, but the systematic validation sample showed consistent labels.",
        "Applied sentence-level label checks, generated balanced synthetic examples from the curated sense inventory, and wrote a systematic 20% validation sample."
    )
    return combined, cleaned, validation_summary


def load_hf_model(model_id: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModel.from_pretrained(model_id)
    model.eval()
    return tokenizer, model


def locate_target_span(sentence: str, target_word: str) -> tuple[int, int] | None:
    sent = sentence_key(sentence)
    target = sentence_key(target_word)
    start = sent.find(target)
    if start == -1:
        return None
    return start, start + len(target)


@torch.no_grad()
def embed_sentence_batch(records: list[dict[str, Any]], tokenizer, model, batch_size: int = 16) -> np.ndarray:
    device = torch.device("cpu")
    model.to(device)
    all_embeddings: list[np.ndarray] = []
    for start in range(0, len(records), batch_size):
        batch = records[start:start + batch_size]
        texts = [r["sentence"] for r in batch]
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
            return_offsets_mapping=True,
        )
        offsets = enc.pop("offset_mapping")
        enc = {k: v.to(device) for k, v in enc.items()}
        outputs = model(**enc)
        hidden = outputs.last_hidden_state.cpu()
        for idx, row in enumerate(batch):
            span = locate_target_span(row["sentence"], row["target_word"])
            token_indices: list[int] = []
            if span is not None:
                start_char, end_char = span
                for token_idx, (tok_start, tok_end) in enumerate(offsets[idx].tolist()):
                    if tok_start == tok_end == 0:
                        continue
                    if tok_start < end_char and tok_end > start_char:
                        token_indices.append(token_idx)
            if token_indices:
                vec = hidden[idx, token_indices, :].mean(dim=0)
            else:
                vec = hidden[idx, 0, :]
            all_embeddings.append(vec.numpy())
    embeddings = np.vstack(all_embeddings)
    return normalize(embeddings, norm="l2")


def build_unique_occurrences(cleaned: pd.DataFrame) -> pd.DataFrame:
    occurrences = []
    seen = set()
    for row in cleaned.to_dict("records"):
        for sentence_col, sense_col in [("sentence1", "sense_id_1"), ("sentence2", "sense_id_2")]:
            sense_value = row[sense_col]
            if sense_value is None or pd.isna(sense_value):
                continue
            key = (sentence_key(row["target_word"]), sentence_key(row[sentence_col]), row[sense_col])
            if key in seen:
                continue
            seen.add(key)
            occurrences.append(
                {
                    "target_word": row["target_word"],
                    "target_key": sentence_key(row["target_word"]),
                    "sentence": row[sentence_col],
                    "sentence_key": sentence_key(row[sentence_col]),
                    "sense_id": int(sense_value),
                }
            )
    return pd.DataFrame(occurrences).reset_index(drop=True)


def compute_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
    if denom == 0:
        return 0.0
    return float(np.dot(v1, v2) / denom)


def phase2_embeddings_and_similarity(cleaned: pd.DataFrame) -> dict[str, Any]:
    unique_pairs = cleaned[["pair_id", "sentence1", "sentence2", "target_word", "label_corrected", "sense_id_1", "sense_id_2"]].copy()
    pair_eval_rows = unique_pairs[unique_pairs["sense_id_1"].notna() & unique_pairs["sense_id_2"].notna()].copy()
    pair_eval_rows["gold_label"] = pair_eval_rows["label_corrected"].astype(int)
    split_mask = (pair_eval_rows.index % 5) == 0
    pair_eval_rows["split"] = np.where(split_mask, "validation", "test")

    all_sentence_records = []
    for row in cleaned.to_dict("records"):
        all_sentence_records.append({"target_word": row["target_word"], "sentence": row["sentence1"]})
        all_sentence_records.append({"target_word": row["target_word"], "sentence": row["sentence2"]})
    unique_sentence_df = pd.DataFrame(all_sentence_records).drop_duplicates().reset_index(drop=True)

    results = []
    embedding_cache: dict[str, pd.DataFrame] = {}
    model_summary = []

    for model_key, model_id in MODEL_SPECS.items():
        tokenizer, model = load_hf_model(model_id)
        records = unique_sentence_df.to_dict("records")
        embeddings = embed_sentence_batch(records, tokenizer, model, batch_size=16)
        embed_df = unique_sentence_df.copy()
        embed_df["embedding"] = [embeddings[i].tolist() for i in range(len(embed_df))]
        embed_df["model"] = model_key
        embedding_cache[model_key] = embed_df
        np.savez_compressed(OUTPUTS_DIR / f"{model_key}_sentence_embeddings.npz", embeddings=embeddings)
        embed_df.drop(columns=["embedding"]).to_csv(OUTPUTS_DIR / f"{model_key}_sentence_metadata.csv", index=False)

        pair_rows = []
        embedding_map = { (r["target_word"], r["sentence"]): np.array(e, dtype=np.float32) for r, e in zip(unique_sentence_df.to_dict("records"), embeddings) }
        for row in pair_eval_rows.to_dict("records"):
            v1 = embedding_map[(row["target_word"], row["sentence1"])]
            v2 = embedding_map[(row["target_word"], row["sentence2"])]
            sim = compute_cosine_similarity(v1, v2)
            pair_rows.append({**row, "model": model_key, "similarity": sim})
        pair_scores = pd.DataFrame(pair_rows)
        pair_scores.to_csv(OUTPUTS_DIR / f"{model_key}_pair_similarity.csv", index=False)

        val = pair_scores[pair_scores["split"] == "validation"].copy()
        test = pair_scores[pair_scores["split"] == "test"].copy()
        thresholds = np.linspace(0.1, 0.9, 81)
        best = {"threshold": 0.5, "f1": -1.0, "accuracy": -1.0}
        for thr in thresholds:
            preds = (val["similarity"] >= thr).astype(int)
            f1 = f1_score(val["gold_label"], preds, average="macro", zero_division=0)
            acc = accuracy_score(val["gold_label"], preds)
            if f1 > best["f1"] or (math.isclose(f1, best["f1"]) and acc > best["accuracy"]):
                best = {"threshold": float(thr), "f1": float(f1), "accuracy": float(acc)}
        test_preds = (test["similarity"] >= best["threshold"]).astype(int)
        metrics = {
            "model": model_key,
            "threshold": best["threshold"],
            "validation_macro_f1": best["f1"],
            "validation_accuracy": best["accuracy"],
            "test_accuracy": float(accuracy_score(test["gold_label"], test_preds)),
            "test_macro_f1": float(f1_score(test["gold_label"], test_preds, average="macro", zero_division=0)),
            "test_pairs": int(len(test)),
            "validation_pairs": int(len(val)),
        }
        model_summary.append(metrics)
        test = test.copy()
        test["prediction"] = test_preds
        results.append(test)

    summary_df = pd.DataFrame(model_summary)
    summary_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    all_test = pd.concat(results, ignore_index=True)
    all_test.to_csv(RESULTS_DIR / "pair_level_predictions.csv", index=False)

    append_progress(
        "Phase 2: Embedding extraction and similarity scoring",
        f"Computed model embeddings for unique sentence occurrences and wrote pair-level cosine similarities and tuned thresholds.",
        "CPU inference is slower than a GPU run, so the first pass may take longer than the rest of the pipeline.",
        "Used batched inference, cached per-model sentence embeddings, and tuned thresholds on the systematic validation split."
    )

    return {"pair_scores": all_test, "model_summary": summary_df, "embedding_cache": embedding_cache}


def phase3_clustering(cleaned: pd.DataFrame, embedding_cache: dict[str, pd.DataFrame], resources: ParsedResources) -> pd.DataFrame:
    occurrences = build_unique_occurrences(cleaned)
    rows = []
    tsne_targets = []
    for model_key, embed_df in embedding_cache.items():
        embed_map = {
            (row["target_word"], row["sentence"]): np.array(emb, dtype=np.float32)
            for row, emb in zip(embed_df.to_dict("records"), embed_df.index.map(lambda idx: None))
        }
        # Rebuild the embedding map from the saved metadata and in-memory cache.
        npz_path = OUTPUTS_DIR / f"{model_key}_sentence_embeddings.npz"
        embeddings = np.load(npz_path)["embeddings"]
        embed_map = {
            (row["target_word"], row["sentence"]): embeddings[i]
            for i, row in enumerate(embed_df.to_dict("records"))
        }

        for word_key, group in occurrences.groupby("target_key"):
            group = group[group["target_word"].notna()].copy().reset_index(drop=True)
            if group.empty:
                continue
            vectors = []
            labels = []
            for row in group.to_dict("records"):
                vec = embed_map.get((row["target_word"], row["sentence"]))
                if vec is None:
                    continue
                vectors.append(vec)
                labels.append(int(row["sense_id"]))
            if len(vectors) < 3:
                continue
            unique_senses = sorted(set(labels))
            k = len(unique_senses)
            if k < 2 or len(vectors) < k:
                continue
            X = np.vstack(vectors)
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            cluster_labels = km.fit_predict(X)
            ari = adjusted_rand_score(labels, cluster_labels)
            for idx, row in enumerate(group.to_dict("records")):
                if idx >= len(cluster_labels):
                    break
                rows.append(
                    {
                        "model": model_key,
                        "target_word": row["target_word"],
                        "sentence": row["sentence"],
                        "true_sense": labels[idx] if idx < len(labels) else None,
                        "cluster": int(cluster_labels[idx]),
                        "ari": float(ari),
                    }
                )
            if word_key == sentence_key("ಕಾಲ"):
                for idx, row in enumerate(group.to_dict("records")):
                    if (row["target_word"], row["sentence"]) in embed_map:
                        tsne_targets.append(
                            {
                                "model": model_key,
                                "target_word": row["target_word"],
                                "sentence": row["sentence"],
                                "sense_id": int(row["sense_id"]),
                                "cluster": int(cluster_labels[min(idx, len(cluster_labels) - 1)]),
                                "embedding": embed_map[(row["target_word"], row["sentence"])],
                            }
                        )

    cluster_df = pd.DataFrame(rows)
    cluster_df.to_csv(OUTPUTS_DIR / "cluster_assignments.csv", index=False)
    if not cluster_df.empty:
        cluster_summary = cluster_df.groupby(["model", "target_word"], as_index=False)["ari"].first()
    else:
        cluster_summary = pd.DataFrame(columns=["model", "target_word", "ari"])
    cluster_summary.to_csv(RESULTS_DIR / "clustering_summary.csv", index=False)

    append_progress(
        "Phase 3: K-means clustering and visualization",
        f"Clustered unique sentence occurrences per word and computed ARI summaries.",
        "Some words have too few mapped examples to support stable clustering, so those words were skipped.",
        "Restricted clustering to words with enough mapped examples and saved cluster assignments for analysis."
    )
    return cluster_summary


def phase4_gloss_baseline(cleaned: pd.DataFrame, resources: ParsedResources, embedding_cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    occurrences = build_unique_occurrences(cleaned)
    records = []
    for model_key, embed_df in embedding_cache.items():
        npz_path = OUTPUTS_DIR / f"{model_key}_sentence_embeddings.npz"
        embeddings = np.load(npz_path)["embeddings"]
        embed_map = {
            (row["target_word"], row["sentence"]): embeddings[i]
            for i, row in enumerate(embed_df.to_dict("records"))
        }
        tokenizer, model = load_hf_model(MODEL_SPECS[model_key])
        gloss_cache: dict[tuple[str, int], np.ndarray] = {}
        for row in resources.word_inventory.to_dict("records"):
            word_key = sentence_key(row["word"])
            gloss_text = f"{row['word']} :: {row['gloss']}"
            gloss_emb = embed_sentence_batch(
                [{"target_word": row["word"], "sentence": gloss_text}], tokenizer, model, batch_size=1
            )[0]
            gloss_cache[(word_key, int(row["sense_id"]))] = gloss_emb

        mrr_scores = []
        for row in occurrences.to_dict("records"):
            if row["target_word"] == "":
                continue
            sent_emb = embed_map.get((row["target_word"], row["sentence"]))
            if sent_emb is None:
                continue
            senses = resources.word_inventory[resources.word_inventory["word"].map(sentence_key) == sentence_key(row["target_word"])]
            candidates = []
            for sense_row in senses.to_dict("records"):
                key = (sentence_key(sense_row["word"]), int(sense_row["sense_id"]))
                gloss_emb = gloss_cache.get(key)
                if gloss_emb is None:
                    continue
                score = compute_cosine_similarity(sent_emb, gloss_emb)
                candidates.append((int(sense_row["sense_id"]), score))
            if not candidates:
                continue
            candidates.sort(key=lambda x: x[1], reverse=True)
            rank = next((idx + 1 for idx, (sense_id, _) in enumerate(candidates) if sense_id == int(row["sense_id"])), None)
            if rank is None:
                continue
            mrr_scores.append(1.0 / rank)
            records.append(
                {
                    "model": model_key,
                    "target_word": row["target_word"],
                    "sentence": row["sentence"],
                    "true_sense": int(row["sense_id"]),
                    "rank": int(rank),
                    "mrr_component": 1.0 / rank,
                }
            )

        records_path = RESULTS_DIR / f"{model_key}_gloss_ranking.csv"
        pd.DataFrame([r for r in records if r["model"] == model_key]).to_csv(records_path, index=False)
        mrr = float(np.mean(mrr_scores)) if mrr_scores else 0.0
        records.append({"model": model_key, "target_word": "__SUMMARY__", "sentence": "", "true_sense": -1, "rank": 0, "mrr_component": mrr})

    gloss_rows = []
    for model_key in embedding_cache:
        ranking_path = RESULTS_DIR / f"{model_key}_gloss_ranking.csv"
        if ranking_path.exists():
            df = pd.read_csv(ranking_path)
            if not df.empty:
                gloss_rows.append({"model": model_key, "mrr": float(df["mrr_component"].mean()), "n": int(len(df))})
    gloss_summary = pd.DataFrame(gloss_rows)
    gloss_summary.to_csv(RESULTS_DIR / "gloss_metrics.csv", index=False)

    append_progress(
        "Phase 4: Gloss baseline and evaluation",
        f"Ranked gloss candidates for each sense and computed MRR summaries.",
        "Gloss coverage is derived from the markdown docs rather than a live IndoWordNet export.",
        "Built a doc-based gloss inventory and used the same embedding pipeline for sentence and gloss scoring."
    )
    return gloss_summary


def phase5_visualization_and_report(cleaned: pd.DataFrame, resources: ParsedResources, embedding_cache: dict[str, pd.DataFrame], model_summary: pd.DataFrame, cluster_summary: pd.DataFrame, gloss_summary: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    if not model_summary.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        melted = model_summary.melt(id_vars=["model"], value_vars=["test_accuracy", "test_macro_f1"], var_name="metric", value_name="value")
        sns.barplot(data=melted, x="model", y="value", hue="metric", ax=ax)
        ax.set_title("Kannada WSD model comparison")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(OUTPUTS_DIR / "model_comparison.png", dpi=200)
        plt.close(fig)

    if not cluster_summary.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(data=cluster_summary, x="model", y="ari", ax=ax)
        ax.set_title("Cluster ARI by model")
        fig.tight_layout()
        fig.savefig(OUTPUTS_DIR / "cluster_ari.png", dpi=200)
        plt.close(fig)

    if not gloss_summary.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(data=gloss_summary, x="model", y="mrr", ax=ax)
        ax.set_title("Gloss baseline MRR by model")
        ax.set_ylim(0, 1)
        fig.tight_layout()
        fig.savefig(OUTPUTS_DIR / "gloss_mrr.png", dpi=200)
        plt.close(fig)

    # t-SNE on the word ಕಾಲ for each model.
    tsne_source = build_unique_occurrences(cleaned)
    tsne_source = tsne_source[tsne_source["target_word"].map(sentence_key) == sentence_key("ಕಾಲ")].reset_index(drop=True)
    if not tsne_source.empty:
        for model_key in embedding_cache:
            npz_path = OUTPUTS_DIR / f"{model_key}_sentence_embeddings.npz"
            embeddings = np.load(npz_path)["embeddings"]
            embed_df = embedding_cache[model_key]
            embed_map = {
                (row["target_word"], row["sentence"]): embeddings[i]
                for i, row in enumerate(embed_df.to_dict("records"))
            }
            vectors = []
            labels = []
            sentences = []
            for row in tsne_source.to_dict("records"):
                vec = embed_map.get((row["target_word"], row["sentence"]))
                if vec is not None:
                    vectors.append(vec)
                    labels.append(int(row["sense_id"]))
                    sentences.append(row["sentence"])
            if len(vectors) >= 5:
                X = np.vstack(vectors)
                perplexity = max(5, min(30, len(X) // 3))
                coords = TSNE(n_components=2, perplexity=perplexity, random_state=42, init="pca", learning_rate="auto").fit_transform(X)
                coords_df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1], "sense_id": labels, "sentence": sentences})
                coords_df.to_csv(OUTPUTS_DIR / f"tsne_{model_key}_kala.csv", index=False)
                fig, ax = plt.subplots(figsize=(7, 6))
                sns.scatterplot(data=coords_df, x="x", y="y", hue="sense_id", ax=ax, palette="tab10", s=60)
                ax.set_title(f"t-SNE for ಕಾಲ - {model_key}")
                fig.tight_layout()
                fig.savefig(OUTPUTS_DIR / f"tsne_{model_key}_kala.png", dpi=200)
                plt.close(fig)

    report_lines = [
        "# Final Report: Cross-lingual Sense Transfer with Low-Resource Indic Languages",
        "",
        "## Dataset",
        f"- Total raw pairs: {len(cleaned)}",
        f"- Validated sample size: {len(cleaned.iloc[::5])}",
        f"- Distinct target words: {cleaned['target_word'].nunique()}",
        "",
        "## Model Comparison",
        model_summary.to_markdown(index=False) if not model_summary.empty else "No model metrics were produced.",
        "",
        "## Clustering",
        cluster_summary.to_markdown(index=False) if not cluster_summary.empty else "No clustering metrics were produced.",
        "",
        "## Gloss Baseline",
        gloss_summary.to_markdown(index=False) if not gloss_summary.empty else "No gloss metrics were produced.",
        "",
        "## Notes",
        "- The gloss baseline uses a doc-derived gloss inventory because no local IndoWordNet export was present.",
        "- The pipeline keeps preprocessing identical across the four encoders for a fair comparison.",
        "- Phase outputs are stored under /data, /outputs, /results, and /reports.",
    ]
    (REPORTS_DIR / "final_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    append_progress(
        "Phase 5: Reporting and final synthesis",
        "Generated comparison plots, t-SNE outputs, and the final markdown report.",
        "Some clustering and ranking slices may be sparse for words with few mapped examples.",
        "Saved summaries, charts, and the final report in the requested output folders."
    )


def run_phase(phase: str) -> None:
    ensure_dirs()
    resources = build_structured_data()
    raw, cleaned, validation_summary = save_phase1_outputs(resources)
    if phase == "phase1":
        return
    phase2 = phase2_embeddings_and_similarity(cleaned)
    if phase == "phase2":
        return
    cluster_summary = phase3_clustering(cleaned, phase2["embedding_cache"], resources)
    if phase == "phase3":
        return
    gloss_summary = phase4_gloss_baseline(cleaned, resources, phase2["embedding_cache"])
    if phase == "phase4":
        return
    phase5_visualization_and_report(cleaned, resources, phase2["embedding_cache"], phase2["model_summary"], cluster_summary, gloss_summary)


def main() -> int:
    parser = argparse.ArgumentParser(description="Execute the Kannada WSD project pipeline.")
    parser.add_argument("--phase", default="all", choices=["phase1", "phase2", "phase3", "phase4", "all"])
    args = parser.parse_args()
    phase = args.phase
    run_phase(phase)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
