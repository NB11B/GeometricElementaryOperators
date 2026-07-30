from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


def get_character_ngrams(text: str, n: int = 5) -> set[str]:
    text_clean = re.sub(r"\s+", " ", text.lower()).strip()
    if len(text_clean) < n:
        return {text_clean}
    return {text_clean[i : i + n] for i in range(len(text_clean) - n + 1)}


def compute_jaccard_similarity(set_a: set[str], set_b: set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))
    return intersection / max(1, union)


def deduplicate_documents_pipeline(docs: list[dict], threshold: float = 0.90) -> tuple[list[dict], list[dict], dict]:
    """Multi-stage deduplication pipeline:

    1. Stable Source Record ID
    2. Raw SHA-256
    3. Normalized SHA-256
    4. Character 5-gram Jaccard Similarity >= threshold
    """
    seen_ids = set()
    seen_raw_hashes = set()
    seen_norm_hashes = set()

    stage1_2_3_clean = []
    dropped_exact = 0

    for d in docs:
        rec_id = d["source_record_id"]
        raw_h = d["raw_sha256"]
        norm_h = d["normalized_sha256"]

        if rec_id in seen_ids or raw_h in seen_raw_hashes or norm_h in seen_norm_hashes:
            dropped_exact += 1
            continue

        seen_ids.add(rec_id)
        seen_raw_hashes.add(raw_h)
        seen_norm_hashes.add(norm_h)
        stage1_2_3_clean.append(d)

    # Near-duplicate pairwise audit
    flagged_pairs = []
    final_clean = []
    ngrams_cache = [get_character_ngrams(d["text"], n=5) for d in stage1_2_3_clean]

    dropped_near_dup_indices = set()
    n = len(stage1_2_3_clean)

    for i in range(n):
        if i in dropped_near_dup_indices:
            continue
        final_clean.append(stage1_2_3_clean[i])

        for j in range(i + 1, n):
            if j in dropped_near_dup_indices:
                continue
            sim = compute_jaccard_similarity(ngrams_cache[i], ngrams_cache[j])
            if sim >= threshold:
                flagged_pairs.append({
                    "doc_id_1": stage1_2_3_clean[i]["document_id"],
                    "doc_id_2": stage1_2_3_clean[j]["document_id"],
                    "source_family_1": stage1_2_3_clean[i]["source_family"],
                    "source_family_2": stage1_2_3_clean[j]["source_family"],
                    "jaccard_similarity": sim,
                })
                dropped_near_dup_indices.add(j)

    report = {
        "input_documents": len(docs),
        "stage_1_2_3_clean_documents": len(stage1_2_3_clean),
        "exact_duplicates_dropped": dropped_exact,
        "near_duplicate_pairs_flagged": len(flagged_pairs),
        "near_duplicates_dropped": len(dropped_near_dup_indices),
        "final_deduplicated_documents": len(final_clean),
    }

    return final_clean, flagged_pairs, report
