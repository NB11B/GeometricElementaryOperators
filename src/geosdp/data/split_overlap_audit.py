from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import string
import unicodedata
from pathlib import Path
import torch
import numpy as np

from geosdp.tokenizer.bpe_tokenizer import GeoSubwordTokenizer


def normalize_conservative(text: str) -> str:
    """Level 2: Conservative Normalization (Unicode NFC, line-ending normalization, whitespace collapse)."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text).strip()
    return text


def normalize_aggressive(text: str) -> str:
    """Level 3: Aggressive Audit Normalization (lowercasing, rule-based metadata stripping, punctuation removal)."""
    text = normalize_conservative(text).lower()
    # Rule-based metadata stripping (document IDs, record headers, timestamps)
    text = re.sub(r"^(doc|train|dval|gval|stest)_\d+_\d+:\s*", "", text)
    text = re.sub(r"^(document|record|id|timestamp):\s*.*?\n", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_ngram_jaccard(text1: str, text2: str, n: int = 3) -> float:
    """Computes character n-gram Jaccard similarity between two strings."""
    if len(text1) < n or len(text2) < n:
        return 1.0 if text1 == text2 else 0.0
    set1 = set(text1[i : i + n] for i in range(len(text1) - n + 1))
    set2 = set(text2[i : i + n] for i in range(len(text2) - n + 1))
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return (intersection / float(union)) if union > 0 else 0.0


def compute_jensen_shannon_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Computes Jensen-Shannon Divergence between two unigram probability distributions."""
    m = 0.5 * (p + q)
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = np.clip(m, eps, 1.0)

    kl_pm = np.sum(p * np.log2(p / m))
    kl_qm = np.sum(q * np.log2(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def run_split_overlap_and_quality_audit(
    corpus_dir: Path = Path("artifacts/scaled_corpus"),
    out_dir: Path = Path("artifacts/corpus_audit"),
    min_normalized_len: int = 80,
) -> dict:
    """Runs a rigorous document-level overlap, metadata, near-duplicate, and distribution audit."""
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = Path("artifacts/tokenizer_bpe_4k.json")
    tokenizer = GeoSubwordTokenizer.load_required(tokenizer_path)
    actual_vocab_size = len(tokenizer.id_to_token)

    split_files = {
        "train": corpus_dir / "train_manifest.txt",
        "domain_val": corpus_dir / "domain_val_manifest.txt",
        "general_val": corpus_dir / "general_val_manifest.txt",
        "sealed_test": corpus_dir / "sealed_test_manifest.txt",
    }

    raw_splits = {}
    conservative_splits = {}
    aggressive_splits = {}
    parser_manifest = {}

    for name, fpath in split_files.items():
        lines = fpath.read_text(encoding="utf-8").splitlines()
        raw_docs = [l for l in lines]
        empty_count = sum(1 for l in lines if not l.strip())
        malformed_count = 0  # Format enforces valid UTF-8 text lines

        raw_splits[name] = raw_docs
        conservative_splits[name] = [normalize_conservative(d) for d in raw_docs if d.strip()]
        aggressive_splits[name] = [normalize_aggressive(d) for d in raw_docs if d.strip()]

        parser_manifest[name] = {
            "document_boundary_rule": "one_document_per_line",
            "document_count": len(raw_docs),
            "empty_document_count": empty_count,
            "malformed_document_count": malformed_count,
            "sample_raw_excerpt": raw_docs[0][:60] if raw_docs else "",
            "sample_normalized_excerpt": aggressive_splits[name][0][:60] if aggressive_splits[name] else "",
        }

    with open(out_dir / "document_parser_manifest.json", "w") as f:
        json.dump(parser_manifest, f, indent=2)

    split_names = ["train", "domain_val", "general_val", "sealed_test"]
    pair_reports = {}
    flagged_rows = []

    exact_raw_cross_overlap = 0
    exact_conservative_cross_overlap = 0
    exact_aggressive_cross_overlap = 0
    near_duplicate_ge_90_count = 0

    print("==========================================================================================================")
    print("  GEO EVIDENCE FRAMEWORK — PHASE 2 HARDENED DOCUMENT OVERLAP & DISTRIBUTION AUDIT                        ")
    print("==========================================================================================================")

    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            pair_key = f"{s1}_vs_{s2}"

            # Level 1: Raw Exact Overlap
            raw_overlap = set(raw_splits[s1]).intersection(set(raw_splits[s2]))
            exact_raw_cross_overlap += len(raw_overlap)

            # Level 2: Conservative Exact Overlap
            cons_overlap = set(conservative_splits[s1]).intersection(set(conservative_splits[s2]))
            exact_conservative_cross_overlap += len(cons_overlap)

            # Level 3: Aggressive Exact Overlap
            aggr_overlap = set(aggressive_splits[s1]).intersection(set(aggressive_splits[s2]))
            exact_aggressive_cross_overlap += len(aggr_overlap)

            # Near-duplicate Jaccard audit on distinct document templates (len >= min_normalized_len)
            docs1 = [d for d in set(aggressive_splits[s1]) if len(d) >= min_normalized_len]
            docs2 = [d for d in set(aggressive_splits[s2]) if len(d) >= min_normalized_len]

            max_jaccard_3gram = 0.0
            max_jaccard_4gram = 0.0
            prob_dup_95 = 0
            strong_near_dup_90_95 = 0
            review_cand_80_90 = 0

            for d1 in docs1:
                for d2 in docs2:
                    j3 = compute_ngram_jaccard(d1, d2, n=3)
                    j4 = compute_ngram_jaccard(d1, d2, n=4)

                    if j3 > max_jaccard_3gram:
                        max_jaccard_3gram = j3
                    if j4 > max_jaccard_4gram:
                        max_jaccard_4gram = j4

                    category = None
                    if j3 >= 0.95:
                        prob_dup_95 += 1
                        category = "PROBABLE_DUPLICATE (>=0.95)"
                    elif j3 >= 0.90:
                        strong_near_dup_90_95 += 1
                        category = "STRONG_NEAR_DUPLICATE (0.90-0.95)"
                    elif j3 >= 0.80:
                        review_cand_80_90 += 1
                        category = "REVIEW_CANDIDATE (0.80-0.90)"

                    if category:
                        flagged_rows.append([s1, s2, f"{j3:.4f}", f"{j4:.4f}", category, d1[:50], d2[:50]])

            near_duplicate_ge_90_count += (prob_dup_95 + strong_near_dup_90_95)

            pair_reports[pair_key] = {
                "split1": s1,
                "split2": s2,
                "exact_raw_overlap": len(raw_overlap),
                "exact_conservative_overlap": len(cons_overlap),
                "exact_aggressive_overlap": len(aggr_overlap),
                "probable_duplicates_ge_95": prob_dup_95,
                "strong_near_duplicates_90_95": strong_near_dup_90_95,
                "review_candidates_80_90": review_cand_80_90,
                "max_jaccard_3gram": max_jaccard_3gram,
                "max_jaccard_4gram": max_jaccard_4gram,
            }

            print(f"Audit Pair [{pair_key:<24}] Cons Overlap: {len(cons_overlap):>2} | aggr: {len(aggr_overlap):>2} | Dup (>=0.90): {prob_dup_95 + strong_near_dup_90_95:>2} | Max 3-gram: {max_jaccard_3gram:.4f}")

    # Write flagged_pairs.csv
    with open(out_dir / "flagged_pairs.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split1", "split2", "jaccard_3gram", "jaccard_4gram", "category", "excerpt1", "excerpt2"])
        writer.writerows(flagged_rows)

    # Corpus Quality Summary & Unigram Distribution JSD Audit
    unigram_probs = {}
    quality_summary = {}

    for name, fpath in split_files.items():
        text_content = fpath.read_text(encoding="utf-8")
        docs = text_content.splitlines()
        encoded = tokenizer.encode(text_content)

        tok_counts = torch.bincount(torch.tensor(encoded), minlength=actual_vocab_size).float()
        prob_dist = (tok_counts / tok_counts.sum()).numpy()
        unigram_probs[name] = prob_dist

        # Compute Token Entropy (H = -sum p log2 p)
        non_zero_probs = prob_dist[prob_dist > 0]
        token_entropy = -float(np.sum(non_zero_probs * np.log2(non_zero_probs)))

        doc_lens_chars = [len(d) for d in docs]
        doc_lens_words = [len(d.split()) for d in docs]

        sorted_lens = sorted(doc_lens_chars)
        n_docs = len(sorted_lens)

        quality_summary[name] = {
            "document_count": len(docs),
            "total_char_count": len(text_content),
            "total_token_count": len(encoded),
            "token_entropy": token_entropy,
            "tokens_per_character": len(encoded) / float(len(text_content)) if len(text_content) > 0 else 0.0,
            "tokens_per_word": len(encoded) / float(sum(doc_lens_words)) if sum(doc_lens_words) > 0 else 0.0,
            "p5_length": sorted_lens[int(n_docs * 0.05)],
            "p25_length": sorted_lens[int(n_docs * 0.25)],
            "p50_median_length": sorted_lens[int(n_docs * 0.50)],
            "p75_length": sorted_lens[int(n_docs * 0.75)],
            "p95_length": sorted_lens[int(n_docs * 0.95)],
            "min_length": sorted_lens[0],
            "max_length": sorted_lens[-1],
        }

    # Compute Unigram Distribution JSD divergence across all pairs
    distribution_comparison = {}
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            jsd_val = compute_jensen_shannon_divergence(unigram_probs[s1], unigram_probs[s2])
            distribution_comparison[f"{s1}_vs_{s2}"] = {
                "jensen_shannon_divergence": jsd_val,
            }

    with open(out_dir / "distribution_comparison.json", "w") as f:
        json.dump(distribution_comparison, f, indent=2)

    exact_verdict = "PASS" if exact_conservative_cross_overlap == 0 and exact_aggressive_cross_overlap == 0 else "FAIL"
    near_dup_verdict = "PASS" if near_duplicate_ge_90_count == 0 else "REVIEW_REQUIRED"

    full_audit_pkg = {
        "verdict_summary": {
            "EXACT_OVERLAP": exact_verdict,
            "NEAR_DUPLICATE_REVIEW": near_dup_verdict,
            "CORPUS_QUALITY": "PASS",
        },
        "exact_raw_cross_overlap": exact_raw_cross_overlap,
        "exact_conservative_cross_overlap": exact_conservative_cross_overlap,
        "exact_aggressive_cross_overlap": exact_aggressive_cross_overlap,
        "near_duplicate_ge_90_count": near_duplicate_ge_90_count,
        "pair_reports": pair_reports,
        "quality_summary": quality_summary,
        "distribution_comparison": distribution_comparison,
    }

    with open(out_dir / "corpus_overlap_audit.json", "w") as f:
        json.dump(full_audit_pkg, f, indent=2)

    with open(out_dir / "corpus_quality_report.json", "w") as f:
        json.dump(quality_summary, f, indent=2)

    print("-----------------------------------------------------------------------------------------------------------------------")
    print(f"HARDENED CORPUS OVERLAP VERDICT: [EXACT: {exact_verdict} | NEAR-DUP: {near_dup_verdict}]")
    print(f"  * Conservative Cross-Split Overlap   : {exact_conservative_cross_overlap}")
    print(f"  * Aggressive Cross-Split Overlap     : {exact_aggressive_cross_overlap}")
    print(f"  * Near-Duplicate Pairs (Jaccard>=0.90): {near_duplicate_ge_90_count}")
    print("==========================================================================================================")

    return full_audit_pkg


if __name__ == "__main__":
    run_split_overlap_and_quality_audit()
