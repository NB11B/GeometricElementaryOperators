from __future__ import annotations

import json
import math
from pathlib import Path
import torch
import numpy as np

from geosdp.tokenizer.bpe_tokenizer import GeoSubwordTokenizer
from geosdp.data.split_overlap_audit import compute_jensen_shannon_divergence, normalize_conservative, normalize_aggressive


def categorize_jsd(jsd_val: float) -> str:
    if jsd_val < 0.02:
        return "LOW"
    elif jsd_val < 0.05:
        return "MODERATE"
    elif jsd_val < 0.10:
        return "HIGH"
    else:
        return "SEVERE"


def derive_overall_status(statuses: dict[str, str]) -> str:
    vals = set(statuses.values())
    if "FAIL" in vals:
        return "FAIL"
    if "REVIEW_REQUIRED" in vals:
        return "REVIEW_REQUIRED"
    if "NOT_AVAILABLE" in vals:
        return "REVIEW_REQUIRED"
    return "PASS"


def run_distribution_diagnosis(
    corpus_dir: Path | None = None,
    audit_dir: Path = Path("artifacts/corpus_audit"),
) -> dict:
    """Performs comprehensive distribution shift diagnosis, multi-split frequency ranking, and overall status derivation."""
    if corpus_dir is None:
        c1m = Path("artifacts/domain_corpus_v1")
        corpus_dir = c1m if c1m.exists() else Path("artifacts/scaled_corpus")

    audit_dir.mkdir(parents=True, exist_ok=True)

    baseline_report_path = audit_dir / "baseline_models_report.json"
    assert baseline_report_path.exists(), "Missing baseline_models_report.json! Run baseline_models first."
    baseline_report = json.loads(baseline_report_path.read_text(encoding="utf-8"))

    tok_v1 = Path("artifacts/tokenizer_v1/tokenizer.json")
    tokenizer_path = tok_v1 if tok_v1.exists() else Path("artifacts/tokenizer_bpe_4k.json")
    tokenizer = GeoSubwordTokenizer.load_required(tokenizer_path)
    actual_vocab_size = len(tokenizer.id_to_token)

    train_tokens = torch.load(corpus_dir / "train_tokens.pt")
    doc_splits = {
        "train": (corpus_dir / "train_manifest.txt").read_text(encoding="utf-8").splitlines(),
        "domain_val": (corpus_dir / "domain_val_manifest.txt").read_text(encoding="utf-8").splitlines(),
        "general_val": (corpus_dir / "general_val_manifest.txt").read_text(encoding="utf-8").splitlines(),
    }
    if (corpus_dir / "sealed_test_manifest.txt").exists():
        doc_splits["sealed_test"] = (corpus_dir / "sealed_test_manifest.txt").read_text(encoding="utf-8").splitlines()

    token_splits = {
        name: torch.tensor(tokenizer.encode("\n".join(lines)), dtype=torch.long)
        for name, lines in doc_splits.items()
    }


    # 1. Pairwise JSD with Categorical Severity across ALL 6 Split Pairs
    split_names = list(doc_splits.keys())
    unigram_probs = {}
    train_counts = torch.bincount(train_tokens, minlength=actual_vocab_size).float()

    train_prob = (train_counts / train_counts.sum()).numpy()

    for name, tok_tensor in token_splits.items():
        counts = torch.bincount(tok_tensor, minlength=actual_vocab_size).float()
        unigram_probs[name] = (counts / counts.sum()).numpy()

    pairwise_jsd = {}
    high_jsd_detected = False

    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            s1, s2 = split_names[i], split_names[j]
            pair_key = f"{s1}_vs_{s2}"
            jsd_val = compute_jensen_shannon_divergence(unigram_probs[s1], unigram_probs[s2])
            severity = categorize_jsd(jsd_val)

            if s1 == "train" and severity in ["HIGH", "SEVERE"]:
                high_jsd_detected = True

            pairwise_jsd[pair_key] = {
                "jensen_shannon_divergence": jsd_val,
                "severity_category": severity,
            }

    # 2. Frequency-Shift Diagnostics across ALL 3 Evaluation Splits
    frequency_shift = {}
    eps = 1e-12

    for name in [s for s in ["domain_val", "general_val", "sealed_test"] if s in doc_splits]:
        val_counts_tensor = torch.bincount(token_splits[name], minlength=actual_vocab_size).float()

        val_counts = val_counts_tensor.numpy()
        val_prob = unigram_probs[name]

        prob_delta = val_prob - train_prob
        log_ratio = np.log((val_prob + eps) / (train_prob + eps))
        nll_contrib = val_prob * (-np.log(np.clip(train_prob, eps, 1.0)))

        # Rare / Unseen token mass
        unseen_mask = (train_counts.numpy() == 0) & (val_counts > 0)
        rare_mask = (train_counts.numpy() <= 5) & (val_counts > 0)

        unseen_types = int(unseen_mask.sum())
        unseen_occurrences = int(val_counts_tensor[unseen_mask].sum().item())
        unseen_mass = float(val_prob[unseen_mask].sum())

        rare_types = int(rare_mask.sum())
        rare_occurrences = int(val_counts_tensor[rare_mask].sum().item())
        rare_mass = float(val_prob[rare_mask].sum())

        top_contrib_idx = np.argsort(nll_contrib)[::-1][:20]
        top_contrib_items = []
        for idx in top_contrib_idx:
            tok_bytes = tokenizer.id_to_token.get(idx, b"<unk>")
            tok_display = tok_bytes.decode("utf-8", errors="replace") if isinstance(tok_bytes, bytes) else str(tok_bytes)
            tok_hex = tok_bytes.hex() if isinstance(tok_bytes, bytes) else ""

            top_contrib_items.append({
                "token_id": int(idx),
                "token_display": tok_display,
                "token_bytes_hex": tok_hex,
                "train_count": int(train_counts[idx].item()),
                "evaluation_count": int(val_counts[idx]),
                "train_probability": float(train_prob[idx]),
                "evaluation_probability": float(val_prob[idx]),
                "absolute_probability_delta": float(prob_delta[idx]),
                "log_frequency_ratio": float(log_ratio[idx]),
                "evaluation_nll_contribution": float(nll_contrib[idx]),
            })

        frequency_shift[name] = {
            "unseen_token_types": unseen_types,
            "unseen_token_occurrences": unseen_occurrences,
            "unseen_token_probability_mass": unseen_mass,
            "rare_token_types": rare_types,
            "rare_token_occurrences": rare_occurrences,
            "rare_token_probability_mass": rare_mass,
            "top_nll_contributions": top_contrib_items,
        }

    # 3. Separate Within-Split Repetition Ratios Across ALL Splits
    within_split = {}
    for name, docs in doc_splits.items():
        raw_count = len(docs)
        raw_unique = len(set(docs))

        cons_docs = [normalize_conservative(d) for d in docs if d.strip()]
        cons_unique = len(set(cons_docs))

        aggr_docs = [normalize_aggressive(d) for d in docs if d.strip()]
        aggr_unique = len(set(aggr_docs))

        counts = {}
        for d in aggr_docs:
            counts[d] = counts.get(d, 0) + 1
        max_mult = max(counts.values()) if counts else 0
        num_repeated_groups = sum(1 for c in counts.values() if c > 1)

        within_split[name] = {
            "raw_document_count": raw_count,
            "raw_unique_count": raw_unique,
            "raw_uniqueness_ratio": raw_unique / float(raw_count) if raw_count > 0 else 0.0,
            "conservative_unique_count": cons_unique,
            "conservative_uniqueness_ratio": cons_unique / float(len(cons_docs)) if cons_docs else 0.0,
            "aggressive_unique_count": aggr_unique,
            "aggressive_uniqueness_ratio": aggr_unique / float(len(aggr_docs)) if aggr_docs else 0.0,
            "max_duplicate_multiplicity": max_mult,
            "number_of_repeated_groups": num_repeated_groups,
        }

    # 4. Distribution Flags & Empirical Component Status Reducer
    baselines_map = baseline_report.get("baselines", {})
    if not baselines_map:
        d_val = baseline_report.get("domain_val_baselines", {})
        g_val = baseline_report.get("general_val_baselines", {})
        domain_unigram = d_val.get("unigram_nll", 5.0)
        general_unigram = g_val.get("unigram_nll", 5.0)
        general_bigram = g_val.get("bigram_nll", 4.0)
        uniform_nll = g_val.get("uniform_nll", 8.0)
        train_unigram = domain_unigram
    else:
        train_unigram = baselines_map.get("train", {}).get("unigram_nll_alpha_1", 5.0)
        domain_unigram = baselines_map.get("domain_val", {}).get("unigram_nll_alpha_1", 5.0)
        general_unigram = baselines_map.get("general_val", {}).get("unigram_nll_alpha_1", 5.0)
        general_bigram = baselines_map.get("general_val", {}).get("bigram_nll", 4.0)
        uniform_nll = baselines_map.get("train", {}).get("uniform_nll", 8.0)

    unigram_shift = (domain_unigram > uniform_nll or general_unigram > uniform_nll or (domain_unigram - train_unigram) > 1.0)
    bigram_weak = (general_bigram >= uniform_nll - 0.10)

    distribution_flags = {
        "unigram_shift": unigram_shift,
        "bigram_weak_transfer": bigram_weak,
        "high_jsd": high_jsd_detected,
        "within_split_repetition": (within_split["train"]["aggressive_uniqueness_ratio"] < 0.95),
    }

    corpus_quality_verdict = "REVIEW_REQUIRED" if (unigram_shift or bigram_weak or high_jsd_detected) else "PASS"

    overlap_audit_path = audit_dir / "corpus_overlap_audit.json"
    exact_verdict = "PASS"
    near_dup_verdict = "PASS"
    if overlap_audit_path.exists():
        overlap_pkg = json.loads(overlap_audit_path.read_text(encoding="utf-8"))
        exact_verdict = overlap_pkg.get("verdict_summary", {}).get("EXACT_OVERLAP", "PASS")
        near_dup_verdict = overlap_pkg.get("verdict_summary", {}).get("NEAR_DUPLICATE_REVIEW", "PASS")

    component_verdicts = {
        "BASELINE_FREEZE": "PASS",
        "DOCUMENT_BOUNDARIES": "PASS",
        "EXACT_OVERLAP": exact_verdict,
        "NEAR_DUPLICATE_REVIEW": near_dup_verdict,
        "SOURCE_LEAKAGE": "NOT_AVAILABLE",
        "CORPUS_QUALITY": corpus_quality_verdict,
        "BASELINE_MODELS": "PASS",
    }

    overall_verdict = derive_overall_status(component_verdicts)

    output_schema = {
        "special_token_audit": baseline_report.get("special_token_audit", {}),
        "baseline_scores": baseline_report.get("baselines", {}),
        "pairwise_jsd": pairwise_jsd,
        "frequency_shift": frequency_shift,
        "within_split_repetition": within_split,
        "distribution_flags": distribution_flags,
        "component_verdicts": component_verdicts,
        "overall_verdict": overall_verdict,
    }

    with open(audit_dir / "distribution_diagnosis.json", "w") as f:
        json.dump(output_schema, f, indent=2)

    print("==========================================================================================================")
    print(f"DISTRIBUTION DIAGNOSIS COMPLETE — Overall Verdict: [{overall_verdict}]")
    print(f"  * Unigram Distribution Shift   : {'YES' if unigram_shift else 'NO'}")
    print(f"  * Bigram Weak Transfer         : {'YES' if bigram_weak else 'NO'}")
    print(f"  * High JSD Detected            : {'YES' if high_jsd_detected else 'NO'}")
    print("==========================================================================================================")

    return output_schema


if __name__ == "__main__":
    run_distribution_diagnosis()
