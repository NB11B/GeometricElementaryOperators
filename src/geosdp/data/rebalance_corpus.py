from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
import torch

from geosdp.tokenizer.bpe_tokenizer import GeoSubwordTokenizer


def rebalance_and_resplit_corpus(
    corpus_dir: Path = Path("artifacts/scaled_corpus"),
    seed: int = 42,
) -> dict:
    """Rebalances and re-splits the corpus into i.i.d. splits from a clean document pool to eliminate train-val distribution shift."""
    print("==========================================================================================================")
    print("  GEO EVIDENCE FRAMEWORK — STEP B CORPUS REBALANCING & RE-SPLITTING                                       ")
    print("==========================================================================================================")

    tokenizer_path = Path("artifacts/tokenizer_bpe_4k.json")
    tokenizer = GeoSubwordTokenizer.load_required(tokenizer_path)

    # Collect all unique document lines from existing manifests
    manifest_files = [
        corpus_dir / "train_manifest.txt",
        corpus_dir / "domain_val_manifest.txt",
        corpus_dir / "general_val_manifest.txt",
        corpus_dir / "sealed_test_manifest.txt",
    ]

    all_raw_docs = []
    for mf in manifest_files:
        if mf.exists():
            lines = mf.read_text(encoding="utf-8").splitlines()
            all_raw_docs.extend([l.strip() for l in lines if l.strip()])

    unique_docs = list(dict.fromkeys(all_raw_docs))
    n_docs = len(unique_docs)
    print(f"Total Unique Documents Collected: {n_docs}")

    # Deterministic shuffle
    rng = random.Random(seed)
    rng.shuffle(unique_docs)

    # Split sizes: 70% train, 10% domain_val, 10% general_val, 10% sealed_test
    n_train = int(n_docs * 0.70)
    n_dval = int(n_docs * 0.10)
    n_gval = int(n_docs * 0.10)

    train_docs = unique_docs[:n_train]
    dval_docs = unique_docs[n_train : n_train + n_dval]
    gval_docs = unique_docs[n_train + n_dval : n_train + n_dval + n_gval]
    stest_docs = unique_docs[n_train + n_dval + n_gval :]

    # Assert zero exact cross-split overlap
    set_train = set(train_docs)
    set_dval = set(dval_docs)
    set_gval = set(gval_docs)
    set_stest = set(stest_docs)

    assert len(set_train.intersection(set_dval)) == 0, "Cross-split overlap detected: train vs domain_val"
    assert len(set_train.intersection(set_gval)) == 0, "Cross-split overlap detected: train vs general_val"
    assert len(set_train.intersection(set_stest)) == 0, "Cross-split overlap detected: train vs sealed_test"
    assert len(set_dval.intersection(set_gval)) == 0, "Cross-split overlap detected: domain_val vs general_val"
    assert len(set_dval.intersection(set_stest)) == 0, "Cross-split overlap detected: domain_val vs sealed_test"
    assert len(set_gval.intersection(set_stest)) == 0, "Cross-split overlap detected: general_val vs sealed_test"

    # Write manifests
    (corpus_dir / "train_manifest.txt").write_text("\n".join(train_docs), encoding="utf-8")
    (corpus_dir / "domain_val_manifest.txt").write_text("\n".join(dval_docs), encoding="utf-8")
    (corpus_dir / "general_val_manifest.txt").write_text("\n".join(gval_docs), encoding="utf-8")
    (corpus_dir / "sealed_test_manifest.txt").write_text("\n".join(stest_docs), encoding="utf-8")

    # Encode train_tokens.pt from new train_manifest.txt
    train_manifest_text = "\n".join(train_docs)
    train_tokens_list = tokenizer.encode(train_manifest_text)
    train_tokens_tensor = torch.tensor(train_tokens_list, dtype=torch.long)
    torch.save(train_tokens_tensor, corpus_dir / "train_tokens.pt")

    summary = {
        "seed": seed,
        "total_unique_documents": n_docs,
        "train_document_count": len(train_docs),
        "domain_val_document_count": len(dval_docs),
        "general_val_document_count": len(gval_docs),
        "sealed_test_document_count": len(stest_docs),
        "train_tokens_count": train_tokens_tensor.numel(),
        "train_manifest_sha256": hashlib.sha256((corpus_dir / "train_manifest.txt").read_bytes()).hexdigest(),
        "domain_val_manifest_sha256": hashlib.sha256((corpus_dir / "domain_val_manifest.txt").read_bytes()).hexdigest(),
        "general_val_manifest_sha256": hashlib.sha256((corpus_dir / "general_val_manifest.txt").read_bytes()).hexdigest(),
        "sealed_test_manifest_sha256": hashlib.sha256((corpus_dir / "sealed_test_manifest.txt").read_bytes()).hexdigest(),
    }

    with open(corpus_dir / "rebalanced_corpus_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("-----------------------------------------------------------------------------------------------------------------------")
    print(f"CORPUS REBALANCING COMPLETE:")
    print(f"  * Train Documents       : {len(train_docs)} ({train_tokens_tensor.numel():,} tokens)")
    print(f"  * Domain Val Documents  : {len(dval_docs)}")
    print(f"  * General Val Documents : {len(gval_docs)}")
    print(f"  * Sealed Test Documents : {len(stest_docs)}")
    print(f"  * Cross-Split Overlap   : 0 (PASSED)")
    print("==========================================================================================================")

    return summary


if __name__ == "__main__":
    rebalance_and_resplit_corpus()
