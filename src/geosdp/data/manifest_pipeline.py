from __future__ import annotations

import hashlib
import json
from pathlib import Path
import torch


def generate_and_save_manifests(corpus_path: Path, manifest_dir: Path, seq_len: int = 512) -> tuple[Path, Path, Path]:
    """Generate and serialize document split, train batch, and val sequence manifests to disk."""
    manifest_dir.mkdir(parents=True, exist_ok=True)

    text_records = []
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            query = rec.get("query", "")
            ans = rec.get("answer", "")
            text_records.append({"query": query, "answer": ans, "text": f"Query: {query}\nAnswer: {ans}\n"})

    # Document-Level Split: 124 Train (80%), 32 Val (20%)
    split_idx = int(len(text_records) * 0.8)
    train_recs = text_records[:split_idx]
    val_recs = text_records[split_idx:]

    doc_split = {
        "train_document_count": len(train_recs),
        "val_document_count": len(val_recs),
        "train_doc_hashes": [hashlib.sha256(r["text"].encode()).hexdigest()[:12] for r in train_recs],
        "val_doc_hashes": [hashlib.sha256(r["text"].encode()).hexdigest()[:12] for r in val_recs],
    }
    doc_split_path = manifest_dir / "document_split_manifest.json"
    doc_split_path.write_text(json.dumps(doc_split, indent=2), encoding="utf-8")

    # Serialize raw texts
    train_text = "\n".join(r["text"] for r in train_recs)
    val_text = "\n".join(r["text"] for r in val_recs)

    train_path = manifest_dir / "train_raw_text.txt"
    val_path = manifest_dir / "val_raw_text.txt"
    train_path.write_text(train_text, encoding="utf-8")
    val_path.write_text(val_text, encoding="utf-8")

    return doc_split_path, train_path, val_path
