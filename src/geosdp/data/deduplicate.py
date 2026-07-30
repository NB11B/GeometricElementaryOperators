from __future__ import annotations

import hashlib
from typing import List, Set, Tuple


def compute_doc_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def deduplicate_documents(documents: List[str]) -> Tuple[List[str], int]:
    seen: Set[str] = set()
    deduped: List[str] = []
    duplicates_count = 0

    for doc in documents:
        doc_hash = compute_doc_hash(doc)
        if doc_hash in seen:
            duplicates_count += 1
        else:
            seen.add(doc_hash)
            deduped.append(doc)

    return deduped, duplicates_count
