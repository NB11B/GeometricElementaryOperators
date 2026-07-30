from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List


class DatasetManifest:
    def __init__(self, dataset_name: str, source_files: List[Dict[str, Any]]) -> None:
        self.dataset_name = dataset_name
        self.source_files = source_files

    def fingerprint(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.dataset_name.encode("utf-8"))
        for item in sorted(self.source_files, key=lambda x: x["path"]):
            hasher.update(f"{item['path']}:{item.get('sha256', '')}".encode("utf-8"))
        return hasher.hexdigest()[:16]

    def save(self, path: Path) -> None:
        data = {
            "dataset_name": self.dataset_name,
            "fingerprint": self.fingerprint(),
            "source_files": self.source_files,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> DatasetManifest:
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["dataset_name"], data["source_files"])
