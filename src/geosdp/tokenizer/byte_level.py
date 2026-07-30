from __future__ import annotations

import json
from pathlib import Path
from typing import List, Type, TypeVar

from .base import Tokenizer

T = TypeVar("T", bound="ByteLevelBPETokenizer")


class ByteLevelBPETokenizer(Tokenizer):
    """Deterministic byte-level BPE tokenizer for GEOSDP."""

    def __init__(
        self,
        vocab_size: int = 256 + 4,
        bos_id: int = 1,
        eos_id: int = 2,
        pad_id: int = 0,
        unk_id: int = 3,
    ) -> None:
        super().__init__(vocab_size, bos_id, eos_id, pad_id, unk_id)
        # Reserved special token mapping
        self.special_tokens = {
            self.pad_id: "<PAD>",
            self.bos_id: "<BOS>",
            self.eos_id: "<EOS>",
            self.unk_id: "<UNK>",
        }

    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        raw_bytes = text.encode("utf-8")
        tokens: List[int] = []
        if add_bos:
            tokens.append(self.bos_id)

        # Map each byte to token ID offset by 4 reserved special tokens
        for b in raw_bytes:
            token_id = b + 4
            if token_id >= self.vocab_size:
                tokens.append(self.unk_id)
            else:
                tokens.append(token_id)

        if add_eos:
            tokens.append(self.eos_id)
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        byte_list = bytearray()
        for t in token_ids:
            if t in self.special_tokens:
                continue
            if t >= 4:
                byte_list.append(t - 4)
        return byte_list.decode("utf-8", errors="replace")

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "tokenizer_type": "ByteLevelBPETokenizer",
            "vocab_size": self.vocab_size,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "pad_id": self.pad_id,
            "unk_id": self.unk_id,
            "fingerprint": self.fingerprint(),
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls: Type[T], path: Path) -> T:
        data = json.loads(path.read_text(encoding="utf-8"))
        tok = cls(
            vocab_size=data["vocab_size"],
            bos_id=data["bos_id"],
            eos_id=data["eos_id"],
            pad_id=data["pad_id"],
            unk_id=data["unk_id"],
        )
        return tok
