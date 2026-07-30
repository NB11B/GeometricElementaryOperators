from __future__ import annotations

import abc
import hashlib
from pathlib import Path
from typing import List, Type, TypeVar

T = TypeVar("T", bound="Tokenizer")


class Tokenizer(abc.ABC):
    def __init__(
        self,
        vocab_size: int,
        bos_id: int = 1,
        eos_id: int = 2,
        pad_id: int = 0,
        unk_id: int = 3,
    ) -> None:
        self.vocab_size = vocab_size
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.unk_id = unk_id

    @abc.abstractmethod
    def encode(self, text: str, *, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        pass

    @abc.abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        pass

    @abc.abstractmethod
    def save(self, path: Path) -> None:
        pass

    @classmethod
    @abc.abstractmethod
    def load(cls: Type[T], path: Path) -> T:
        pass

    def fingerprint(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(f"vocab_size={self.vocab_size};bos={self.bos_id};eos={self.eos_id};pad={self.pad_id};unk={self.unk_id}".encode("utf-8"))
        return hasher.hexdigest()[:16]
