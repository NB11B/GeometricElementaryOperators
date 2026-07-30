from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re


class GeoSubwordTokenizer:
    """Subword BPE Tokenizer for GEO Language Model Training."""

    def __init__(self, vocab_path: Path | None = None, target_vocab_size: int = 8192):
        self.target_vocab_size = target_vocab_size
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3

        self.id_to_token: dict[int, bytes] = {
            0: b"<pad>",
            1: b"<bos>",
            2: b"<eos>",
            3: b"<unk>",
        }
        self.token_to_id: dict[bytes, int] = {v: k for k, v in self.id_to_token.items()}

        # Initialize base byte vocabulary [0..255]
        for b in range(256):
            byte_tok = bytes([b])
            if byte_tok not in self.token_to_id:
                idx = len(self.id_to_token)
                self.id_to_token[idx] = byte_tok
                self.token_to_id[byte_tok] = idx

        self.merges: dict[tuple[int, int], int] = {}
        self.merge_frequencies: dict[tuple[int, int], int] = {}
        self.vocab_path: Path | None = vocab_path

        if vocab_path and vocab_path.exists():
            self.load(vocab_path)

    @classmethod
    def load_required(cls, vocab_path: Path) -> GeoSubwordTokenizer:
        """Fail-closed loader enforcing non-empty BPE merges and trained vocabulary > 256."""
        if not vocab_path.exists():
            raise FileNotFoundError(f"Required trained tokenizer missing at {vocab_path}")
        tokenizer = cls(vocab_path=vocab_path)
        if len(tokenizer.merges) == 0:
            raise ValueError(f"Tokenizer at {vocab_path} contains 0 merges (operating as byte-level!)")
        if len(tokenizer.id_to_token) <= 260:
            raise ValueError(f"Tokenizer vocab size ({len(tokenizer.id_to_token)}) is too small for trained BPE")
        return tokenizer

    def train_bpe_from_corpus(self, texts: list[str], target_vocab_size: int = 1024, min_frequency: int = 2) -> dict[str, str | int]:
        """Train BPE merges deterministically from corpus text documents."""
        self.target_vocab_size = target_vocab_size

        # Count word frequencies to speed up BPE training dramatically
        from collections import Counter
        word_counts = Counter()
        for text in texts:
            words = text.split()
            word_counts.update(words)

        # Convert words to initial token ID sequences paired with frequency count
        # Limit to top 20,000 most frequent words for fast and effective BPE merge selection
        word_seq_counts: list[tuple[list[int], int]] = []
        for word, count in word_counts.most_common(20000):
            word_bytes = (word + " ").encode("utf-8")
            seq = [self.token_to_id.get(bytes([b]), self.unk_token_id) for b in word_bytes]
            if seq:
                word_seq_counts.append((seq, count))

        while len(self.id_to_token) < self.target_vocab_size:
            # Count adjacent token pairs weighted by word frequency
            pair_counts: dict[tuple[int, int], int] = {}
            for seq, count in word_seq_counts:
                for p in zip(seq[:-1], seq[1:]):
                    pair_counts[p] = pair_counts.get(p, 0) + count

            if not pair_counts:
                break

            # Find most frequent pair; tie-break deterministically on lexicographical pair values
            best_pair = max(pair_counts.keys(), key=lambda p: (pair_counts[p], -p[0], -p[1]))
            max_freq = pair_counts[best_pair]

            if max_freq < min_frequency:
                break

            # Create new subword token
            new_id = len(self.id_to_token)
            p0_bytes = self.id_to_token[best_pair[0]]
            p1_bytes = self.id_to_token[best_pair[1]]
            new_bytes = p0_bytes + p1_bytes

            self.id_to_token[new_id] = new_bytes
            self.token_to_id[new_bytes] = new_id
            self.merges[best_pair] = new_id
            self.merge_frequencies[best_pair] = max_freq

            # Apply merge across word sequences
            new_word_seq_counts = []
            for seq, count in word_seq_counts:
                new_seq = []
                i = 0
                while i < len(seq):
                    if i < len(seq) - 1 and (seq[i], seq[i+1]) == best_pair:
                        new_seq.append(new_id)
                        i += 2
                    else:
                        new_seq.append(seq[i])
                        i += 1
                new_word_seq_counts.append((new_seq, count))
            word_seq_counts = new_word_seq_counts

        return self.get_tokenizer_telemetry()

    def get_tokenizer_telemetry(self) -> dict[str, str | int]:
        """Return structured tokenizer mode and verification metrics."""
        mode = "trained_bpe" if len(self.merges) > 0 else "byte_level"
        vocab_str = str(sorted(self.token_to_id.items()))
        sha256_digest = hashlib.sha256(vocab_str.encode("utf-8")).hexdigest()[:12]

        freqs = list(self.merge_frequencies.values())
        freq_2 = sum(1 for f in freqs if f == 2)
        freq_3_5 = sum(1 for f in freqs if 3 <= f <= 5)
        freq_gt_10 = sum(1 for f in freqs if f > 10)

        return {
            "tokenizer_mode": mode,
            "actual_vocab_size": len(self.id_to_token),
            "merge_count": len(self.merges),
            "target_vocab_size": self.target_vocab_size,
            "vocab_sha256": sha256_digest,
            "merges_freq_2": freq_2,
            "merges_freq_3_to_5": freq_3_5,
            "merges_freq_gt_10": freq_gt_10,
        }


    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode raw text string into token IDs using word-level caching."""
        if not hasattr(self, "_encode_cache") or self._encode_cache is None:
            self._encode_cache = {}

        # Split into words and whitespace chunks
        words = re.findall(r"\S+|\s+", text)
        token_ids = []

        for word in words:
            if word not in self._encode_cache:
                word_bytes = word.encode("utf-8")
                w_ids = [self.token_to_id.get(bytes([b]), self.unk_token_id) for b in word_bytes]

                while len(w_ids) >= 2:
                    stats = {}
                    for pair in zip(w_ids[:-1], w_ids[1:]):
                        if pair in self.merges:
                            stats[pair] = self.merges[pair]
                    if not stats:
                        break

                    best_pair = min(stats, key=lambda p: self.merges[p])
                    new_id = self.merges[best_pair]

                    new_ids = []
                    i = 0
                    while i < len(w_ids):
                        if i < len(w_ids) - 1 and (w_ids[i], w_ids[i+1]) == best_pair:
                            new_ids.append(new_id)
                            i += 2
                        else:
                            new_ids.append(w_ids[i])
                            i += 1
                    w_ids = new_ids
                self._encode_cache[word] = w_ids

            token_ids.extend(self._encode_cache[word])

        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]

        return token_ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back into raw text string."""
        byte_chunks = []
        for idx in token_ids:
            if skip_special_tokens and idx in (self.pad_token_id, self.bos_token_id, self.eos_token_id, self.unk_token_id):
                continue
            byte_chunks.append(self.id_to_token.get(idx, b""))

        raw_bytes = b"".join(byte_chunks)
        return raw_bytes.decode("utf-8", errors="replace")

    def save(self, vocab_path: Path):
        """Save vocabulary and BPE merges to JSON artifact."""
        data = {
            "target_vocab_size": self.target_vocab_size,
            "vocab": {idx: token.hex() for idx, token in self.id_to_token.items()},
            "merges": [f"{p0},{p1}->{new_id}" for (p0, p1), new_id in self.merges.items()]
        }
        content_bytes = json.dumps(data, indent=2).encode("utf-8")
        digest = hashlib.sha256(content_bytes).hexdigest()
        data["sha256_digest"] = digest

        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        vocab_path.write_bytes(json.dumps(data, indent=2).encode("utf-8"))

    def load(self, vocab_path: Path):
        """Load vocabulary and BPE merges from JSON artifact."""
        data = json.loads(vocab_path.read_text(encoding="utf-8"))
        self.target_vocab_size = data.get("target_vocab_size", 8192)
        self.id_to_token = {int(k): bytes.fromhex(v) for k, v in data["vocab"].items()}
        self.token_to_id = {v: k for k, v in self.id_to_token.items()}
        self.merges = {}
        for merge_str in data.get("merges", []):
            pair_str, new_id_str = merge_str.split("->")
            p0, p1 = map(int, pair_str.split(","))
            self.merges[(p0, p1)] = int(new_id_str)


def train_bpe_tokenizer(text: str, target_vocab_size: int = 1024, min_frequency: int = 2) -> GeoSubwordTokenizer:
    """Helper function to train a fresh BPE tokenizer from raw corpus text."""
    tokenizer = GeoSubwordTokenizer(target_vocab_size=target_vocab_size)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    tokenizer.train_bpe_from_corpus(lines, target_vocab_size=target_vocab_size, min_frequency=min_frequency)
    return tokenizer


