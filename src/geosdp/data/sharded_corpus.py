from __future__ import annotations

import json
from pathlib import Path
import random
import torch


class GeoShardedCorpusDataset:
    """Scalable Sharded Token Corpus Dataset for GEO Language Model Training."""

    def __init__(self, shards: list[torch.Tensor], seq_len: int = 512, seed: int = 42):
        self.seq_len = seq_len
        self.seed = seed

        # Combine shards into sequence windows
        all_tokens = torch.cat(shards, dim=0) if shards else torch.tensor([], dtype=torch.long)
        self.unique_token_count = len(torch.unique(all_tokens))
        self.total_tokens_available = len(all_tokens)

        # Slice into chunks of length seq_len + 1 (for input x and target y)
        num_samples = len(all_tokens) // (seq_len + 1)
        if num_samples > 0:
            usable_tokens = num_samples * (seq_len + 1)
            self.samples = all_tokens[:usable_tokens].view(num_samples, seq_len + 1)
        else:
            self.samples = torch.empty((0, seq_len + 1), dtype=torch.long)

        self.indices = list(range(len(self.samples)))
        self.rng = random.Random(seed)
        self.rng.shuffle(self.indices)

        self.cursor: int = 0
        self.total_tokens_consumed: int = 0

    @property
    def effective_epochs(self) -> float:
        """Calculate effective training epochs consumed."""
        if self.total_tokens_available == 0:
            return 0.0
        return self.total_tokens_consumed / self.total_tokens_available

    def get_batch(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch next microbatch [x, y] with cursor tracking."""
        if len(self.samples) == 0:
            raise ValueError("Corpus dataset is empty!")

        batch_indices = []
        for _ in range(batch_size):
            if self.cursor >= len(self.indices):
                # Reset epoch
                self.cursor = 0
                self.rng.shuffle(self.indices)

            batch_indices.append(self.indices[self.cursor])
            self.cursor += 1

        batch_samples = self.samples[batch_indices].to(device)
        x = batch_samples[:, :-1]
        y = batch_samples[:, 1:]

        tokens_in_batch = batch_size * self.seq_len
        self.total_tokens_consumed += tokens_in_batch

        return x, y

    def get_checkpoint_state(self) -> dict:
        """Serialize dataset cursor state for deterministic training resume."""
        return {
            "cursor": self.cursor,
            "total_tokens_consumed": self.total_tokens_consumed,
            "seed": self.seed,
            "indices": self.indices
        }

    def load_checkpoint_state(self, state: dict):
        """Restore dataset cursor state from checkpoint."""
        self.cursor = state.get("cursor", 0)
        self.total_tokens_consumed = state.get("total_tokens_consumed", 0)
        self.seed = state.get("seed", 42)
        if "indices" in state:
            self.indices = state["indices"]
