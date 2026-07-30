from __future__ import annotations

import random
from pathlib import Path
import torch


class DeterministicDocumentBatchSampler:
    """Generates every valid document window with stride=seq_len, padding short windows with -100 targets."""

    def __init__(
        self,
        documents: list[torch.Tensor],
        batch_size: int = 2,
        seq_len: int = 512,
        stride: int | None = None,
        seed: int = 42,
        mode: str = "shuffled_documents",
        source_path: Path | None = None,
        pad_token_id: int = 0,
        ignore_index: int = -100,
    ):
        if source_path is not None:
            assert "sealed_test" not in source_path.name.lower(), "SEALED TEST ACCESS PROHIBITED IN CALIBRATION SAMPLER!"

        self.documents = [doc for doc in documents if doc.numel() > 0]
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.stride = stride if stride is not None else seq_len
        self.seed = seed
        self.mode = mode
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

        self.windows = []  # List of dicts: {"doc_id": int, "start": int, "end": int}
        self._build_window_manifest()

        self.num_windows = len(self.windows)
        self.window_indices = list(range(self.num_windows))
        self.reset_epoch(seed)

    def _build_window_manifest(self):
        """Constructs windows covering 100% of every document across all token positions."""
        for doc_id, doc_toks in enumerate(self.documents):
            n_tokens = doc_toks.numel()
            if n_tokens <= 1:
                # 1 token document: single window
                self.windows.append({"doc_id": doc_id, "start": 0, "end": n_tokens})
                continue

            start = 0
            while start < n_tokens - 1:
                end = min(start + self.seq_len + 1, n_tokens)
                self.windows.append({"doc_id": doc_id, "start": start, "end": end})
                start += self.stride

    def reset_epoch(self, epoch_seed: int):
        """Idempotent epoch reset re-initializing from canonical sorted indices before shuffling."""
        self.window_indices = list(range(self.num_windows))
        if self.mode == "shuffled_documents":
            rng = random.Random(epoch_seed)
            rng.shuffle(self.window_indices)

    def get_batch(self, batch_idx: int, device: torch.device = torch.device("cuda")) -> dict:
        """Returns batch dict containing x, y (with -100 padded targets), masks, and coverage telemetry."""
        batch_x = []
        batch_y = []
        batch_masks = []
        doc_ids = []
        window_offsets = []

        total_valid_targets = 0
        total_padded_targets = 0

        for i in range(self.batch_size):
            if self.num_windows == 0:
                # Fallback for empty corpus
                x_seq = torch.full((self.seq_len,), self.pad_token_id, dtype=torch.long)
                y_seq = torch.full((self.seq_len,), self.ignore_index, dtype=torch.long)
                mask_seq = torch.zeros((self.seq_len,), dtype=torch.bool)
                w_info = {"doc_id": -1, "start": 0, "end": 0}
            else:
                w_idx = self.window_indices[(batch_idx * self.batch_size + i) % self.num_windows]
                w_info = self.windows[w_idx]
                doc_toks = self.documents[w_info["doc_id"]]
                sub_toks = doc_toks[w_info["start"] : w_info["end"]]

                req_len = self.seq_len + 1
                if sub_toks.numel() >= req_len:
                    chunk = sub_toks[:req_len]
                    x_raw = chunk[:-1]
                    y_raw = chunk[1:]
                    mask_seq = torch.ones((self.seq_len,), dtype=torch.bool)
                    x_seq = x_raw
                    y_seq = y_raw
                    valid_cnt = self.seq_len
                    pad_cnt = 0
                else:
                    valid_pair_len = max(0, sub_toks.numel() - 1)
                    x_valid = sub_toks[:-1] if valid_pair_len > 0 else torch.empty((0,), dtype=torch.long)
                    y_valid = sub_toks[1:] if valid_pair_len > 0 else torch.empty((0,), dtype=torch.long)

                    pad_len = self.seq_len - valid_pair_len
                    x_pad = torch.full((pad_len,), self.pad_token_id, dtype=torch.long)
                    y_pad = torch.full((pad_len,), self.ignore_index, dtype=torch.long)
                    mask_pad = torch.zeros((pad_len,), dtype=torch.bool)
                    mask_valid = torch.ones((valid_pair_len,), dtype=torch.bool)

                    x_seq = torch.cat([x_valid, x_pad], dim=0)
                    y_seq = torch.cat([y_valid, y_pad], dim=0)
                    mask_seq = torch.cat([mask_valid, mask_pad], dim=0)

                    valid_cnt = valid_pair_len
                    pad_cnt = pad_len

                total_valid_targets += valid_cnt
                total_padded_targets += pad_cnt

            batch_x.append(x_seq)
            batch_y.append(y_seq)
            batch_masks.append(mask_seq)
            doc_ids.append(w_info["doc_id"])
            window_offsets.append((w_info["start"], w_info["end"]))

        x = torch.stack(batch_x, dim=0).to(device)
        y = torch.stack(batch_y, dim=0).to(device)
        valid_target_mask = torch.stack(batch_masks, dim=0).to(device)

        return {
            "x": x,
            "y": y,
            "valid_target_mask": valid_target_mask,
            "document_ids": doc_ids,
            "window_offsets": window_offsets,
            "valid_target_count": total_valid_targets,
            "padded_target_count": total_padded_targets,
        }
