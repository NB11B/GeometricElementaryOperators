from __future__ import annotations

import hashlib
import math
from pathlib import Path
import torch
import torch.nn as nn


def evaluate_token_weighted_nll(
    model: nn.Module,
    tokens: torch.Tensor,
    batch_size: int = 2,
    seq_len: int = 512,
    device: torch.device = torch.device("cuda"),
    pad_token_id: int = 0,
) -> dict:
    """Evaluates exact token-weighted Negative Log-Likelihood (NLL = ln(PPL)) with padding and coverage accounting."""
    model.eval()
    total_input_tokens = tokens.numel()

    if total_input_tokens < 2:
        return {
            "mean_nll": float("inf"),
            "display_ppl": float("inf"),
            "total_input_tokens": total_input_tokens,
            "evaluated_tokens": 0,
            "discarded_tail_tokens": total_input_tokens,
            "token_coverage_pct": 0.0,
        }

    # Dynamic adjustment if token stream is shorter than requested batch chunk
    effective_batch_size = batch_size
    if total_input_tokens < (batch_size * seq_len + 1):
        effective_batch_size = 1

    effective_seq_len = min(seq_len, total_input_tokens - 1)

    # Pad tokens if shorter than effective_seq_len + 1
    if total_input_tokens < effective_seq_len + 1:
        pad_len = (effective_seq_len + 1) - total_input_tokens
        padding = torch.full((pad_len,), pad_token_id, dtype=tokens.dtype, device=tokens.device)
        eval_tokens = torch.cat([tokens, padding], dim=0)
    else:
        eval_tokens = tokens

    total_nll_sum = 0.0
    total_valid_tokens = 0
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    chunk_size = effective_batch_size * effective_seq_len
    num_eval_tokens = eval_tokens.numel()

    with torch.no_grad():
        for i in range(0, num_eval_tokens - effective_seq_len, chunk_size):
            chunk = eval_tokens[i : i + chunk_size + 1]
            if chunk.numel() < chunk_size + 1:
                # Pad remaining tail chunk
                pad_needed = (chunk_size + 1) - chunk.numel()
                chunk = torch.cat([chunk, torch.full((pad_needed,), pad_token_id, dtype=tokens.dtype, device=tokens.device)], dim=0)

            x = chunk[:-1].reshape(effective_batch_size, effective_seq_len).to(device)
            y = chunk[1:].reshape(effective_batch_size, effective_seq_len).to(device)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x, y)
                losses = loss_fn(logits.view(-1, logits.size(-1)).float(), y.view(-1)).reshape(effective_batch_size, effective_seq_len)

            # Mask out padding target tokens
            valid_mask = (y != pad_token_id)
            valid_count = int(valid_mask.sum().item())

            if valid_count > 0:
                total_nll_sum += float(losses[valid_mask].sum().item())
                total_valid_tokens += valid_count

    model.train()

    if total_valid_tokens == 0:
        return {
            "mean_nll": float("inf"),
            "display_ppl": float("inf"),
            "total_input_tokens": total_input_tokens,
            "evaluated_tokens": 0,
            "discarded_tail_tokens": total_input_tokens,
            "token_coverage_pct": 0.0,
        }

    mean_nll = total_nll_sum / float(total_valid_tokens)
    display_ppl = math.exp(min(mean_nll, 20.0))

    return {
        "mean_nll": mean_nll,
        "display_ppl": display_ppl,
        "total_input_tokens": total_input_tokens,
        "evaluated_tokens": total_valid_tokens,
        "discarded_tail_tokens": max(0, total_input_tokens - total_valid_tokens),
        "token_coverage_pct": (total_valid_tokens / float(total_input_tokens) * 100.0),
    }


def inspect_manifest_provenance(manifest_path: Path) -> dict:
    """Inspects manifest text file and computes raw character count, line count, and SHA256 digest."""
    assert manifest_path.exists(), f"Manifest missing at {manifest_path}"
    content = manifest_path.read_text(encoding="utf-8")
    lines = content.splitlines()

    return {
        "filename": manifest_path.name,
        "char_count": len(content),
        "line_count": len(lines),
        "excerpt_50_chars": content[:50].strip(),
        "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }
