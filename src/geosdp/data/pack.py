from __future__ import annotations

from typing import List, Tuple

from geosdp.tokenizer.base import Tokenizer


def pack_token_sequences(
    documents: List[str],
    tokenizer: Tokenizer,
    seq_len: int,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Packs documents into fixed-length sequence tensors with <BOS>/<EOS> boundaries."""
    all_tokens: List[int] = []

    for doc in documents:
        tokens = tokenizer.encode(doc, add_bos=True, add_eos=True)
        all_tokens.extend(tokens)

    inputs: List[List[int]] = []
    targets: List[List[int]] = []

    stride = seq_len
    for i in range(0, len(all_tokens) - seq_len, stride):
        chunk = all_tokens[i : i + seq_len + 1]
        if len(chunk) == seq_len + 1:
            inputs.append(chunk[:seq_len])
            targets.append(chunk[1 : seq_len + 1])

    return inputs, targets
