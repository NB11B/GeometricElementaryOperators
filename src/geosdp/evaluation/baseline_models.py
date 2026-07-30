from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import torch

from geosdp.tokenizer.bpe_tokenizer import GeoSubwordTokenizer


def resolve_special_token(tokenizer: GeoSubwordTokenizer, string_key: str, byte_key: bytes) -> tuple[int, str]:
    """Fail-closed special token resolution raising explicit KeyError if missing."""
    if string_key in tokenizer.token_to_id:
        return int(tokenizer.token_to_id[string_key]), "str"
    if byte_key in tokenizer.token_to_id:
        return int(tokenizer.token_to_id[byte_key]), "bytes"
    raise KeyError(f"Required special token missing: {string_key!r} / {byte_key!r}")


class UniformBaseline:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def evaluate_nll(self, tokens: torch.Tensor) -> float:
        return math.log(self.vocab_size)


class UnigramBaseline:
    def __init__(self, vocab_size: int, alpha: float = 1.0):
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.probs = torch.full((vocab_size,), alpha, dtype=torch.float64)

    def fit(self, train_tokens: torch.Tensor):
        counts = torch.bincount(train_tokens, minlength=self.vocab_size).to(torch.float64)
        total = counts.sum() + (self.alpha * self.vocab_size)
        self.probs = (counts + self.alpha) / total

    def evaluate_nll(self, tokens: torch.Tensor) -> float:
        token_probs = self.probs[tokens]
        log_probs = torch.log(token_probs)
        return -float(log_probs.mean().item())


class BigramBaseline:
    def __init__(self, vocab_size: int, alpha: float = 1.0, special_token_mode: str = "required"):
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.special_token_mode = special_token_mode
        self.transition_counts = torch.zeros((vocab_size, vocab_size), dtype=torch.float64)
        self.context_counts = torch.zeros(vocab_size, dtype=torch.float64)
        self.probs = torch.full((vocab_size, vocab_size), 1.0 / vocab_size, dtype=torch.float64)
        self.observed_bigram_types = 0

    def fit_from_documents(self, tokenizer: GeoSubwordTokenizer, doc_lines: list[str]) -> dict:
        """Fit document-boundary-preserving bigram model with explicit mode control."""
        special_telemetry = {"special_token_mode": self.special_token_mode}

        if self.special_token_mode == "required":
            bos_id, bos_type = resolve_special_token(tokenizer, "<bos>", b"<bos>")
            eos_id, eos_type = resolve_special_token(tokenizer, "<eos>", b"<eos>")
            special_telemetry.update({
                "bos_present": True,
                "eos_present": True,
                "bos_id": bos_id,
                "eos_id": eos_id,
                "bos_key_type": bos_type,
                "eos_key_type": eos_type,
            })
            use_special = True
        elif self.special_token_mode == "none":
            special_telemetry.update({
                "bos_present": False,
                "eos_present": False,
                "bos_id": None,
                "eos_id": None,
                "bos_key_type": None,
                "eos_key_type": None,
            })
            use_special = False
        else:
            raise ValueError(f"Invalid special_token_mode: {self.special_token_mode!r}. Must be 'required' or 'none'.")

        for line in doc_lines:
            if not line.strip():
                continue
            toks = tokenizer.encode(line)
            if not toks:
                continue
            doc_toks = [bos_id] + toks + [eos_id] if use_special else toks

            srcs = doc_toks[:-1]
            tgts = doc_toks[1:]

            for s, t in zip(srcs, tgts):
                self.transition_counts[s, t] += 1.0
                self.context_counts[s] += 1.0

        self.observed_bigram_types = int((self.transition_counts > 0).sum().item())
        denom = self.context_counts.unsqueeze(1) + (self.alpha * self.vocab_size)
        self.probs = (self.transition_counts + self.alpha) / denom
        return special_telemetry

    def evaluate_nll_from_documents(self, tokenizer: GeoSubwordTokenizer, doc_lines: list[str]) -> tuple[float, dict]:
        """Evaluates exact token-weighted NLL without crossing document boundaries."""
        if self.special_token_mode == "required":
            bos_id, _ = resolve_special_token(tokenizer, "<bos>", b"<bos>")
            eos_id, _ = resolve_special_token(tokenizer, "<eos>", b"<eos>")
            use_special = True
        elif self.special_token_mode == "none":
            use_special = False
        else:
            raise ValueError(f"Invalid special_token_mode: {self.special_token_mode!r}")

        total_nll = 0.0
        total_tokens = 0
        unseen_context_count = 0
        unseen_bigram_count = 0

        for line in doc_lines:
            if not line.strip():
                continue
            toks = tokenizer.encode(line)
            if not toks:
                continue
            doc_toks = [bos_id] + toks + [eos_id] if use_special else toks

            srcs = doc_toks[:-1]
            tgts = doc_toks[1:]

            for s, t in zip(srcs, tgts):
                if self.context_counts[s] == 0:
                    unseen_context_count += 1
                if self.transition_counts[s, t] == 0:
                    unseen_bigram_count += 1

                p = self.probs[s, t].item()
                total_nll += -math.log(p)
                total_tokens += 1

        mean_nll = (total_nll / float(total_tokens)) if total_tokens > 0 else float("inf")
        telemetry = {
            "evaluated_tokens": total_tokens,
            "unseen_context_rate": (unseen_context_count / float(total_tokens)) if total_tokens > 0 else 0.0,
            "unseen_bigram_rate": (unseen_bigram_count / float(total_tokens)) if total_tokens > 0 else 0.0,
        }
        return mean_nll, telemetry


def run_baseline_models_evaluation(
    corpus_dir: Path | None = None,
    out_dir: Path = Path("artifacts/corpus_audit"),
    special_token_mode: str = "required",
) -> dict:
    """Evaluates Uniform, Unigram, and Bigram baselines with explicit mode control."""
    if corpus_dir is None:
        c1m = Path("artifacts/domain_corpus_1m")
        corpus_dir = c1m if c1m.exists() else Path("artifacts/scaled_corpus")

    out_dir.mkdir(parents=True, exist_ok=True)


    tokenizer_path = Path("artifacts/tokenizer_bpe_4k.json")
    assert tokenizer_path.exists(), "Missing tokenizer_bpe_4k.json!"

    tokenizer = GeoSubwordTokenizer.load_required(tokenizer_path)
    actual_vocab_size = len(tokenizer.id_to_token)
    assert actual_vocab_size == 1286, f"Expected actual vocab size 1,286, found {actual_vocab_size}"

    train_manifest_path = corpus_dir / "train_manifest.txt"
    domain_val_manifest_path = corpus_dir / "domain_val_manifest.txt"
    general_val_manifest_path = corpus_dir / "general_val_manifest.txt"
    sealed_test_manifest_path = corpus_dir / "sealed_test_manifest.txt"

    train_tokens = torch.load(corpus_dir / "train_tokens.pt")
    train_manifest_text = train_manifest_path.read_text(encoding="utf-8")
    reencoded_train = torch.tensor(tokenizer.encode(train_manifest_text), dtype=torch.long)
    assert torch.equal(train_tokens, reencoded_train), "train_tokens.pt does NOT match re-encoded train_manifest.txt!"

    doc_splits = {
        "train": train_manifest_path.read_text(encoding="utf-8").splitlines(),
        "domain_val": domain_val_manifest_path.read_text(encoding="utf-8").splitlines(),
        "general_val": general_val_manifest_path.read_text(encoding="utf-8").splitlines(),
    }
    if sealed_test_manifest_path.exists():
        doc_splits["sealed_test"] = sealed_test_manifest_path.read_text(encoding="utf-8").splitlines()

    token_splits = {
        name: torch.tensor(tokenizer.encode("\n".join(lines)), dtype=torch.long)
        for name, lines in doc_splits.items()
    }


    uniform_model = UniformBaseline(vocab_size=actual_vocab_size)
    unigram_model_1 = UnigramBaseline(vocab_size=actual_vocab_size, alpha=1.0)
    unigram_model_01 = UnigramBaseline(vocab_size=actual_vocab_size, alpha=0.1)
    bigram_model = BigramBaseline(vocab_size=actual_vocab_size, alpha=1.0, special_token_mode=special_token_mode)

    print("Fitting Unigram and Document-Boundary Bigram frequency baselines...")
    unigram_model_1.fit(train_tokens)
    unigram_model_01.fit(train_tokens)
    special_telemetry = bigram_model.fit_from_documents(tokenizer, doc_splits["train"])

    tc_bytes = bigram_model.transition_counts.element_size() * bigram_model.transition_counts.nelement()
    p_bytes = bigram_model.probs.element_size() * bigram_model.probs.nelement()
    total_tensor_bytes_mb = (tc_bytes + p_bytes) / (1024.0 * 1024.0)

    report = {
        "actual_vocab_size": actual_vocab_size,
        "special_token_audit": special_telemetry,
        "bigram_telemetry": {
            "observed_bigram_types": bigram_model.observed_bigram_types,
            "matrix_density_pct": (bigram_model.observed_bigram_types / float(actual_vocab_size * actual_vocab_size)) * 100.0,
            "actual_tensor_bytes_mb": total_tensor_bytes_mb,
        },
        "baselines": {}
    }

    for name in doc_splits.keys():
        tok_tensor = token_splits[name]
        lines = doc_splits[name]


        u_nll = uniform_model.evaluate_nll(tok_tensor)
        uni_nll_1 = unigram_model_1.evaluate_nll(tok_tensor)
        uni_nll_01 = unigram_model_01.evaluate_nll(tok_tensor)
        bi_nll, bi_telem = bigram_model.evaluate_nll_from_documents(tokenizer, lines)

        report["baselines"][name] = {
            "uniform_nll": u_nll,
            "uniform_ppl": math.exp(u_nll),
            "unigram_nll_alpha_1": uni_nll_1,
            "unigram_ppl_alpha_1": math.exp(uni_nll_1),
            "unigram_nll_alpha_01": uni_nll_01,
            "unigram_ppl_alpha_01": math.exp(uni_nll_01),
            "bigram_nll": bi_nll,
            "bigram_ppl": math.exp(bi_nll),
            "bigram_telemetry": bi_telem,
        }

    with open(out_dir / "baseline_models_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    run_baseline_models_evaluation()
