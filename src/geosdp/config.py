from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 32768
    d_model: int = 96
    n_layers: int = 6
    n_heads: int = 4
    ffn_hidden: int = 256
    seq_len: int = 512
    ple_dim: int = 128
    rope_theta: float = 10000.0
    dropout: float = 0.0

    def validate(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if (self.d_model // self.n_heads) % 2 != 0:
            raise ValueError("attention head dimension must be even for RoPE")
        if min(self.vocab_size, self.d_model, self.n_layers, self.n_heads, self.seq_len) <= 0:
            raise ValueError("all model dimensions must be positive")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads
