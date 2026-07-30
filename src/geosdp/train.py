from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

import time

from .backends import create_backend
from .config import ModelConfig
from .model import GeoDomainLM
from .telemetry import TelemetryLogger


def load_config(path: Path) -> tuple[ModelConfig, dict]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return ModelConfig(**raw["model"]), raw["training"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--backend", choices=("native", "reference"), default="native")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--telemetry-path", type=Path, default=None)
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()

    cfg, train_cfg = load_config(args.config)
    backend = create_backend(args.backend)
    if not args.smoke and not backend.native:
        raise RuntimeError("production training requires --backend native")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if backend.native and device.type != "cuda":
        raise RuntimeError("native GEO training requires CUDA")

    telemetry = TelemetryLogger(log_path=args.telemetry_path)

    model = GeoDomainLM(cfg, backend).to(device)
    learning_rate = float(train_cfg["learning_rate"])
    grad_clip = float(train_cfg["grad_clip"])
    weight_decay = float(train_cfg.get("weight_decay", 0.01))

    if backend.native:
        import geo_dl_runtime

        geo_dl_runtime.require_stage("training")
        optimizer = geo_dl_runtime.GeoAdamW(
            model.parameters(),
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=grad_clip,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    steps = 2 if args.smoke else int(train_cfg["steps"])
    batch_size = int(train_cfg["batch_size"])
    tokens_seen = 0

    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        idx = torch.randint(
            0,
            cfg.vocab_size,
            (batch_size, cfg.seq_len),
            device=device,
        )
        targets = torch.roll(idx, shifts=-1, dims=1)
        tokens_seen += batch_size * cfg.seq_len

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(idx, targets)
        assert loss is not None
        loss.backward()
        if backend.native:
            optimizer.step()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        t1 = time.perf_counter()
        step_time_ms = (t1 - t0) * 1000.0

        if step % args.log_every == 0 or step == steps:
            telemetry.log_step(
                step=step,
                loss=loss.item(),
                lr=learning_rate,
                tokens_seen=tokens_seen,
                step_time_ms=step_time_ms,
                model=model,
                device=device,
                grad_clip=grad_clip,
            )
            print(
                f"step={step} loss={loss.item():.6f} "
                f"backend={backend.name} device={device}"
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_alloc_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        peak_res_mb = torch.cuda.max_memory_reserved(device) / (1024 * 1024)
        print(
            f"peak_gpu_memory_allocated={peak_alloc_mb:.2f}MB "
            f"peak_gpu_memory_reserved={peak_res_mb:.2f}MB"
        )


if __name__ == "__main__":
    main()
