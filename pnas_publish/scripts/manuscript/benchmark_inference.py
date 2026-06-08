#!/usr/bin/env python3
"""Benchmark posterior sampling throughput for the analyzed checkpoint."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

SCRIPTS = Path(__file__).resolve().parent
OVERLEAF = SCRIPTS.parent
ROOT = OVERLEAF.parent
OUT = OVERLEAF / "figures"
OUT.mkdir(parents=True, exist_ok=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from make_paper_figures import collect_dataset, load_module, restore_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark neural posterior sampling throughput.")
    parser.add_argument("--n-examples", type=int, default=128, help="Held-out dispersion curves to sample.")
    parser.add_argument("--posterior-samples", type=int, default=16, help="Posterior samples per dispersion curve.")
    parser.add_argument("--sampling-steps", type=int, default=24, help="Euler steps for rectified-flow sampling.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for sampling.")
    parser.add_argument("--repeats", type=int, default=5, help="Timed repeats after warm-up.")
    parser.add_argument("--warmup", type=int, default=1, help="Warm-up passes over one batch.")
    parser.add_argument("--threads", type=int, default=0, help="Torch CPU threads; 0 keeps the runtime default.")
    parser.add_argument("--output", type=Path, default=OUT / "inference_benchmark.json", help="Output JSON path.")
    return parser.parse_args()


def sample_all(model, disp_batch, mask_batch, args) -> float:
    total = 0.0
    with torch.no_grad():
        for i in range(0, disp_batch.size(0), args.batch_size):
            out = model.sample(
                disp_batch[i : i + args.batch_size],
                mask_batch[i : i + args.batch_size],
                num_samples=args.posterior_samples,
                num_steps=args.sampling_steps,
            )
            total += float(out["profile_samples"].mean())
    return total


def main() -> None:
    args = parse_args()
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(2026)
    np.random.seed(2026)

    mod12 = load_module("disp_inv_train_v12", ROOT / "disp_inv_train.v1.2.py")
    model, ckpt = restore_model(mod12, ROOT / "ckpt/disp2struct_crf.v1.2_cp/best.pt")
    _, disp_batch, mask_batch = collect_dataset(mod12, n=args.n_examples)
    disp_batch = disp_batch.float()
    mask_batch = mask_batch.float()

    warmup_disp = disp_batch[: args.batch_size]
    warmup_mask = mask_batch[: args.batch_size]
    warmup_args = argparse.Namespace(**{**vars(args), "n_examples": warmup_disp.size(0)})
    for _ in range(args.warmup):
        sample_all(model, warmup_disp, warmup_mask, warmup_args)

    elapsed = []
    checksums = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        checksums.append(sample_all(model, disp_batch, mask_batch, args))
        elapsed.append(time.perf_counter() - start)

    median_seconds = statistics.median(elapsed)
    mean_seconds = statistics.mean(elapsed)
    std_seconds = statistics.pstdev(elapsed) if len(elapsed) > 1 else 0.0
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_epoch": int(ckpt["epoch"]),
        "checkpoint_global_step": int(ckpt["global_step"]),
        "device": "cpu",
        "n_examples": int(args.n_examples),
        "posterior_samples": int(args.posterior_samples),
        "sampling_steps": int(args.sampling_steps),
        "batch_size": int(args.batch_size),
        "repeats": int(args.repeats),
        "warmup": int(args.warmup),
        "torch_threads": int(torch.get_num_threads()),
        "torch_interop_threads": int(torch.get_num_interop_threads()),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "elapsed_seconds": [float(x) for x in elapsed],
        "median_seconds": float(median_seconds),
        "mean_seconds": float(mean_seconds),
        "std_seconds": float(std_seconds),
        "curves_per_second": float(args.n_examples / median_seconds),
        "posterior_profiles_per_second": float(args.n_examples * args.posterior_samples / median_seconds),
        "seconds_per_curve_for_all_samples": float(median_seconds / args.n_examples),
        "seconds_per_posterior_profile": float(median_seconds / (args.n_examples * args.posterior_samples)),
        "checksum": float(sum(checksums)),
    }

    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
