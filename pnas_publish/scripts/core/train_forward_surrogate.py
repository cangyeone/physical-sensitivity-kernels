#!/usr/bin/env python3
"""Train a small Struct2Disp forward surrogate for the prior-boundary test."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_eval_module():
    path = ROOT / "scripts" / "eval_prior_boundary_effect.py"
    spec = importlib.util.spec_from_file_location("prior_boundary_eval", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["prior_boundary_eval"] = module
    spec.loader.exec_module(module)
    return module


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "ckpt" / "struct2disp_cpmlp.prior_boundary_v3.pt")
    p.add_argument("--n-train", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ev = load_eval_module()
    ev.set_seed(args.seed)
    device = ev.choose_device(args.device)
    strong_mod = ev.import_from_path("train_forward_generate_data", ROOT / "utils" / "generate_data.py")
    ev.train_tiny_forward_surrogate(
        strong_mod=strong_mod,
        ckpt_path=args.output,
        device=device,
        n_train=args.n_train,
        epochs=args.epochs,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    print(f"[done] wrote {args.output}")


if __name__ == "__main__":
    main()
