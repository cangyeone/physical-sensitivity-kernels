#!/usr/bin/env python3
"""Run IND-FWD control-point inversion and uncertainty diagnostics."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np

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
    p.add_argument("--forward-ckpt", type=Path, default=ROOT / "ckpt" / "struct2disp_cpmlp.prior_boundary_v3.pt")
    p.add_argument("--train-forward-if-missing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--n-test", type=int, default=128)
    p.add_argument("--n-envelope", type=int, default=512)
    p.add_argument("--n-forward-train", type=int, default=4096)
    p.add_argument("--forward-epochs", type=int, default=20)
    p.add_argument("--n-forward-eval", type=int, default=50)
    p.add_argument("--forward-inv-steps", type=int, default=220)
    p.add_argument("--forward-inv-lr", type=float, default=0.04)
    p.add_argument("--indirect-uncertainty-samples", type=int, default=8)
    p.add_argument("--indirect-multistarts", type=int, default=6)
    p.add_argument("--indirect-uncertainty-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ev = load_eval_module()
    ev.set_seed(args.seed)
    device = ev.choose_device(args.device)
    strong_mod = ev.import_from_path("indirect_generate_data", ROOT / "utils" / "generate_data.py")
    envelope = ev.prior_envelope(strong_mod, args.n_envelope, args.seed + 10)

    periods = np.linspace(2.0, 60.0, 59).astype(np.float32)
    in_models_full, in_disp, in_mask = ev.dataset_to_arrays(ev.strong_dataset(strong_mod, args.n_test, args.seed + 20))
    in_models = in_models_full[:, 1:4, :]
    test_sets = {
        "in-prior": (in_models, in_disp, in_mask),
        "boundary": ev.parametric_dataset(strong_mod, "boundary", args.n_test, args.seed + 30, in_models.shape[-1], periods),
        "out-of-prior": ev.parametric_dataset(strong_mod, "out-of-prior", args.n_test, args.seed + 40, in_models.shape[-1], periods),
    }

    fwd_model = ev.load_forward_model(args.forward_ckpt, device)
    if fwd_model is None and args.train_forward_if_missing:
        fwd_model = ev.train_tiny_forward_surrogate(
            strong_mod=strong_mod,
            ckpt_path=args.forward_ckpt,
            device=device,
            n_train=args.n_forward_train,
            epochs=args.forward_epochs,
            seed=args.seed + 50,
            batch_size=8,
        )
    if fwd_model is None:
        raise FileNotFoundError(f"Forward surrogate not found: {args.forward_ckpt}")

    rows, diag = ev.evaluate_forward_iterative(fwd_model, test_sets, strong_mod, envelope, device, args)
    ev.write_csv(ev.RESULTS_DIR / "prior_boundary_forward_iterative.csv", rows)
    ev.plot_forward_examples(ev.FIGURES_DIR / "indirect_forward_inversion_examples.png", diag)

    u_rows, u_diag = ev.indirect_uncertainty_diagnostic(fwd_model, test_sets, strong_mod, envelope, device, args)
    ev.write_csv(ev.RESULTS_DIR / "prior_boundary_indirect_uncertainty.csv", u_rows)
    ev.plot_indirect_uncertainty(ev.FIGURES_DIR / "indirect_uncertainty_diagnostics.png", u_diag)
    print(f"[done] wrote {ev.RESULTS_DIR / 'prior_boundary_forward_iterative.csv'}")


if __name__ == "__main__":
    main()
