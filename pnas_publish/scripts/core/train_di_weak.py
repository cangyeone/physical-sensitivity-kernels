#!/usr/bin/env python3
"""Train the weak-prior direct inversion / CRF control model."""

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


def import_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-train", type=int, default=20_000)
    p.add_argument("--n-val", type=int, default=1_024)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=3026)
    p.add_argument("--device", default="auto")
    p.add_argument("--save-dir", type=Path, default=ROOT / "ckpt" / "disp2struct_crf.v1.2_cp_weak")
    p.add_argument("--fig-dir", type=Path, default=ROOT / "tfig_inv_v1.2_cp_weak")
    p.add_argument("--init-checkpoint", type=Path, default=ROOT / "ckpt" / "disp2struct_crf.v1.2_cp" / "best.pt")
    p.add_argument("--no-warm-start", action="store_true")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--stats-batches", type=int, default=32)
    p.add_argument("--log-every-steps", type=int, default=100)
    p.add_argument("--plot-every-steps", type=int, default=0)
    p.add_argument("--eval-num-samples", type=int, default=2)
    p.add_argument("--eval-num-steps", type=int, default=12)
    return p.parse_args()


def choose_device(direct_mod, requested: str) -> str:
    if requested != "auto":
        return requested
    return direct_mod.default_device()


def main() -> None:
    args = parse_args()
    direct_mod = import_from_path("disp_inv_train_v12_weak", ROOT / "disp_inv_train.v1.2.py")
    from utils import generate_data_weak_prior as weak_mod

    cfg = direct_mod.TrainConfig(
        n_train=args.n_train,
        n_val=args.n_val,
        z_max_km=150.0,
        z_max_num=256,
        dz_km=0.5,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        stats_batches=args.stats_batches,
        save_dir=str(args.save_dir),
        fig_dir=str(args.fig_dir),
        resume=args.resume,
        init_checkpoint=None if args.no_warm_start else str(args.init_checkpoint),
        epochs=args.epochs,
        lr=1.0e-4 if not args.no_warm_start else 2.0e-4,
        min_lr=1.0e-6,
        val_every=1,
        eval_num_samples=args.eval_num_samples,
        eval_num_steps=args.eval_num_steps,
        log_every_steps=args.log_every_steps,
        plot_every_steps=args.plot_every_steps,
        device=choose_device(direct_mod, args.device),
    )

    train_ds = weak_mod.SurfaceWaveDataset(
        n_samples=cfg.n_train,
        z_max_km=cfg.z_max_km,
        z_max_num=cfg.z_max_num,
        dz_km=cfg.dz_km,
        seed=cfg.seed,
    )
    val_ds = weak_mod.SurfaceWaveDataset(
        n_samples=cfg.n_val,
        z_max_km=cfg.z_max_km,
        z_max_num=cfg.z_max_num,
        dz_km=cfg.dz_km,
        seed=cfg.seed + 1_000_000,
    )
    device = direct_mod.torch.device(cfg.device)
    train_loader = direct_mod.make_loader(train_ds, cfg=cfg, shuffle=True, device=device)
    val_loader = direct_mod.make_loader(val_ds, cfg=cfg, shuffle=False, device=device)
    direct_mod.train_disp2struct_crf(train_loader, val_loader=val_loader, cfg=cfg)
    print(f"[done] DI-Weak checkpoint directory: {cfg.save_dir}")


if __name__ == "__main__":
    main()
