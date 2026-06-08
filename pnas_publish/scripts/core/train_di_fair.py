#!/usr/bin/env python3
"""Train a matched DI-Strong or DI-Weak model from a YAML config.

This entry point is for fair prior-comparison experiments.  It intentionally
does not warm-start either model; the only intended difference between paired
configs should be the structural prior generator.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import yaml

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
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--device", default=None, help="Override config device, e.g. cpu, mps, cuda.")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=None)
    return p.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return cfg


def comparable_config(cfg: dict) -> dict:
    """Return fields that must match between paired fair configs."""
    keep = {}
    for section in ("train", "model", "optimizer", "loss", "mask_augmentation", "sampling"):
        keep[section] = cfg.get(section, {})
    keep["experiment_id"] = cfg.get("experiment_id")
    keep["role"] = cfg.get("role")
    return keep


def assert_paired_config_matches(config_path: Path, cfg: dict) -> None:
    paired = cfg.get("paired_config")
    if not paired:
        return
    paired_path = (ROOT / paired).resolve()
    if not paired_path.exists():
        raise FileNotFoundError(f"Paired fair config does not exist: {paired_path}")
    paired_cfg = load_config(paired_path)
    lhs = comparable_config(cfg)
    rhs = comparable_config(paired_cfg)
    if lhs != rhs:
        raise ValueError(
            "Fair paired configs differ in matched-budget fields. "
            f"Current={config_path}, paired={paired_path}"
        )
    if cfg.get("prior") == paired_cfg.get("prior"):
        raise ValueError("Paired fair configs must use different prior values.")


def parse_period_windows(direct_mod, cfg: dict):
    windows = cfg.get("windows", cfg.get("presets", [[2.0, 40.0], [10.0, 30.0], [5.0, 35.0], [15.0, 45.0]]))
    if isinstance(windows, str):
        return direct_mod.parse_period_windows(windows)
    return tuple((float(lo), float(hi)) for lo, hi in windows)


def choose_device(direct_mod, requested: str | None) -> str:
    if requested and requested != "auto":
        return requested
    return direct_mod.default_device()


def main() -> None:
    args = parse_args()
    cfg_yaml = load_config(args.config)
    assert_paired_config_matches(args.config, cfg_yaml)
    prior = cfg_yaml.get("prior")
    if prior not in {"strong", "weak"}:
        raise ValueError("Config field 'prior' must be either 'strong' or 'weak'.")

    direct_mod = import_from_path("disp_inv_train_v13_fair", ROOT / "disp_inv_train.v1.3.py")
    dataset_mod = (
        __import__("utils.generate_data", fromlist=["SurfaceWaveDataset"])
        if prior == "strong"
        else __import__("utils.generate_data_weak_prior", fromlist=["SurfaceWaveDataset"])
    )

    train = cfg_yaml.get("train", {})
    opt = cfg_yaml.get("optimizer", {})
    model_cfg = cfg_yaml.get("model", {})
    loss = cfg_yaml.get("loss", {})
    mask = cfg_yaml.get("mask_augmentation", {})
    sampling = cfg_yaml.get("sampling", {})
    output = cfg_yaml.get("output", {})

    save_dir = ROOT / output.get("save_dir", f"ckpt/fair_di_{prior}")
    fig_dir = ROOT / output.get("fig_dir", f"figures/fair_di_{prior}")
    resume = bool(train.get("resume", False) if args.resume is None else args.resume)

    if cfg_yaml.get("init_checkpoint"):
        raise ValueError("Fair-comparison configs must not set init_checkpoint; train from scratch.")

    device = choose_device(direct_mod, args.device or train.get("device", "auto"))
    cfg = direct_mod.TrainConfig(
        n_train=int(train.get("n_train", 100_000)),
        n_val=int(train.get("n_val", 2_048)),
        z_max_km=float(train.get("z_max_km", 150.0)),
        z_max_num=int(train.get("z_max_num", 256)),
        dz_km=float(train.get("dz_km", 0.5)),
        batch_size=int(train.get("batch_size", 64)),
        num_workers=int(train.get("num_workers", 4)),
        seed=int(train.get("seed", 642026)),
        stats_batches=int(train.get("stats_batches", 64)),
        save_dir=str(save_dir),
        fig_dir=str(fig_dir),
        resume=resume,
        init_checkpoint=None,
        mask_augment=bool(mask.get("enabled", True)),
        mask_augment_val=bool(mask.get("validation_enabled", True)),
        mask_window_presets=parse_period_windows(direct_mod, mask),
        mask_window_preset_prob=float(mask.get("preset_prob", 0.65)),
        mask_random_min_width_s=float(mask.get("random_min_width_s", 15.0)),
        mask_random_max_width_s=float(mask.get("random_max_width_s", 45.0)),
        mask_internal_holes_max=int(mask.get("internal_holes_max", 6)),
        mask_hole_point_prob=float(mask.get("hole_point_prob", 0.04)),
        cond_base_channels=int(model_cfg.get("cond_base_channels", 64)),
        cond_dim=int(model_cfg.get("cond_dim", 256)),
        flow_hidden=int(model_cfg.get("flow_hidden", 1024)),
        time_dim=int(model_cfg.get("time_dim", 64)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        epochs=int(train.get("epochs", 24)),
        lr=float(opt.get("lr", 2.0e-4)),
        min_lr=float(opt.get("min_lr", 1.0e-6)),
        weight_decay=float(opt.get("weight_decay", 1.0e-4)),
        grad_clip=float(opt.get("grad_clip", 5.0)),
        use_amp=bool(opt.get("use_amp", True)),
        lambda_rec=float(loss.get("lambda_rec", 0.5)),
        lambda_slope=float(loss.get("lambda_slope", 0.05)),
        lambda_curvature=float(loss.get("lambda_curvature", 0.01)),
        flow_beta=float(loss.get("flow_beta", 0.5)),
        val_every=int(train.get("val_every", 1)),
        eval_num_samples=int(sampling.get("train_eval_num_samples", 4)),
        eval_num_steps=int(sampling.get("train_eval_num_steps", 24)),
        log_every_steps=int(train.get("log_every_steps", 100)),
        plot_every_steps=int(train.get("plot_every_steps", 0)),
        plot_num_samples=int(sampling.get("plot_num_samples", 16)),
        device=device,
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, save_dir / "source_config.yaml")
    with (save_dir / "fair_protocol_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "config": str(args.config),
                "prior": prior,
                "fairness_constraint": "No warm-start. Only structural prior generator differs between paired configs.",
                "dataset_module": dataset_mod.__name__,
                "save_dir": str(save_dir),
                "fig_dir": str(fig_dir),
            },
            f,
            indent=2,
        )

    train_ds = dataset_mod.SurfaceWaveDataset(
        n_samples=cfg.n_train,
        z_max_km=cfg.z_max_km,
        z_max_num=cfg.z_max_num,
        dz_km=cfg.dz_km,
        seed=cfg.seed,
    )
    val_ds = dataset_mod.SurfaceWaveDataset(
        n_samples=cfg.n_val,
        z_max_km=cfg.z_max_km,
        z_max_num=cfg.z_max_num,
        dz_km=cfg.dz_km,
        seed=cfg.seed + 1_000_000,
    )
    device_obj = direct_mod.torch.device(cfg.device)
    train_loader = direct_mod.make_loader(train_ds, cfg=cfg, shuffle=True, device=device_obj)
    val_loader = direct_mod.make_loader(val_ds, cfg=cfg, shuffle=False, device=device_obj)
    direct_mod.train_disp2struct_crf(train_loader, val_loader=val_loader, cfg=cfg)
    print(f"[done] Fair DI-{prior.capitalize()} checkpoint directory: {cfg.save_dir}")


if __name__ == "__main__":
    main()
