#!/usr/bin/env python3
"""Train a matched deterministic DNN direct-inversion baseline."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def write_csv_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


class DeterministicControlPointDNN(nn.Module):
    def __init__(self, direct_mod, stats: dict, cfg: dict):
        super().__init__()
        self.adapter = direct_mod.Disp2StructCRF(
            H=int(stats["H"].item()),
            T=int(stats["T"].item()),
            profile_channels=3,
            cond_base_channels=int(cfg["model"].get("cond_base_channels", 64)),
            cond_dim=int(cfg["model"].get("cond_dim", 256)),
            flow_hidden=int(cfg["model"].get("flow_hidden", 1024)),
            time_dim=int(cfg["model"].get("time_dim", 64)),
            dropout=float(cfg["model"].get("dropout", 0.1)),
            reference_profile=stats["reference_profile"],
            profile_scale=stats["profile_scale"],
            depth_grid=stats["depth_grid"],
            period_minmax=tuple(float(x) for x in stats["period_minmax"].tolist()),
            disp_mean=stats["disp_mean"],
            disp_scale=stats["disp_scale"],
        )
        cond_dim = int(cfg["model"].get("cond_dim", 256))
        hidden = int(cfg["model"].get("flow_hidden", 1024))
        dropout = float(cfg["model"].get("dropout", 0.1))
        self.head = nn.Sequential(
            nn.Linear(cond_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.adapter.output_dim),
        )

    def forward(self, disp: torch.Tensor, mask: torch.Tensor) -> dict:
        cond = self.adapter.encode(disp, mask)
        z = self.head(cond).reshape(disp.size(0), 3, self.adapter.Nc)
        profile = self.adapter.z_to_profile(z)
        return {"z": z, "profile": profile}


def parse_period_windows(direct_mod, cfg: dict):
    windows = cfg.get("windows", [[2.0, 40.0], [10.0, 30.0], [5.0, 35.0], [15.0, 45.0]])
    if isinstance(windows, str):
        return direct_mod.parse_period_windows(windows)
    return tuple((float(lo), float(hi)) for lo, hi in windows)


def to_train_config(direct_mod, cfg_yaml: dict, save_dir: Path, fig_dir: Path, device: str):
    train = cfg_yaml.get("train", {})
    opt = cfg_yaml.get("optimizer", {})
    loss = cfg_yaml.get("loss", {})
    mask = cfg_yaml.get("mask_augmentation", {})
    return direct_mod.TrainConfig(
        n_train=int(train.get("n_train", 100000)),
        n_val=int(train.get("n_val", 2048)),
        z_max_km=float(train.get("z_max_km", 150.0)),
        z_max_num=int(train.get("z_max_num", 256)),
        dz_km=float(train.get("dz_km", 0.5)),
        batch_size=int(train.get("batch_size", 64)),
        num_workers=int(train.get("num_workers", 4)),
        seed=int(train.get("seed", 642026)),
        stats_batches=int(train.get("stats_batches", 64)),
        save_dir=str(save_dir),
        fig_dir=str(fig_dir),
        resume=False,
        init_checkpoint=None,
        mask_augment=bool(mask.get("enabled", True)),
        mask_augment_val=bool(mask.get("validation_enabled", True)),
        mask_window_presets=parse_period_windows(direct_mod, mask),
        mask_window_preset_prob=float(mask.get("preset_prob", 0.65)),
        mask_random_min_width_s=float(mask.get("random_min_width_s", 15.0)),
        mask_random_max_width_s=float(mask.get("random_max_width_s", 45.0)),
        mask_internal_holes_max=int(mask.get("internal_holes_max", 6)),
        mask_hole_point_prob=float(mask.get("hole_point_prob", 0.04)),
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
        log_every_steps=int(train.get("log_every_steps", 100)),
        device=device,
    )


def batch_loss(model, direct_mod, batch, device, cfg, mask_kwargs):
    model_batch, disp_batch, mask_batch = direct_mod.move_to_device(batch, device)
    target = model_batch[:, 1:4, :].float()
    disp_batch = disp_batch.float()
    mask_batch = mask_batch.float()
    if mask_kwargs:
        mask_batch = direct_mod.augment_dispersion_period_mask(disp_batch, mask_batch, **mask_kwargs)
    out = model(disp_batch, mask_batch)
    target_z = model.adapter.profile_to_full_z(target)
    pred_z = model.adapter.profile_to_full_z(out["profile"])
    rec = F.smooth_l1_loss(pred_z, target_z, beta=cfg.flow_beta)
    slope = direct_mod.slope_loss(pred_z, target_z)
    curv = direct_mod.curvature_loss(pred_z, target_z)
    loss = rec + cfg.lambda_slope * slope + cfg.lambda_curvature * curv
    metrics = direct_mod.profile_metrics(out["profile"].detach(), target)
    return loss, {"loss": loss, "rec_loss": rec, "slope_loss": slope, "curvature_loss": curv, **metrics}


def run_epoch(model, loader, direct_mod, device, cfg, mask_kwargs, optimizer=None):
    train = optimizer is not None
    model.train(train)
    totals = {}
    n_seen = 0
    for batch in loader:
        if train:
            optimizer.zero_grad(set_to_none=True)
        loss, stats = batch_loss(model, direct_mod, batch, device, cfg, mask_kwargs)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        bsz = batch[0].size(0)
        n_seen += bsz
        for key, value in stats.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu() if torch.is_tensor(value) else value) * bsz
    return {key: value / max(n_seen, 1) for key, value in totals.items()}


def save_checkpoint(path: Path, model, optimizer, scheduler, epoch, best_loss, cfg_yaml, stats, cfg) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": "deterministic_control_point_dnn",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_loss,
            "config_yaml": cfg_yaml,
            "train_config": direct_config_dict(cfg),
            "normalization": stats,
            "depth_grid": model.adapter.depth_grid.detach().cpu(),
            "control_indices": model.adapter.control_indices.detach().cpu(),
        },
        path,
    )


def direct_config_dict(cfg) -> dict:
    return {key: getattr(cfg, key) for key in cfg.__dataclass_fields__}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg_yaml = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    prior = cfg_yaml.get("prior")
    if prior not in {"strong", "weak"}:
        raise ValueError("Config field 'prior' must be strong or weak")
    direct_mod = import_from_path("disp_inv_train_v13_det", ROOT / "disp_inv_train.v1.3.py")
    dataset_mod = __import__("utils.generate_data" if prior == "strong" else "utils.generate_data_weak_prior", fromlist=["SurfaceWaveDataset"])
    output = cfg_yaml.get("output", {})
    save_dir = ROOT / output.get("save_dir", f"ckpt/det_di_{prior}_full_seed642026")
    fig_dir = ROOT / output.get("fig_dir", f"figures/det_di_{prior}_full_seed642026")
    device_str = args.device or cfg_yaml.get("train", {}).get("device", "auto")
    device = torch.device(direct_mod.default_device() if device_str == "auto" else device_str)
    cfg = to_train_config(direct_mod, cfg_yaml, save_dir, fig_dir, str(device))
    direct_mod.set_seed(cfg.seed)

    save_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, save_dir / "source_config.yaml")
    train_ds = dataset_mod.SurfaceWaveDataset(cfg.n_train, z_max_km=cfg.z_max_km, z_max_num=cfg.z_max_num, dz_km=cfg.dz_km, seed=cfg.seed)
    val_ds = dataset_mod.SurfaceWaveDataset(cfg.n_val, z_max_km=cfg.z_max_km, z_max_num=cfg.z_max_num, dz_km=cfg.dz_km, seed=cfg.seed + 1_000_000)
    train_loader = direct_mod.make_loader(train_ds, cfg, shuffle=True, device=device)
    val_loader = direct_mod.make_loader(val_ds, cfg, shuffle=False, device=device)
    mask_kwargs = {
        "enabled": cfg.mask_augment,
        "preset_windows": cfg.mask_window_presets,
        "preset_prob": cfg.mask_window_preset_prob,
        "random_min_width_s": cfg.mask_random_min_width_s,
        "random_max_width_s": cfg.mask_random_max_width_s,
        "max_internal_holes": cfg.mask_internal_holes_max,
        "hole_point_prob": cfg.mask_hole_point_prob,
    }
    stats = direct_mod.estimate_training_stats(train_loader, device=device, max_batches=cfg.stats_batches, mask_augment_kwargs=mask_kwargs)
    model = DeterministicControlPointDNN(direct_mod, stats, cfg_yaml).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs, eta_min=cfg.min_lr)
    write_json(save_dir / "config_resolved.json", {"config_yaml": cfg_yaml, "train_config": direct_config_dict(cfg)})
    write_json(save_dir / "runtime_metadata.json", direct_mod.runtime_metadata(device, cfg))
    torch.save(stats, save_dir / "normalization_stats.pt")

    best = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_stats = run_epoch(model, train_loader, direct_mod, device, cfg, mask_kwargs, optimizer=optimizer)
        val_kwargs = dict(mask_kwargs)
        val_kwargs["enabled"] = cfg.mask_augment and cfg.mask_augment_val
        with torch.no_grad():
            val_stats = run_epoch(model, val_loader, direct_mod, device, cfg, val_kwargs, optimizer=None)
        scheduler.step()
        row = {
            "epoch": epoch,
            "epochs": cfg.epochs,
            "time_s": time.time() - t0,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"val_{k}": v for k, v in val_stats.items()},
        }
        row["selection_score"] = val_stats["loss"]
        row["is_best"] = bool(val_stats["loss"] < best)
        write_csv_row(save_dir / "epoch_metrics.csv", row)
        save_checkpoint(save_dir / "latest.pt", model, optimizer, scheduler, epoch, best, cfg_yaml, stats, cfg)
        if val_stats["loss"] < best:
            best = val_stats["loss"]
            save_checkpoint(save_dir / "best.pt", model, optimizer, scheduler, epoch, best, cfg_yaml, stats, cfg)
            write_json(save_dir / "best_selection.json", {"epoch": epoch, "best_val_loss": best, "row": row})
        print(
            f"[Epoch {epoch:03d}/{cfg.epochs:03d}] train={train_stats['loss']:.5f} "
            f"val={val_stats['loss']:.5f} vs={val_stats['vs_mae']:.4f}"
        )
    write_json(save_dir / "training_complete.json", {"epochs": cfg.epochs, "best_val_loss": best, "best_checkpoint": str(save_dir / "best.pt")})
    print(f"[done] deterministic DI-{prior} checkpoint directory: {save_dir}")


if __name__ == "__main__":
    main()
