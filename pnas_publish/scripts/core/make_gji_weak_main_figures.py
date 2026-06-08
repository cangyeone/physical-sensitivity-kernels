#!/usr/bin/env python3
"""Generate weak-prior manuscript figures for the GJI draft."""

from pathlib import Path
import importlib.util
import sys
from types import SimpleNamespace

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
GJI = ROOT / "gji_dnn_posterior_inversion"
FIG_DIR = GJI / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PAPER_SCRIPT = ROOT / "overleaf_inversion_paper" / "disp_inv_scripts" / "make_paper_figures.py"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    sys.dont_write_bytecode = True
    torch.manual_seed(2026)
    np.random.seed(2026)

    paper = load_module("make_paper_figures_weak_gji", PAPER_SCRIPT)
    paper.OUT = FIG_DIR
    boundary = load_module("eval_prior_boundary_effect_weak_gji", ROOT / "scripts" / "eval_prior_boundary_effect.py")
    boundary.FIGURES_DIR = FIG_DIR
    boundary.RESULTS_DIR = GJI / "results"

    direct_mod = load_module("disp_inv_train_v12_weak_gji", ROOT / "disp_inv_train.v1.2.py")
    weak_data_mod = load_module("generate_data_weak_prior_gji", ROOT / "utils" / "generate_data_weak_prior.py")

    model, ckpt = paper.restore_model(direct_mod, ROOT / "ckpt" / "disp2struct_crf.v1.2_cp_weak" / "best.pt")
    model_batch, disp_batch, mask_batch = paper.collect_dataset(weak_data_mod, n=128)
    target = model_batch[:, 1:4, :].float()
    disp_batch = disp_batch.float()
    mask_batch = mask_batch.float()

    with torch.no_grad():
        sample_chunks = []
        for i in range(0, target.size(0), 8):
            out = model.sample(
                disp_batch[i : i + 8],
                mask_batch[i : i + 8],
                num_samples=32,
                num_steps=24,
            )
            sample_chunks.append(out["profile_samples"])
        samples = torch.cat(sample_chunks, dim=0)

    median = samples.median(dim=1).values
    per_example_vs_mae = (median[:, 1] - target[:, 1]).abs().mean(dim=1)
    example_index = int(torch.argsort(per_example_vs_mae)[len(per_example_vs_mae) // 2])

    qs = np.array([20, 40, 60, 68, 80, 90])
    split_index = target.size(0) // 2
    calibration_samples = samples[:split_index]
    calibration_target = target[:split_index]
    test_samples = samples[split_index:]
    test_target = target[split_index:]
    temperature_scale = paper.fit_temperature_scale(calibration_samples, calibration_target, nominal_percent=68.0)
    raw_test_cov = paper.nominal_coverage(test_samples, test_target, qs)
    scaled_test_cov = paper.nominal_coverage(
        paper.scaled_samples_about_median(test_samples, temperature_scale),
        test_target,
        qs,
    )

    paper.fig_control_points(model, target[:100], highlight_index=example_index % 100)
    paper.fig_posterior_profiles(model, target[example_index], samples[example_index])
    paper.fig_posterior_density_vs(model, target[example_index], samples[example_index])
    paper.fig_uncertainty(model, samples[example_index])
    paper.fig_coverage(qs, raw_test_cov, scaled_test_cov, temperature_scale)

    device = torch.device("cpu")
    strong_mod = boundary.import_from_path("prior_boundary_generate_data_weak_gji", ROOT / "utils" / "generate_data.py")
    periods = np.linspace(2.0, 60.0, 59).astype(np.float32)
    envelope = boundary.prior_envelope(strong_mod, 512, 2026 + 10)
    in_models_full, in_disp, in_mask = boundary.dataset_to_arrays(boundary.strong_dataset(strong_mod, 64, 2026 + 20))
    test_sets = {
        "in-prior": (in_models_full[:, 1:4, :], in_disp, in_mask),
        "boundary": boundary.parametric_dataset(strong_mod, "boundary", 64, 2026 + 30, 256, periods),
        "out-of-prior": boundary.parametric_dataset(strong_mod, "out-of-prior", 64, 2026 + 40, 256, periods),
    }
    boundary_model, _ = boundary.load_direct_model(
        ROOT / "disp_inv_train.v1.2.py",
        ROOT / "ckpt" / "disp2struct_crf.v1.2_cp_weak" / "best.pt",
        device,
    )
    eval_args = SimpleNamespace(posterior_samples=16, sampling_steps=12, batch_size=4)
    weak_rows, weak_diag = boundary.evaluate_direct(
        "DI-Weak",
        boundary_model,
        test_sets,
        strong_mod,
        envelope,
        device,
        eval_args,
    )
    boundary.write_csv(GJI / "results" / "prior_boundary_weak.csv", weak_rows)
    boundary.plot_direct_examples(FIG_DIR / "direct_prior_boundary_examples.png", weak_diag)

    np.savez_compressed(
        FIG_DIR / "weak_posterior_figure_samples.npz",
        depth_km=model.depth_grid.detach().cpu().numpy(),
        channel_names=np.asarray(["Vp", "Vs", "rho"]),
        channel_units=np.asarray(["km s^-1", "km s^-1", "g/cm^3"]),
        weak_example_index=np.asarray(example_index, dtype=np.int64),
        weak_target=target[example_index].detach().cpu().numpy(),
        weak_dispersion=disp_batch[example_index].detach().cpu().numpy(),
        weak_mask=mask_batch[example_index].detach().cpu().numpy(),
        weak_posterior_samples=samples[example_index].detach().cpu().numpy(),
        posterior_samples=np.asarray(samples.size(1), dtype=np.int64),
        sampling_steps=np.asarray(24, dtype=np.int64),
        checkpoint_epoch=np.asarray(int(ckpt["epoch"]), dtype=np.int64),
        checkpoint_global_step=np.asarray(int(ckpt["global_step"]), dtype=np.int64),
        median_example_vs_mae=np.asarray(float(per_example_vs_mae[example_index]), dtype=np.float32),
    )

    print(f"wrote weak-prior figures to {FIG_DIR}")
    print(f"selected weak example {example_index} with Vs MAE {float(per_example_vs_mae[example_index]):.4f} km s^-1")
    print(f"weak-prior coverage scale {temperature_scale:.3f}")


if __name__ == "__main__":
    main()
