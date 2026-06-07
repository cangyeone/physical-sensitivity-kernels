#!/usr/bin/env python3
"""Update the GJI manuscript from verified fair-comparison result files."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Required result file is missing: {path}")
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def require_outputs(paths: list[Path]) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        joined = "\n  ".join(str(path) for path in missing)
        raise SystemExit(
            "Production fair-comparison outputs are incomplete; manuscript was not modified.\n"
            "Missing files:\n  "
            + joined
        )


def fmt(value, digits=3):
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "--"


def row_for(rows: list[dict], method: str, test_set: str) -> dict:
    for row in rows:
        if row.get("method") == method and row.get("test_set") == test_set:
            return row
    raise KeyError(f"Missing row for {method}/{test_set}")


def build_validation_table(rows: list[dict]) -> str:
    strong = row_for(rows, "DI-Strong", "in-prior")
    weak = row_for(rows, "DI-Weak", "in-prior")
    n = int(float(strong["n"]))
    return rf"""\begin{{table*}}
\caption{{Matched in-prior diagnostic statistics for DI-Strong and DI-Weak trained from scratch with identical budgets. Posterior summaries use the production sampling configuration recorded in the fair-comparison archive. Coverage is mean pointwise 16--84 per cent interval coverage over $V_P$, $V_S$ and density; bootstrap confidence intervals are reported in the archived CSV/JSON tables.}}
\label{{tab:validation}}
\centering
\begin{{tabular}}{{@{{}}lccccc@{{}}}}
\toprule
Method & $N$ & $V_P$ MAE & $V_S$ MAE & Density MAE & Coverage \\
 & & (km s$^{{-1}}$) & (km s$^{{-1}}$) & (g cm$^{{-3}}$) & \\
\midrule
DI-Strong & {n} & {fmt(strong['vp_mae'])} & {fmt(strong['vs_mae'])} & {fmt(strong['rho_mae'])} & {fmt(strong['coverage_16_84_mean'])} \\
DI-Weak & {int(float(weak['n']))} & {fmt(weak['vp_mae'])} & {fmt(weak['vs_mae'])} & {fmt(weak['rho_mae'])} & {fmt(weak['coverage_16_84_mean'])} \\
\bottomrule
\end{{tabular}}
\end{{table*}}"""


def build_prior_table(rows: list[dict]) -> str:
    order = [("DI-Strong", "in-prior"), ("DI-Strong", "boundary"), ("DI-Strong", "out-of-prior"), ("DI-Weak", "in-prior"), ("DI-Weak", "boundary"), ("DI-Weak", "out-of-prior")]
    body = []
    for method, test_set in order:
        r = row_for(rows, method, test_set)
        body.append(
            f"{method} & {test_set.title()} & {int(float(r['n']))} & {fmt(r['vs_mae'])} & "
            f"{fmt(r.get('pred_disp_mae', 'nan'))} & {fmt(r['coverage_vs'])} & "
            f"{fmt(r.get('pred_inside_given_target_outside', 'nan'))} \\\\"
        )
    return rf"""\begin{{table*}}
\caption{{Matched prior-support diagnostic for DI-Strong and DI-Weak trained from scratch with identical budgets. The table is a prior-support reliability audit, not a general ranking of priors. Pull-in is the fraction of target values outside the strong-prior envelope whose posterior medians return inside that envelope. Bootstrap confidence intervals are archived with the full result tables.}}
\label{{tab:priorboundary}}
\centering
\begin{{tabular}}{{@{{}}llccccc@{{}}}}
\toprule
Method & Regime & $N$ & $V_S$ MAE & Disp. MAE & $V_S$ coverage & Pull-in \\
 & & & (km s$^{{-1}}$) & (km s$^{{-1}}$) & & \\
\midrule
{chr(10).join(body)}
\bottomrule
\end{{tabular}}
\end{{table*}}"""


def replace_table(tex: str, label: str, replacement: str) -> str:
    pattern = re.compile(r"\\begin\{table\*\}.*?\\label\{" + re.escape(label) + r"\}.*?\\end\{table\*\}", re.S)
    tex_new, n = pattern.subn(lambda _: replacement, tex, count=1)
    if n != 1:
        raise RuntimeError(f"Could not replace table label {label}")
    return tex_new


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", type=Path, default=ROOT / "results/fair_di_comparison/production")
    p.add_argument("--fig-dir", type=Path, default=ROOT / "figures/fair_di_comparison/production")
    p.add_argument("--manuscript", type=Path, default=ROOT / "gji_dnn_posterior_inversion/gjilguid2e.tex")
    args = p.parse_args()
    require_outputs(
        [
            args.results_dir / "fair_di_metrics.csv",
            args.results_dir / "calibration/calibration_metrics.csv",
            args.results_dir / "noise/noise_sensitivity.csv",
            args.results_dir / "missing_band/missing_band_uncertainty.csv",
        ]
    )
    metrics = read_csv(args.results_dir / "fair_di_metrics.csv")
    read_csv(args.results_dir / "calibration/calibration_metrics.csv")
    read_csv(args.results_dir / "noise/noise_sensitivity.csv")
    read_csv(args.results_dir / "missing_band/missing_band_uncertainty.csv")
    tex = args.manuscript.read_text(encoding="utf-8")
    stale_patterns = [
        r"The weak-prior sampler used in this manuscript is.*?not a final production-scale retraining\.",
        r"Because the current DI-Weak checkpoint is limited-scale,.*?not as the final weak-prior benchmark\.",
    ]
    stale_replacements = [
        "The weak-prior sampler used in this manuscript is trained from scratch with the same architecture, optimizer, learning-rate schedule, batch size, train/validation sizes, epochs, loss weights, mask augmentation and seed as the strong-prior sampler. The only intended difference is the structural prior generator.",
        "Because DI-Strong and DI-Weak are trained with matched budgets, the numerical comparison isolates the structural-prior generator more directly than the earlier diagnostic; it remains a prior-support audit rather than a universal prior ranking.",
    ]
    for pattern, replacement in zip(stale_patterns, stale_replacements):
        tex = re.sub(pattern, replacement, tex, flags=re.S)
    tex = replace_table(tex, "tab:validation", build_validation_table(metrics))
    tex = replace_table(tex, "tab:priorboundary", build_prior_table(metrics))
    tex = tex.replace("\\subsection{Synthetic generator and observation protocol}\n\\subsection{Synthetic generator and observation protocol}", "\\subsection{Synthetic generator and observation protocol}")
    args.manuscript.write_text(tex, encoding="utf-8")
    print(f"Updated {args.manuscript} from {args.results_dir}")


if __name__ == "__main__":
    main()
