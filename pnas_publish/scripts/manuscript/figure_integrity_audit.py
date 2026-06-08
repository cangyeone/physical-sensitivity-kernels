#!/usr/bin/env python3
"""Audit manuscript figure PDFs for basic publication-package integrity."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


SCRIPTS = Path(__file__).resolve().parent
OVERLEAF = SCRIPTS.parent


def includegraphics_paths(tex: str) -> list[str]:
    return re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)


def parse_pdfinfo(text: str) -> dict[str, str]:
    info: dict[str, str] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        info[key.strip()] = value.strip()
    return info


def parse_page_size(value: str) -> tuple[float | None, float | None]:
    match = re.search(r"([0-9.]+)\s+x\s+([0-9.]+)\s+pts", value)
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def render_nonwhite_fraction(pdf_path: Path, dpi: int) -> dict[str, float | int | str]:
    with tempfile.TemporaryDirectory(prefix="figure-audit-") as tmp:
        prefix = Path(tmp) / "page"
        command = [
            "pdftoppm",
            "-f",
            "1",
            "-l",
            "1",
            "-r",
            str(dpi),
            "-png",
            "-singlefile",
            str(pdf_path),
            str(prefix),
        ]
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
        if completed.returncode != 0:
            return {
                "render_ok": 0,
                "render_error": completed.stderr.strip() or completed.stdout.strip(),
                "nonwhite_fraction": 0.0,
                "width_px": 0,
                "height_px": 0,
            }
        image_path = prefix.with_suffix(".png")
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            pixels = np.asarray(rgb)
            total = pixels.shape[0] * pixels.shape[1]
            nonwhite = int(np.any(pixels < 250, axis=2).sum())
            dark = int(np.any(pixels < 220, axis=2).sum())
            width, height = rgb.size
        return {
            "render_ok": 1,
            "render_error": "",
            "nonwhite_fraction": float(nonwhite / total) if total else 0.0,
            "dark_pixel_fraction": float(dark / total) if total else 0.0,
            "width_px": int(width),
            "height_px": int(height),
        }


def audit_figure(path: Path, base_dir: Path, args: argparse.Namespace) -> dict:
    resolved_path = path if path.is_absolute() else base_dir / path
    exists = resolved_path.exists()
    record: dict = {
        "path": str(path),
        "exists": bool(exists),
        "status": "FAIL",
        "checks": {},
    }
    if not exists:
        record["checks"]["exists"] = False
        record["failure_reasons"] = ["missing file"]
        return record

    completed = subprocess.run(["pdfinfo", str(resolved_path)], check=False, text=True, capture_output=True)
    pdfinfo_ok = completed.returncode == 0
    info = parse_pdfinfo(completed.stdout) if pdfinfo_ok else {}
    width_pts, height_pts = parse_page_size(info.get("Page size", ""))
    size_bytes = resolved_path.stat().st_size
    render = render_nonwhite_fraction(resolved_path, args.render_dpi)

    checks = {
        "pdfinfo_ok": pdfinfo_ok,
        "single_page": info.get("Pages") == "1",
        "not_encrypted": info.get("Encrypted", "no").lower() == "no",
        "no_javascript": info.get("JavaScript", "no").lower() == "no",
        "size_bytes_min": size_bytes >= args.min_bytes,
        "page_size_min": (
            width_pts is not None
            and height_pts is not None
            and min(width_pts, height_pts) >= args.min_short_side_pts
        ),
        "render_ok": bool(render["render_ok"]),
        "render_nonblank": float(render["nonwhite_fraction"]) >= args.min_nonwhite_fraction,
    }
    failure_reasons = [name for name, ok in checks.items() if not ok]
    record.update(
        {
            "status": "PASS" if not failure_reasons else "FAIL",
            "checks": checks,
            "failure_reasons": failure_reasons,
            "size_bytes": int(size_bytes),
            "pages": int(info["Pages"]) if info.get("Pages", "").isdigit() else None,
            "encrypted": info.get("Encrypted"),
            "javascript": info.get("JavaScript"),
            "page_width_pts": width_pts,
            "page_height_pts": height_pts,
            "pdf_version": info.get("PDF version"),
            "creator": info.get("Creator"),
            "producer": info.get("Producer"),
            "render": render,
        }
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit included manuscript figure PDFs.")
    parser.add_argument("--manuscript", type=Path, default=OVERLEAF / "agujournaltemplate.tex")
    parser.add_argument("--output", type=Path, default=OVERLEAF / "figures/figure_integrity_audit.json")
    parser.add_argument("--expected-count", type=int, default=8)
    parser.add_argument("--min-bytes", type=int, default=5000)
    parser.add_argument("--min-short-side-pts", type=float, default=150.0)
    parser.add_argument("--render-dpi", type=int, default=72)
    parser.add_argument("--min-nonwhite-fraction", type=float, default=0.005)
    args = parser.parse_args()

    manuscript = args.manuscript.resolve()
    tex = manuscript.read_text(encoding="utf-8")
    base_dir = manuscript.parent
    figure_paths = [Path(path) for path in includegraphics_paths(tex)]
    records = [audit_figure(path, base_dir, args) for path in figure_paths]
    failures = [record for record in records if record["status"] != "PASS"]
    output = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Pre-submission integrity audit for included manuscript figure PDFs.",
        "manuscript": str(manuscript),
        "expected_count": int(args.expected_count),
        "figure_count": len(records),
        "overall_status": "PASS" if len(records) == args.expected_count and not failures else "FAIL",
        "thresholds": {
            "min_bytes": int(args.min_bytes),
            "min_short_side_pts": float(args.min_short_side_pts),
            "render_dpi": int(args.render_dpi),
            "min_nonwhite_fraction": float(args.min_nonwhite_fraction),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0 if output["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
