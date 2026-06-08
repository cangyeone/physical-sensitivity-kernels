#!/usr/bin/env python3
"""Run production-oriented preflight checks on included manuscript figure PDFs."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent
OVERLEAF = SCRIPTS.parent

DEFAULT_IMAGE_BUDGET = {
    "fig05_posterior_density_vs.pdf": 1,
    "fig07_bayan_obo_field_test.pdf": 2,
}

DEFAULT_CREATOR_OVERRIDES = {
    "fig07_bayan_obo_field_test.pdf": "run_bayan_obo_field_test.py",
}


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


def run_tool(command: list[str]) -> tuple[bool, str, str]:
    try:
        completed = subprocess.run(command, check=False, text=True, capture_output=True)
    except FileNotFoundError as exc:
        return False, "", str(exc)
    return completed.returncode == 0, completed.stdout, completed.stderr


def parse_pdffonts(text: str) -> list[dict[str, str]]:
    fonts = []
    for line in text.splitlines()[2:]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        fonts.append(
            {
                "name": parts[0],
                "type": " ".join(parts[1:-6]),
                "encoding": parts[-6],
                "embedded": parts[-5],
                "subset": parts[-4],
                "unicode": parts[-3],
            }
        )
    return fonts


def parse_pdfimages(text: str) -> list[dict[str, str]]:
    images = []
    for line in text.splitlines()[2:]:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        images.append(
            {
                "page": parts[0],
                "number": parts[1],
                "type": parts[2],
                "width_px": parts[3],
                "height_px": parts[4],
                "color": parts[5],
                "components": parts[6],
            }
        )
    return images


def audit_pdf(path: Path, base_dir: Path, args: argparse.Namespace) -> dict:
    resolved = path if path.is_absolute() else base_dir / path
    record: dict = {
        "path": str(path),
        "exists": resolved.exists(),
        "status": "FAIL",
        "checks": {},
    }
    if not resolved.exists():
        record["failure_reasons"] = ["missing file"]
        return record

    pdfinfo_ok, pdfinfo_out, pdfinfo_err = run_tool(["pdfinfo", str(resolved)])
    info = parse_pdfinfo(pdfinfo_out) if pdfinfo_ok else {}
    width_pts, height_pts = parse_page_size(info.get("Page size", ""))
    pdffonts_ok, pdffonts_out, pdffonts_err = run_tool(["pdffonts", str(resolved)])
    fonts = parse_pdffonts(pdffonts_out) if pdffonts_ok else []
    pdfimages_ok, pdfimages_out, pdfimages_err = run_tool(["pdfimages", "-list", str(resolved)])
    images = parse_pdfimages(pdfimages_out) if pdfimages_ok else []
    filename = resolved.name
    allowed_images = DEFAULT_IMAGE_BUDGET.get(filename, 0)
    expected_creator = DEFAULT_CREATOR_OVERRIDES.get(filename, args.expected_creator)

    page_dimensions = [value for value in (width_pts, height_pts) if value is not None]
    checks = {
        "pdfinfo_ok": pdfinfo_ok,
        "pdffonts_ok": pdffonts_ok,
        "pdfimages_ok": pdfimages_ok,
        "single_page": info.get("Pages") == "1",
        "creator_ok": info.get("Creator") == expected_creator,
        "producer_ok": args.expected_producer_fragment in info.get("Producer", ""),
        "page_width_within_limit": bool(page_dimensions) and max(page_dimensions) <= args.max_page_side_pts,
        "page_short_side_ok": bool(page_dimensions) and min(page_dimensions) >= args.min_page_side_pts,
        "fonts_present": bool(fonts),
        "fonts_embedded": bool(fonts) and all(font["embedded"].lower() == "yes" for font in fonts),
        "fonts_subset": bool(fonts) and all(font["subset"].lower() == "yes" for font in fonts),
        "fonts_unicode_mapped": bool(fonts) and all(font["unicode"].lower() == "yes" for font in fonts),
        "image_budget_ok": len(images) <= allowed_images,
    }
    failure_reasons = [name for name, ok in checks.items() if not ok]
    record.update(
        {
            "status": "PASS" if not failure_reasons else "FAIL",
            "checks": checks,
            "failure_reasons": failure_reasons,
            "pdfinfo_error": pdfinfo_err.strip(),
            "pdffonts_error": pdffonts_err.strip(),
            "pdfimages_error": pdfimages_err.strip(),
            "pages": int(info["Pages"]) if info.get("Pages", "").isdigit() else None,
            "page_width_pts": width_pts,
            "page_height_pts": height_pts,
            "creator": info.get("Creator"),
            "expected_creator": expected_creator,
            "producer": info.get("Producer"),
            "pdf_version": info.get("PDF version"),
            "font_count": len(fonts),
            "fonts": fonts,
            "image_count": len(images),
            "allowed_image_count": allowed_images,
            "images": images,
        }
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit included manuscript figure PDFs for production preflight.")
    parser.add_argument("--manuscript", type=Path, default=OVERLEAF / "agujournaltemplate.tex")
    parser.add_argument("--output", type=Path, default=OVERLEAF / "figures/figure_preflight_audit.json")
    parser.add_argument("--expected-count", type=int, default=8)
    parser.add_argument("--expected-creator", default="make_paper_figures.py")
    parser.add_argument("--expected-producer-fragment", default="Matplotlib pdf backend")
    parser.add_argument("--max-page-side-pts", type=float, default=520.0)
    parser.add_argument("--min-page-side-pts", type=float, default=150.0)
    args = parser.parse_args()

    manuscript = args.manuscript.resolve()
    tex = manuscript.read_text(encoding="utf-8")
    base_dir = manuscript.parent
    figure_paths = [Path(path) for path in includegraphics_paths(tex)]
    records = [audit_pdf(path, base_dir, args) for path in figure_paths]
    failures = [record for record in records if record["status"] != "PASS"]
    output = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Production preflight audit for included manuscript figure PDFs.",
        "manuscript": str(manuscript),
        "expected_count": int(args.expected_count),
        "figure_count": len(records),
        "overall_status": "PASS" if len(records) == args.expected_count and not failures else "FAIL",
        "thresholds": {
            "expected_creator": args.expected_creator,
            "creator_overrides": DEFAULT_CREATOR_OVERRIDES,
            "expected_producer_fragment": args.expected_producer_fragment,
            "max_page_side_pts": float(args.max_page_side_pts),
            "min_page_side_pts": float(args.min_page_side_pts),
            "image_budget": DEFAULT_IMAGE_BUDGET,
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))
    return 0 if output["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
