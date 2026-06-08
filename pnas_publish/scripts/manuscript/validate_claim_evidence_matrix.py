#!/usr/bin/env python3
"""Validate the manuscript claim-to-evidence matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


SCRIPTS = Path(__file__).resolve().parent
OVERLEAF = SCRIPTS.parent

REQUIRED_CATEGORIES = {
    "posterior_definition",
    "synthetic_accuracy",
    "posterior_calibration",
    "ak135_stress_test",
    "field_data_stress_test",
    "posterior_predictive",
    "observation_noise_sensitivity",
    "sampling_sensitivity",
    "depth_control_basis",
    "inference_cost",
    "open_research",
}

ALLOWED_EVIDENCE_TYPES = {"file", "json", "latex", "npz"}


def load_json(path: Path, errors: list[str]) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        errors.append(f"Missing JSON file: {path}")
    except json.JSONDecodeError as exc:
        errors.append(f"Invalid JSON in {path}: {exc}")
    return None


def json_path_value(data: Any, dotted_path: str) -> tuple[bool, Any]:
    current = data
    for part in dotted_path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return False, None
    return True, current


def value_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict)):
        return bool(value)
    return True


def check_manuscript_terms(claim: dict[str, Any], manuscript_text: str, errors: list[str]) -> None:
    claim_id = claim.get("id", "<missing id>")
    terms = claim.get("manuscript_terms")
    if not isinstance(terms, list) or not terms:
        errors.append(f"{claim_id}: manuscript_terms must be a nonempty list.")
        return
    folded = manuscript_text.casefold()
    missing = [term for term in terms if not isinstance(term, str) or term.casefold() not in folded]
    if missing:
        errors.append(f"{claim_id}: manuscript terms not found: " + ", ".join(str(item) for item in missing))


def check_json_evidence(
    path: Path,
    required_paths: list[str],
    allow_empty_paths: set[str],
    claim_id: str,
    errors: list[str],
) -> None:
    data = load_json(path, errors)
    if data is None:
        return
    for dotted_path in required_paths:
        ok, value = json_path_value(data, dotted_path)
        if not ok:
            errors.append(f"{claim_id}: missing JSON path {dotted_path} in {path}")
        elif dotted_path not in allow_empty_paths and not value_present(value):
            errors.append(f"{claim_id}: empty JSON value at {dotted_path} in {path}")


def check_npz_evidence(path: Path, required_keys: list[str], claim_id: str, errors: list[str]) -> None:
    try:
        import numpy as np
    except ImportError as exc:
        errors.append(f"{claim_id}: numpy is required to inspect NPZ evidence: {exc}")
        return
    try:
        archive = np.load(path)
    except Exception as exc:
        errors.append(f"{claim_id}: could not read NPZ evidence {path}: {exc}")
        return
    keys = set(archive.files)
    missing = [key for key in required_keys if key not in keys]
    if missing:
        errors.append(f"{claim_id}: missing NPZ keys in {path}: " + ", ".join(missing))


def check_evidence(root: Path, claim: dict[str, Any], errors: list[str]) -> int:
    claim_id = claim.get("id", "<missing id>")
    evidence = claim.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        errors.append(f"{claim_id}: evidence must be a nonempty list.")
        return 0
    checked = 0
    for index, item in enumerate(evidence, start=1):
        if not isinstance(item, dict):
            errors.append(f"{claim_id}: evidence item {index} must be an object.")
            continue
        evidence_type = item.get("type")
        rel_path = item.get("path")
        if evidence_type not in ALLOWED_EVIDENCE_TYPES:
            errors.append(f"{claim_id}: unsupported evidence type {evidence_type!r}.")
            continue
        if not isinstance(rel_path, str) or not rel_path.strip():
            errors.append(f"{claim_id}: evidence item {index} has no path.")
            continue
        path = root / rel_path
        if not path.exists():
            errors.append(f"{claim_id}: missing evidence file {rel_path}.")
            continue
        checked += 1
        if evidence_type == "json":
            required_paths = item.get("required_json_paths", [])
            if not isinstance(required_paths, list) or not required_paths:
                errors.append(f"{claim_id}: JSON evidence {rel_path} must list required_json_paths.")
            else:
                allow_empty_paths = item.get("allow_empty_json_paths", [])
                if not isinstance(allow_empty_paths, list):
                    errors.append(f"{claim_id}: allow_empty_json_paths must be a list for {rel_path}.")
                    allow_empty_paths = []
                check_json_evidence(path, required_paths, set(allow_empty_paths), claim_id, errors)
        elif evidence_type == "npz":
            required_keys = item.get("required_npz_keys", [])
            if not isinstance(required_keys, list) or not required_keys:
                errors.append(f"{claim_id}: NPZ evidence {rel_path} must list required_npz_keys.")
            else:
                check_npz_evidence(path, required_keys, claim_id, errors)
    return checked


def validate(root: Path, matrix_path: Path) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    matrix = load_json(matrix_path, errors)
    if not isinstance(matrix, dict):
        return {
            "ok": False,
            "errors": errors or ["Claim-evidence matrix must be a JSON object."],
            "warnings": warnings,
        }

    manuscript_rel = matrix.get("manuscript", "agujournaltemplate.tex")
    manuscript_path = root / manuscript_rel
    try:
        manuscript_text = manuscript_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        manuscript_text = ""
        errors.append(f"Missing manuscript file: {manuscript_rel}")

    claims = matrix.get("claims")
    if not isinstance(claims, list) or not claims:
        errors.append("claims must be a nonempty list.")
        claims = []

    categories = set()
    evidence_checked = 0
    seen_ids = set()
    for index, claim in enumerate(claims, start=1):
        if not isinstance(claim, dict):
            errors.append(f"claims[{index}] must be an object.")
            continue
        claim_id = claim.get("id")
        if not isinstance(claim_id, str) or not claim_id.strip():
            errors.append(f"claims[{index}] is missing id.")
            claim_id = f"<claim {index}>"
        elif claim_id in seen_ids:
            errors.append(f"Duplicate claim id: {claim_id}")
        seen_ids.add(claim_id)

        category = claim.get("category")
        if not isinstance(category, str) or not category.strip():
            errors.append(f"{claim_id}: category is required.")
        else:
            categories.add(category)
        if not isinstance(claim.get("short_claim"), str) or not claim["short_claim"].strip():
            errors.append(f"{claim_id}: short_claim is required.")
        if not isinstance(claim.get("scope_limit"), str) or not claim["scope_limit"].strip():
            errors.append(f"{claim_id}: scope_limit is required.")
        check_manuscript_terms(claim, manuscript_text, errors)
        evidence_checked += check_evidence(root, claim, errors)

    missing_categories = sorted(REQUIRED_CATEGORIES - categories)
    if missing_categories:
        errors.append("Missing required claim categories: " + ", ".join(missing_categories))
    extra_categories = sorted(categories - REQUIRED_CATEGORIES)
    if extra_categories:
        warnings.append("Extra claim categories: " + ", ".join(extra_categories))

    return {
        "ok": not errors,
        "matrix": str(matrix_path),
        "manuscript": str(manuscript_path),
        "claim_count": len(claims),
        "evidence_items_checked": evidence_checked,
        "categories": sorted(categories),
        "missing_categories": missing_categories,
        "errors": errors,
        "warnings": warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the claim-to-evidence matrix for the manuscript.")
    parser.add_argument("--root", type=Path, default=OVERLEAF, help="Overleaf project root.")
    parser.add_argument("--matrix", type=Path, default=OVERLEAF / "claim_evidence_matrix.json")
    parser.add_argument("--output", type=Path, help="Optional JSON report output path.")
    args = parser.parse_args()

    root = args.root.resolve()
    matrix_path = args.matrix if args.matrix.is_absolute() else root / args.matrix
    result = validate(root, matrix_path)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
