#!/usr/bin/env python3
"""Run and summarize the deterministic gates for the JGR:SE submission package."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRACKED_PLACEHOLDERS = [
    "Author Name",
    "Institution, City, Country",
    "email@example.com",
    "software DOI pending",
    "synthetic evaluation DOI pending",
    "Funding sources",
    "Author names, order, affiliations",
]

LATEX_WARNING_PATTERNS = [
    "undefined references",
    "undefined citations",
    "Rerun to get cross-references",
    "LaTeX Warning: Citation",
    "LaTeX Warning: Reference",
]


def run_command(name: str, command: list[str], cwd: Path, timeout: int) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        return {
            "name": name,
            "status": "FAIL",
            "command": command,
            "returncode": None,
            "started_utc": started.isoformat(),
            "error": str(exc),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "name": name,
            "status": "FAIL",
            "command": command,
            "returncode": None,
            "started_utc": started.isoformat(),
            "error": f"Timed out after {timeout} s.",
            "stdout_tail": (exc.stdout or "")[-2000:] if isinstance(exc.stdout, str) else "",
            "stderr_tail": (exc.stderr or "")[-2000:] if isinstance(exc.stderr, str) else "",
        }

    return {
        "name": name,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "command": command,
        "returncode": completed.returncode,
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def placeholder_blockers(root: Path) -> list[str]:
    text = (root / "agujournaltemplate.tex").read_text(encoding="utf-8")
    return [placeholder for placeholder in TRACKED_PLACEHOLDERS if placeholder in text]


def latex_log_findings(root: Path) -> list[str]:
    log_path = root / "agujournaltemplate.log"
    if not log_path.exists():
        return ["Missing agujournaltemplate.log."]
    text = log_path.read_text(encoding="utf-8", errors="replace")
    findings = []
    for pattern in LATEX_WARNING_PATTERNS:
        if re.search(re.escape(pattern), text, flags=re.I):
            findings.append(pattern)
    return findings


def skipped_gate(name: str, reason: str, command: list[str] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "status": "SKIP",
        "reason": reason,
        "command": command or [],
    }


def command_name(command: list[str]) -> str:
    return " ".join(command)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root).resolve()
    python = sys.executable
    script_dir = Path(args.script_dir)
    script = lambda name: str(script_dir / name)
    commands = [
        ("submission_qa", [python, script("submission_qa.py")]),
        ("final_values_template", [python, script("validate_final_submission_values.py"), "--template", "final_submission_values.template.json"]),
        ("final_sync_template", [python, script("validate_final_submission_sync.py"), "--template", "final_submission_values.template.json"]),
        ("final_values_apply_template_dry_run", [python, script("apply_final_submission_values.py"), "--template", "--dry-run", "--output", "/tmp/final_submission_apply_plan.json"]),
        ("zenodo_metadata_template", [python, "-m", "json.tool", "zenodo_metadata_templates.json"]),
        ("claim_evidence_matrix", [python, script("validate_claim_evidence_matrix.py"), "--output", "/tmp/claim_evidence_matrix_check.json"]),
        ("central_posterior_framing_audit", [python, script("central_posterior_framing_audit.py"), "--output", "/tmp/central_posterior_framing_audit.json"]),
        ("framework_scope_audit", [python, script("framework_scope_audit.py"), "--output", "/tmp/framework_scope_audit.json"]),
        ("field_dispersion_template", [python, script("validate_field_dispersion_input.py"), "--template", "field_dispersion_input.template.csv"]),
        ("figure_integrity_audit", [python, script("figure_integrity_audit.py"), "--output", "/tmp/figure_integrity_audit.json"]),
        ("figure_preflight_audit", [python, script("figure_preflight_audit.py"), "--output", "/tmp/figure_preflight_audit.json"]),
        ("publication_unit_audit", [python, script("publication_unit_audit.py"), "--output", "/tmp/publication_unit_audit.json"]),
        ("posterior_predictive_check", [python, script("posterior_predictive_check.py"), "--output", "/tmp/posterior_predictive_check.json"]),
        ("observation_noise_sensitivity", [python, script("observation_noise_sensitivity.py"), "--output", "/tmp/observation_noise_sensitivity.json"]),
        ("calibration_split_sensitivity", [python, script("calibration_split_sensitivity.py"), "--output", "/tmp/calibration_split_sensitivity.json"]),
        ("archive_staging_dry_run", [python, script("prepare_archive_bundles.py"), "--dry-run"]),
        ("latexmk", ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "agujournaltemplate.tex"]),
    ]
    if args.include_environment_validation:
        commands.insert(1, ("environment_spec_validation", [python, script("validate_environment_spec.py")]))

    gates = [run_command(name, command, root, args.timeout) for name, command in commands]
    final_values = root / args.final_values
    if final_values.exists():
        gates.append(
            run_command(
                "final_values",
                [python, script("validate_final_submission_values.py"), str(final_values)],
                root,
                args.timeout,
            )
        )
        gates.append(
            run_command(
                "final_sync",
                [python, script("validate_final_submission_sync.py"), str(final_values)],
                root,
                args.timeout,
            )
        )
    else:
        gates.append(skipped_gate("final_values", f"Missing {args.final_values}; final human values not filled."))
        gates.append(skipped_gate("final_sync", f"Missing {args.final_values}; propagation cannot be checked."))

    placeholders = placeholder_blockers(root)
    log_findings = latex_log_findings(root)
    archive_plan = read_json(root / "archive_bundle_plan.json") or {}
    archive_inventory = read_json(root / "archive_inventory.json") or {}
    publication_units = read_json(root / "publication_unit_audit.json") or {}
    machine_failures = [gate for gate in gates if gate["status"] == "FAIL"]
    skipped = [gate for gate in gates if gate["status"] == "SKIP"]

    human_blockers = []
    if not final_values.exists():
        human_blockers.append("Fill final_submission_values.json with author, DOI, license, reviewer, declaration, and scope values.")
    if placeholders:
        human_blockers.append("Replace manuscript placeholders: " + ", ".join(placeholders))
    if not args.scope_decision_confirmed:
        human_blockers.append("Confirm current synthetic, ak135, and uncalibrated Bayan Obo stress-test scope or add stronger field calibration.")

    draft_package_ok = not machine_failures and not log_findings
    external_submission_ready = draft_package_ok and not skipped and not human_blockers
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "draft_package_ok": draft_package_ok,
        "external_submission_ready": external_submission_ready,
        "machine_failures": [
            {"name": gate["name"], "command": command_name(gate.get("command", [])), "returncode": gate.get("returncode")}
            for gate in machine_failures
        ],
        "skipped_gates": [{"name": gate["name"], "reason": gate.get("reason", "")} for gate in skipped],
        "human_blockers": human_blockers,
        "latex_log_findings": log_findings,
        "archive": {
            "inventory_records": len(archive_inventory.get("files", [])) if isinstance(archive_inventory, dict) else 0,
            "required_missing": archive_inventory.get("required_missing", []) if isinstance(archive_inventory, dict) else [],
            "bundle_records": archive_plan.get("records_total"),
            "bundle_required_records": archive_plan.get("required_records_total"),
            "ready_for_archive": archive_plan.get("ready_for_archive"),
        },
        "publication_unit_audit": {
            "overall_status": publication_units.get("overall_status"),
            "word_count_estimate": publication_units.get("counts", {}).get("word_count_estimate") if isinstance(publication_units, dict) else None,
            "publication_units": publication_units.get("counts", {}).get("publication_units") if isinstance(publication_units, dict) else None,
            "publication_unit_margin": publication_units.get("counts", {}).get("publication_unit_margin") if isinstance(publication_units, dict) else None,
        },
        "gates": gates,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run final deterministic gates and write a JGR:SE submission report.")
    parser.add_argument("--root", default=".", help="Overleaf project root.")
    parser.add_argument("--script-dir", default="disp_inv_scripts", help="Directory containing manuscript support scripts, relative to --root.")
    parser.add_argument("--final-values", default="final_submission_values.json", help="Filled final values JSON.")
    parser.add_argument("--output", type=Path, help="Optional JSON report output path.")
    parser.add_argument("--timeout", type=int, default=180, help="Per-command timeout in seconds.")
    parser.add_argument(
        "--include-environment-validation",
        action="store_true",
        help="Also run validate_environment_spec.py, which may invoke conda.",
    )
    parser.add_argument(
        "--scope-decision-confirmed",
        action="store_true",
        help="Mark the scientific-scope decision as externally confirmed for this report.",
    )
    args = parser.parse_args()

    report = build_report(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["draft_package_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
