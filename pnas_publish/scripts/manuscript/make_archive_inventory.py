#!/usr/bin/env python3
"""Create a checksum inventory for the manuscript archive package."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent
OVERLEAF = SCRIPTS.parent
ROOT = OVERLEAF.parent


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def file_record(category: str, path: Path, required: bool = True, note: str = "") -> dict:
    exists = path.exists()
    record = {
        "category": category,
        "path": str(path),
        "relative_to_project_root": str(path.relative_to(ROOT)) if exists and path.is_relative_to(ROOT) else None,
        "required": required,
        "exists": exists,
        "note": note,
    }
    if exists:
        stat = path.stat()
        record.update(
            {
                "size_bytes": stat.st_size,
                "sha256": sha256_file(path),
            }
        )
    return record


def archive_files() -> list[tuple[str, Path, bool, str]]:
    software = [
        ("software", ROOT / "disp_inv_train.v1.2.py", True, "Neural posterior inversion training script with a depth-control sampling basis."),
        ("software", ROOT / "disp_inv_train.v1.1.py", True, "Dense-output pilot training script."),
        ("software", ROOT / "utils" / "generate_data.py", True, "Synthetic model and dispersion-generation utilities."),
        ("software", SCRIPTS / "make_paper_figures.py", True, "Manuscript figure and metric generation script."),
        ("software", SCRIPTS / "benchmark_inference.py", True, "Posterior-sampling timing benchmark."),
        ("software", SCRIPTS / "sampling_sensitivity.py", True, "Posterior sample-count and Euler-step sensitivity diagnostic."),
        ("software", SCRIPTS / "run_bayan_obo_field_test.py", True, "Bayan Obo field-data stress-test sampler and Figure 7 generator."),
        ("software", SCRIPTS / "posterior_predictive_check.py", True, "Posterior-predictive dispersion residual diagnostic for archived examples."),
        ("software", SCRIPTS / "observation_noise_sensitivity.py", True, "Synthetic observation-noise sensitivity diagnostic."),
        ("software", SCRIPTS / "calibration_split_sensitivity.py", True, "Multi-split posterior-temperature calibration stability diagnostic."),
        ("software", SCRIPTS / "figure_integrity_audit.py", True, "Pre-submission figure PDF integrity audit."),
        ("software", SCRIPTS / "figure_preflight_audit.py", True, "Production preflight audit for manuscript figure PDFs."),
        ("software", SCRIPTS / "publication_unit_audit.py", True, "AGU publication-unit length audit for the manuscript."),
        ("software", SCRIPTS / "make_environment_report.py", True, "Runtime-environment report generator."),
        ("software", OVERLEAF / "environment-repro.yml", True, "Conda environment specification for clean runtime reconstruction."),
        ("software", SCRIPTS / "validate_environment_spec.py", True, "Dry-run validator for the clean runtime environment specification."),
        ("software", SCRIPTS / "plan_archive_bundle.py", True, "File-level bundle planner for Zenodo archive records."),
        ("software", SCRIPTS / "prepare_archive_bundles.py", True, "Archive bundle staging script for DOI-upload directories."),
        ("software", SCRIPTS / "submission_qa.py", True, "Pre-submission QA checks."),
        ("software", SCRIPTS / "submission_gate_report.py", True, "Aggregated final submission gate report runner."),
        ("software", SCRIPTS / "apply_final_submission_values.py", True, "Final-submission values applier for manuscript, references, cover letter, metadata, and archive manifest."),
        ("software", SCRIPTS / "validate_final_submission_values.py", True, "Final-submission metadata validator for author, DOI, license, and reviewer values."),
        ("software", SCRIPTS / "validate_final_submission_sync.py", True, "Final-submission propagation checker for manuscript, references, cover letter, and metadata."),
        ("software", SCRIPTS / "validate_field_dispersion_input.py", True, "Field dispersion CSV validator and model-input tensorizer."),
        ("software", SCRIPTS / "validate_claim_evidence_matrix.py", True, "Claim-to-evidence matrix validator for manuscript claims and archived artifacts."),
        ("software", SCRIPTS / "central_posterior_framing_audit.py", True, "Text audit preserving neural posterior inversion as the central manuscript contribution."),
        ("software", SCRIPTS / "framework_scope_audit.py", True, "Text audit preserving posterior-surrogate scope boundaries and surface-wave demonstration limits."),
        ("software", SCRIPTS / "make_archive_inventory.py", True, "Archive checksum inventory generator."),
        ("software", ROOT / "ckpt" / "disp2struct_crf.v1.2_cp" / "best.pt", True, "Analyzed neural posterior checkpoint."),
        ("software", ROOT / "ckpt" / "disp2struct_crf.v1.1" / "best.pt", True, "Dense-output pilot checkpoint."),
        ("software", ROOT / "ckpt" / "disp2struct_crf.v1.2_cp" / "train.log", True, "Neural posterior training log used in Figure 3."),
    ]
    outputs = [
        ("data_output", OVERLEAF / "figures" / "metrics.json", True, "Numerical values reported in the manuscript."),
        ("data_output", OVERLEAF / "figures" / "inference_benchmark.json", True, "Posterior-sampling timing benchmark output."),
        ("data_output", OVERLEAF / "figures" / "sampling_sensitivity.json", True, "Sampling-configuration sensitivity diagnostic output."),
        ("data_output", OVERLEAF / "figures" / "posterior_predictive_check.json", True, "Posterior-predictive dispersion residual diagnostic for archived Figure 4 examples."),
        ("data_output", OVERLEAF / "figures" / "observation_noise_sensitivity.json", True, "Synthetic observation-noise sensitivity diagnostic output."),
        ("data_output", OVERLEAF / "figures" / "calibration_split_sensitivity.json", True, "Multi-split posterior-temperature calibration stability diagnostic output."),
        ("data_output", OVERLEAF / "figures" / "bayan_obo_field_summary.json", True, "Bayan Obo field-data stress-test summary metrics."),
        ("data_output", OVERLEAF / "figures" / "figure_integrity_audit.json", True, "Pre-submission figure PDF integrity audit output."),
        ("data_output", OVERLEAF / "figures" / "figure_preflight_audit.json", True, "Production preflight audit output for manuscript figure PDFs."),
        ("data_output", OVERLEAF / "publication_unit_audit.json", True, "AGU publication-unit length audit output."),
        ("data_output", OVERLEAF / "figures" / "posterior_figure_samples.npz", True, "Posterior samples and targets used directly in Figures 4--6."),
        ("data_output", OVERLEAF / "figures" / "bayan_obo_field_results.npz", True, "Bayan Obo field input tensors, posterior samples, and SURF96 comparison arrays used in Figure 7."),
        ("data_output", OVERLEAF / "environment_report.json", True, "Runtime-environment report for reproducing training, evaluation, and figures."),
        ("data_output", OVERLEAF / "environment_validation.json", True, "Dry-run validation report for the clean runtime environment specification."),
        ("data_output", OVERLEAF / "figures" / "fig01_workflow.pdf", True, "Manuscript Figure 1."),
        ("data_output", OVERLEAF / "figures" / "fig02_control_points.pdf", True, "Manuscript Figure 2."),
        ("data_output", OVERLEAF / "figures" / "fig03_training_history.pdf", True, "Manuscript Figure 3."),
        ("data_output", OVERLEAF / "figures" / "fig04_posterior_profiles.pdf", True, "Manuscript Figure 4."),
        ("data_output", OVERLEAF / "figures" / "fig05_posterior_density_vs.pdf", True, "Manuscript Figure 5."),
        ("data_output", OVERLEAF / "figures" / "fig06_posterior_uncertainty.pdf", True, "Manuscript Figure 6."),
        ("data_output", OVERLEAF / "figures" / "fig07_bayan_obo_field_test.pdf", True, "Manuscript Figure 7."),
        ("data_output", OVERLEAF / "figures" / "fig08_coverage.pdf", True, "Manuscript Figure 8."),
        ("data_output", OVERLEAF / "figures" / "fig09_dense_vs_control.pdf", True, "Manuscript Figure 9."),
    ]
    manuscript = [
        ("manuscript_source", OVERLEAF / "agujournaltemplate.tex", True, "AGU manuscript source."),
        ("manuscript_source", OVERLEAF / "references.bib", True, "Reference database."),
        ("manuscript_source", OVERLEAF / "agujournal2025.cls", True, "AGU 2025 LaTeX class."),
        ("manuscript_source", OVERLEAF / "tweaklist-git-moderncv-fixed.sty", True, "AGU template dependency."),
        ("manuscript_source", OVERLEAF / "wiley-macros.tex", True, "AGU template dependency."),
        ("manuscript_source", OVERLEAF / "agu-logo-small.pdf", True, "AGU template dependency."),
        ("manuscript_source", OVERLEAF / "agu-logo-large.pdf", True, "AGU template dependency."),
    ]
    support = [
        ("submission_support", OVERLEAF / "archive_manifest_zenodo.md", True, "Archive packaging plan."),
        ("submission_support", OVERLEAF / "archive_readme_zenodo.md", True, "Archive README for public Zenodo/AGU Open Research records."),
        ("submission_support", OVERLEAF / "agu_guidance_snapshot.md", True, "Official AGU/JGR guidance snapshot mapped to manuscript evidence."),
        ("submission_support", OVERLEAF / "zenodo_metadata_templates.json", True, "Machine-checkable Zenodo metadata worksheet for public archive records."),
        ("submission_support", OVERLEAF / "claim_evidence_matrix.json", True, "Machine-checkable claim-to-evidence matrix for manuscript claims and supporting artifacts."),
        ("submission_support", OVERLEAF / "editorial_scope_brief.md", True, "Pre-submission editorial scope and reviewer-risk brief."),
        ("submission_support", OVERLEAF / "reviewer_response_playbook.md", True, "Pre-submission response playbook for likely JGR:SE reviewer concerns."),
        ("submission_support", OVERLEAF / "reproducibility_notes.md", True, "Reproducibility notes."),
        ("submission_support", OVERLEAF / "jgrse_submission_metadata.md", True, "Submission form metadata packet."),
        ("submission_support", OVERLEAF / "jgrse_submission_readiness.md", True, "Submission readiness checklist."),
        ("submission_support", OVERLEAF / "reference_integrity_audit.json", True, "Reference-integrity audit for manuscript bibliography."),
        ("submission_support", OVERLEAF / "jgrse_human_input_packet.md", True, "Human-input completion packet for final submission metadata."),
        ("submission_support", OVERLEAF / "final_submission_values.template.json", True, "Structured template for final author, archive, reviewer, and scope values."),
        ("submission_support", OVERLEAF / "field_dispersion_input.template.csv", True, "Template for optional field dispersion validation input."),
        ("submission_support", OVERLEAF / "field_calibration_upgrade_plan.md", True, "Pre-submission plan for field-data and posterior-calibration upgrades."),
        ("submission_support", OVERLEAF / "internal_peer_review_jgrse.md", True, "Internal review risk register."),
        ("submission_support", OVERLEAF / "cover_letter_jgrse.md", True, "Draft cover letter."),
    ]
    optional = [
        ("software_optional", ROOT / "dispnet.v3.0.py", False, "Original model-design reference if present in this repository."),
        ("software_optional", Path("/Users/yuziye/machinelearning/disp/dispnet.v3.0.py"), False, "External model-design reference provided during development."),
    ]
    return software + outputs + manuscript + support + optional


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a checksum inventory for software/data archiving.")
    parser.add_argument("--output", type=Path, default=OVERLEAF / "archive_inventory.json")
    args = parser.parse_args()

    files = [file_record(*item) for item in archive_files()]
    required_missing = [item["path"] for item in files if item["required"] and not item["exists"]]
    inventory = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Checksum inventory for Zenodo/AGU Open Research archiving of the amortized posterior sampling manuscript.",
        "project_root": str(ROOT),
        "overleaf_project": str(OVERLEAF),
        "python": {
            "version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "packages": {
            "torch": package_version("torch"),
            "numpy": package_version("numpy"),
            "matplotlib": package_version("matplotlib"),
            "obspy": package_version("obspy"),
            "disba": package_version("disba"),
            "scipy": package_version("scipy"),
            "Pillow": package_version("Pillow"),
        },
        "archive_records": {
            "software_record": "Code, checkpoints, training logs, QA, figure-generation, benchmark, and inventory scripts.",
            "data_output_record": "Generated figures, metrics, benchmark outputs, and reproducible evaluation recipe.",
        },
        "files": files,
        "required_missing": required_missing,
    }
    args.output.write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "files": len(files), "required_missing": required_missing}, indent=2))
    return 1 if required_missing else 0


if __name__ == "__main__":
    sys.exit(main())
