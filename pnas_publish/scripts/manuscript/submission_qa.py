#!/usr/bin/env python3
"""Lightweight pre-submission checks for the AGU manuscript draft.

The script intentionally checks deterministic manuscript hygiene rather than
scientific validity. Placeholder author/DOI fields are warnings by default so
the draft can still compile while those human inputs are pending.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Check:
    name: str
    status: str
    detail: str


def strip_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        cut = None
        for idx, char in enumerate(line):
            if char == "%" and (idx == 0 or line[idx - 1] != "\\"):
                cut = idx
                break
        lines.append(line if cut is None else line[:cut])
    return "\n".join(lines)


def extract_env(text: str, env: str) -> str:
    match = re.search(
        rf"\\begin\{{{re.escape(env)}\}}(?P<body>.*?)\\end\{{{re.escape(env)}\}}",
        text,
        flags=re.S,
    )
    return match.group("body").strip() if match else ""


def remove_env(text: str, env: str) -> str:
    return re.sub(
        rf"\\begin\{{{re.escape(env)}\}}.*?\\end\{{{re.escape(env)}\}}",
        " ",
        text,
        flags=re.S,
    )


def brace_groups_after_command(text: str, command: str, count: int) -> list[str]:
    match = re.search(rf"\\{re.escape(command)}\b", text)
    if not match:
        return []
    groups: list[str] = []
    pos = match.end()
    while len(groups) < count:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text) or text[pos] != "{":
            return []
        depth = 0
        start = pos + 1
        pos += 1
        while pos < len(text):
            char = text[pos]
            if char == "{" and text[pos - 1] != "\\":
                depth += 1
            elif char == "}" and text[pos - 1] != "\\":
                if depth == 0:
                    groups.append(text[start:pos])
                    pos += 1
                    break
                depth -= 1
            pos += 1
        else:
            return []
    return groups


def extract_keypoints(text: str) -> list[str]:
    env_body = extract_env(text, "keypoints")
    if env_body:
        return [
            clean_latex_text(item)
            for item in re.findall(r"\\item\s+(.*)", env_body)
        ]
    return [clean_latex_text(item) for item in brace_groups_after_command(text, "keypoints", 3) if item.strip()]


def extract_star_section(text: str, heading: str) -> str:
    match = re.search(
        rf"\\section\*\{{{re.escape(heading)}\}}(?P<body>.*?)(?=\\section\*?\{{|\\begin\{{acknowledgments\}}|\\end\{{document\}}|\Z)",
        text,
        flags=re.S,
    )
    return match.group("body").strip() if match else ""


def markdown_section(markdown: str, heading: str) -> str:
    match = re.search(
        rf"^##\s+{re.escape(heading)}\s*$\n(?P<body>.*?)(?=^##\s+|\Z)",
        markdown,
        flags=re.M | re.S,
    )
    return match.group("body").strip() if match else ""


def markdown_numbered_items(markdown: str) -> list[str]:
    return [
        match.group(1).strip()
        for match in re.finditer(r"^\s*\d+\.\s+(.*?)\s*$", markdown, flags=re.M)
    ]


def split_keywords(text: str) -> list[str]:
    parts: list[str] = []
    for line in text.splitlines():
        line = re.sub(r"^\s*[-*]\s+", "", line)
        parts.extend(re.split(r";", line))
    cleaned = [clean_latex_text(part).strip() for part in parts]
    return [part for part in cleaned if part]


def normalize_submission_text(text: str) -> str:
    text = text.replace("\\%", "%")
    text = re.sub(r"\\([%&#_$])", r"\1", text)
    text = text.replace("$", "")
    text = clean_latex_text(text)
    text = text.replace("–", "-").replace("—", "-").replace("−", "-")
    text = text.replace("--", "-")
    text = re.sub(r"\s+([,.;:%])", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().casefold()


def normalized_sequence(items: list[str]) -> list[str]:
    return [normalize_submission_text(item) for item in items]


def latex_to_words(text: str) -> list[str]:
    text = re.sub(r"\$.*?\$", " ", text, flags=re.S)
    text = re.sub(r"\\cite[A-Za-z*]*(?:\[[^\]]*\])*\{[^}]*\}", " ", text)
    text = re.sub(r"\\(?:texttt|textbf|emph|section\*?|subsection\*?)\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?", " ", text)
    text = re.sub(r"[{}_^~$&]", " ", text)
    text = text.replace("--", " ").replace("\\%", " percent ")
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text)


def clean_latex_text(text: str) -> str:
    text = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[A-Za-z]+\*?", "", text)
    text = re.sub(r"[{}]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def bib_keys(bib_text: str) -> set[str]:
    return set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib_text))


def cited_keys(tex: str) -> set[str]:
    keys: set[str] = set()
    for match in re.finditer(r"\\cite[A-Za-z*]*(?:\[[^\]]*\])*\{([^}]+)\}", tex):
        keys.update(k.strip() for k in match.group(1).split(",") if k.strip())
    return keys


def includegraphics_paths(tex: str) -> list[str]:
    return re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)


def label_names(tex: str) -> set[str]:
    return set(re.findall(r"\\label\{([^}]+)\}", tex))


def ref_names(tex: str) -> set[str]:
    refs: set[str] = set()
    for command in ("ref", "pageref", "autoref", "eqref"):
        refs.update(re.findall(rf"\\{command}\{{([^}}]+)\}}", tex))
    return refs


def add(checks: list[Check], name: str, ok: bool, detail: str, warn: bool = False) -> None:
    status = "PASS" if ok else ("WARN" if warn else "FAIL")
    checks.append(Check(name, status, detail))


def rounded(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_checks(args: argparse.Namespace) -> list[Check]:
    base = Path(args.root)
    manuscript = base / args.manuscript
    bib = base / args.bib
    cover_letter_path = base / args.cover_letter
    reference_audit_path = base / args.reference_audit
    figure_dir = base / args.figures
    figure_generator_path = base / args.figure_generator
    figure_integrity_path = base / args.figure_integrity
    figure_preflight_path = base / args.figure_preflight
    publication_unit_path = base / args.publication_unit_audit
    metrics_path = base / args.metrics
    benchmark_path = base / args.benchmark
    sensitivity_path = base / args.sensitivity
    posterior_predictive_path = base / args.posterior_predictive
    observation_noise_path = base / args.observation_noise
    calibration_split_path = base / args.calibration_split
    figure_samples_path = base / args.figure_samples
    environment_path = base / args.environment
    environment_spec_path = base / args.environment_spec
    environment_validation_path = base / args.environment_validation
    inventory_path = base / args.inventory
    bundle_plan_path = base / args.bundle_plan
    bundle_prep_path = base / args.bundle_prep
    gate_report_path = base / args.gate_report
    archive_readme_path = base / args.archive_readme
    reproducibility_notes_path = base / args.reproducibility_notes
    zenodo_metadata_path = base / args.zenodo_metadata
    claim_evidence_matrix_path = base / args.claim_evidence_matrix
    claim_evidence_validator_path = base / args.claim_evidence_validator
    central_framing_audit_path = base / args.central_framing_audit
    framework_scope_audit_path = base / args.framework_scope_audit
    editorial_scope_path = base / args.editorial_scope_brief
    reviewer_response_playbook_path = base / args.reviewer_response_playbook
    agu_guidance_snapshot_path = base / args.agu_guidance_snapshot
    human_inputs_path = base / args.human_inputs
    final_values_template_path = base / args.final_values_template
    final_values_applier_path = base / args.final_values_applier
    final_values_validator_path = base / args.final_values_validator
    final_sync_validator_path = base / args.final_sync_validator
    field_calibration_plan_path = base / args.field_calibration_plan
    field_dispersion_template_path = base / args.field_dispersion_template
    field_dispersion_validator_path = base / args.field_dispersion_validator

    tex = strip_comments(manuscript.read_text(encoding="utf-8"))
    manuscript_sha256 = sha256_file(manuscript)
    tex_casefold = tex.casefold()
    table_count = len(re.findall(r"\\begin\{table\}", tex))
    bib_text = bib.read_text(encoding="utf-8")
    cover_letter_text = cover_letter_path.read_text(encoding="utf-8") if cover_letter_path.exists() else ""
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    reference_audit = None
    reference_audit_error = ""
    if reference_audit_path.exists():
        try:
            reference_audit = json.loads(reference_audit_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            reference_audit_error = str(exc)
    figure_integrity = None
    figure_integrity_error = ""
    if figure_integrity_path.exists():
        try:
            figure_integrity = json.loads(figure_integrity_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            figure_integrity_error = str(exc)
    figure_preflight = None
    figure_preflight_error = ""
    if figure_preflight_path.exists():
        try:
            figure_preflight = json.loads(figure_preflight_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            figure_preflight_error = str(exc)
    publication_unit = None
    publication_unit_error = ""
    if publication_unit_path.exists():
        try:
            publication_unit = json.loads(publication_unit_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            publication_unit_error = str(exc)
    benchmark = None
    benchmark_error = ""
    if benchmark_path.exists():
        try:
            benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            benchmark_error = str(exc)
    sensitivity = None
    sensitivity_error = ""
    if sensitivity_path.exists():
        try:
            sensitivity = json.loads(sensitivity_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            sensitivity_error = str(exc)
    posterior_predictive = None
    posterior_predictive_error = ""
    if posterior_predictive_path.exists():
        try:
            posterior_predictive = json.loads(posterior_predictive_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            posterior_predictive_error = str(exc)
    observation_noise = None
    observation_noise_error = ""
    if observation_noise_path.exists():
        try:
            observation_noise = json.loads(observation_noise_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            observation_noise_error = str(exc)
    calibration_split = None
    calibration_split_error = ""
    if calibration_split_path.exists():
        try:
            calibration_split = json.loads(calibration_split_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            calibration_split_error = str(exc)
    environment = None
    environment_error = ""
    if environment_path.exists():
        try:
            environment = json.loads(environment_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            environment_error = str(exc)
    inventory = None
    inventory_error = ""
    if inventory_path.exists():
        try:
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            inventory_error = str(exc)
    zenodo_metadata = None
    zenodo_metadata_error = ""
    if zenodo_metadata_path.exists():
        try:
            zenodo_metadata = json.loads(zenodo_metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            zenodo_metadata_error = str(exc)
    claim_evidence_matrix = None
    claim_evidence_matrix_error = ""
    if claim_evidence_matrix_path.exists():
        try:
            claim_evidence_matrix = json.loads(claim_evidence_matrix_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            claim_evidence_matrix_error = str(exc)
    editorial_scope_text = editorial_scope_path.read_text(encoding="utf-8") if editorial_scope_path.exists() else ""

    checks: list[Check] = []

    class_match = re.search(r"\\documentclass(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    document_class = class_match.group(1) if class_match else ""
    add(
        checks,
        "AGU LaTeX class",
        document_class == "agujournal2025",
        f"Using {document_class or 'no detected document class'}; target is agujournal2025.",
    )
    template_files = [
        "agujournal2025.cls",
        "tweaklist-git-moderncv-fixed.sty",
        "wiley-macros.tex",
        "agu-logo-small.pdf",
        "agu-logo-large.pdf",
    ]
    missing_template_files = [name for name in template_files if not (base / name).exists()]
    add(
        checks,
        "AGU 2025 template files",
        not missing_template_files,
        "Missing files: " + ", ".join(missing_template_files) if missing_template_files else "All AGU 2025 template dependency files are present.",
    )

    abstract = remove_env(extract_env(tex, "abstract"), "plainlanguagesummary")
    abstract_words = latex_to_words(abstract)
    add(
        checks,
        "Abstract length",
        0 < len(abstract_words) <= 250,
        f"{len(abstract_words)} words; AGU limit is 250.",
    )

    pls = extract_env(tex, "plainlanguagesummary")
    if not pls:
        pls_match = re.search(
            r"\\section\*\{Plain Language Summary\}(?P<body>.*?)(?=\\section\*?\{)",
            tex,
            flags=re.S,
        )
        pls = pls_match.group("body") if pls_match else ""
    pls_words = latex_to_words(pls)
    add(
        checks,
        "Plain Language Summary length",
        0 < len(pls_words) <= 200,
        f"{len(pls_words)} words; JGR:SE limit is 200.",
    )

    keypoints = extract_keypoints(tex)
    keypoint_ok = len(keypoints) == 3 and all(len(k) <= 140 for k in keypoints)
    add(
        checks,
        "Key Points",
        keypoint_ok,
        "; ".join(f"{len(k)} chars" for k in keypoints) or "No key points found.",
    )
    abbrev_hits = [k for k in keypoints if re.search(r"\b(Vp|Vs|rho|CRF|PDF|ODE)\b", k)]
    add(
        checks,
        "Key Point wording",
        not abbrev_hits,
        "No obvious abbreviations found." if not abbrev_hits else "Potential abbreviations: " + " | ".join(abbrev_hits),
        warn=True,
    )

    cited = cited_keys(tex)
    available = bib_keys(bib_text)
    missing_cites = sorted(cited - available)
    unused_bib = sorted(available - cited)
    add(checks, "Citation closure", not missing_cites, "Missing BibTeX keys: " + ", ".join(missing_cites) if missing_cites else f"{len(cited)} cited keys resolved.")
    add(checks, "Unused bibliography entries", not unused_bib, "Unused entries: " + ", ".join(unused_bib) if unused_bib else "All BibTeX entries are cited.", warn=True)
    audit_records = reference_audit.get("records", []) if isinstance(reference_audit, dict) else []
    audit_by_key = {
        item.get("key"): item
        for item in audit_records
        if isinstance(item, dict) and isinstance(item.get("key"), str)
    }
    missing_audit_keys = sorted(available - set(audit_by_key))
    extra_audit_keys = sorted(set(audit_by_key) - available)
    non_verified_audit = sorted(
        key
        for key, item in audit_by_key.items()
        if item.get("status") != "VERIFIED" or item.get("citation_context_status") != "APPROPRIATE"
    )
    incomplete_audit = sorted(
        key
        for key, item in audit_by_key.items()
        if not item.get("query") or not item.get("primary_source_url") or not isinstance(item.get("verified_fields"), dict)
    )
    reference_audit_ok = (
        reference_audit is not None
        and reference_audit.get("overall_status") == "PASS"
        and not missing_audit_keys
        and not extra_audit_keys
        and not non_verified_audit
        and not incomplete_audit
    )
    reference_audit_details = []
    if missing_audit_keys:
        reference_audit_details.append("missing audit keys: " + ", ".join(missing_audit_keys))
    if extra_audit_keys:
        reference_audit_details.append("extra audit keys: " + ", ".join(extra_audit_keys))
    if non_verified_audit:
        reference_audit_details.append("non-verified records: " + ", ".join(non_verified_audit))
    if incomplete_audit:
        reference_audit_details.append("incomplete records: " + ", ".join(incomplete_audit))
    add(
        checks,
        "Reference integrity audit",
        reference_audit_ok,
        reference_audit_error if reference_audit_error else (
            f"{args.reference_audit} verifies all {len(available)} BibTeX entries with external sources."
            if reference_audit_ok
            else (
                "; ".join(reference_audit_details)
                if reference_audit is not None
                else f"Missing or unreadable {args.reference_audit}."
            )
        ),
    )

    labels = label_names(tex)
    refs = ref_names(tex)
    missing_refs = sorted(refs - labels)
    unused_labels = sorted(labels - refs)
    add(checks, "Reference closure", not missing_refs, "Missing labels: " + ", ".join(missing_refs) if missing_refs else f"{len(refs)} references resolved.")
    add(checks, "Unused labels", not unused_labels, "Unused labels: " + ", ".join(unused_labels) if unused_labels else "All labels are referenced.", warn=True)

    included = includegraphics_paths(tex)
    included_files = {Path(p).name for p in included}
    expected_files = {p.name for p in figure_dir.glob("*.pdf")}
    missing_figures = [p for p in included if not (base / p).exists()]
    uncited_figures = sorted(expected_files - included_files)
    add(checks, "Included figure files exist", not missing_figures, "Missing files: " + ", ".join(missing_figures) if missing_figures else f"{len(included)} included figures exist.")
    add(checks, "Figure count", len(included) == 8, f"{len(included)} main-text figures included; target is 8.")
    add(checks, "Figure file closure", not uncited_figures, "Uncited figure PDFs: " + ", ".join(uncited_figures) if uncited_figures else "All figure PDFs are included.", warn=True)
    figure_integrity_records = figure_integrity.get("records", []) if isinstance(figure_integrity, dict) else []
    audited_figure_paths = {
        item.get("path")
        for item in figure_integrity_records
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    included_path_set = set(included)
    failing_figures = sorted(
        item.get("path", "<unknown>")
        for item in figure_integrity_records
        if isinstance(item, dict) and item.get("status") != "PASS"
    )
    missing_figure_audits = sorted(included_path_set - audited_figure_paths)
    extra_figure_audits = sorted(audited_figure_paths - included_path_set)
    figure_integrity_ok = (
        figure_integrity is not None
        and figure_integrity.get("overall_status") == "PASS"
        and figure_integrity.get("figure_count") == len(included)
        and not failing_figures
        and not missing_figure_audits
        and not extra_figure_audits
    )
    figure_integrity_details = []
    if failing_figures:
        figure_integrity_details.append("failing figures: " + ", ".join(failing_figures))
    if missing_figure_audits:
        figure_integrity_details.append("missing audit records: " + ", ".join(missing_figure_audits))
    if extra_figure_audits:
        figure_integrity_details.append("extra audit records: " + ", ".join(extra_figure_audits))
    add(
        checks,
        "Figure integrity audit",
        figure_integrity_ok,
        figure_integrity_error if figure_integrity_error else (
            f"{args.figure_integrity} verifies {len(included)} included figure PDFs as readable, single-page, and nonblank."
            if figure_integrity_ok
            else (
                "; ".join(figure_integrity_details)
                if figure_integrity is not None
                else f"Missing or unreadable {args.figure_integrity}."
            )
        ),
    )
    figure_preflight_records = figure_preflight.get("records", []) if isinstance(figure_preflight, dict) else []
    preflight_figure_paths = {
        item.get("path")
        for item in figure_preflight_records
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    failing_preflight_figures = sorted(
        item.get("path", "<unknown>")
        for item in figure_preflight_records
        if isinstance(item, dict) and item.get("status") != "PASS"
    )
    missing_preflight_audits = sorted(included_path_set - preflight_figure_paths)
    extra_preflight_audits = sorted(preflight_figure_paths - included_path_set)
    failed_preflight_checks = sorted(
        {
            check_name
            for item in figure_preflight_records
            if isinstance(item, dict)
            for check_name, ok in item.get("checks", {}).items()
            if ok is False
        }
    )
    figure_preflight_ok = (
        figure_preflight is not None
        and figure_preflight.get("overall_status") == "PASS"
        and figure_preflight.get("figure_count") == len(included)
        and not failing_preflight_figures
        and not missing_preflight_audits
        and not extra_preflight_audits
        and not failed_preflight_checks
    )
    figure_preflight_details = []
    if failing_preflight_figures:
        figure_preflight_details.append("failing figures: " + ", ".join(failing_preflight_figures))
    if missing_preflight_audits:
        figure_preflight_details.append("missing audit records: " + ", ".join(missing_preflight_audits))
    if extra_preflight_audits:
        figure_preflight_details.append("extra audit records: " + ", ".join(extra_preflight_audits))
    if failed_preflight_checks:
        figure_preflight_details.append("failed checks: " + ", ".join(failed_preflight_checks))
    add(
        checks,
        "Figure production preflight",
        figure_preflight_ok,
        figure_preflight_error if figure_preflight_error else (
            f"{args.figure_preflight} verifies embedded fonts, page sizes, PDF metadata, and image budgets for {len(included)} included figures."
            if figure_preflight_ok
            else (
                "; ".join(figure_preflight_details)
                if figure_preflight is not None
                else f"Missing or unreadable {args.figure_preflight}."
            )
        ),
    )
    workflow_script_text = (
        figure_generator_path.read_text(encoding="utf-8")
        if figure_generator_path.exists()
        else ""
    )
    required_workflow_caption_terms = [
        "Prior-support boundary workflow",
        "bounded synthetic prior",
        "Direct inversion maps observed dispersion",
        "Indirect inversion trains a neural forward surrogate",
        "in-prior, boundary, and out-of-prior test sets",
        "regional structural bias",
    ]
    missing_workflow_caption_terms = [
        term for term in required_workflow_caption_terms if term.casefold() not in tex_casefold
    ]
    required_workflow_script_terms = [
        "Bounded synthetic prior and simulator",
        "Direct learned inversion",
        "Indirect learned inversion",
        "Prior-boundary tests",
        "Local 1-D inversions assembled into regional structure",
        "prior collapse",
        "Forward surrogate",
    ]
    missing_workflow_script_terms = [
        term for term in required_workflow_script_terms if term not in workflow_script_text
    ]
    add(
        checks,
        "Workflow figure framing",
        (
            "fig01_workflow.pdf" in included_files
            and figure_generator_path.exists()
            and not missing_workflow_caption_terms
            and not missing_workflow_script_terms
        ),
        (
            "Figure 1 caption and generator preserve the prior-boundary direct/indirect workflow framing."
            if (
                "fig01_workflow.pdf" in included_files
                and figure_generator_path.exists()
                and not missing_workflow_caption_terms
                and not missing_workflow_script_terms
            )
            else "Problems: "
            + "; ".join(
                item
                for item in [
                    "fig01_workflow.pdf not included" if "fig01_workflow.pdf" not in included_files else "",
                    f"missing {args.figure_generator}" if not figure_generator_path.exists() else "",
                    "missing caption terms: " + ", ".join(missing_workflow_caption_terms)
                    if missing_workflow_caption_terms
                    else "",
                    "missing generator terms: " + ", ".join(missing_workflow_script_terms)
                    if missing_workflow_script_terms
                    else "",
                ]
                if item
            )
        ),
    )
    publication_counts = publication_unit.get("counts", {}) if isinstance(publication_unit, dict) else {}
    publication_thresholds = publication_unit.get("thresholds", {}) if isinstance(publication_unit, dict) else {}
    publication_checks = publication_unit.get("checks", {}) if isinstance(publication_unit, dict) else {}
    publication_units = publication_counts.get("publication_units")
    publication_word_count = publication_counts.get("word_count_estimate")
    publication_figures = publication_counts.get("figures")
    publication_tables = publication_counts.get("tables")
    publication_limit = publication_thresholds.get("max_publication_units", args.max_publication_units)
    publication_failures = []
    if not isinstance(publication_unit, dict):
        publication_failures.append(f"missing or unreadable {args.publication_unit_audit}")
    else:
        if publication_unit.get("overall_status") != "PASS":
            publication_failures.append("overall_status is not PASS")
        if publication_unit.get("manuscript_sha256") != manuscript_sha256:
            publication_failures.append("manuscript SHA256 is stale")
        if publication_figures != len(included):
            publication_failures.append(f"figure count {publication_figures} != {len(included)}")
        if publication_tables != table_count:
            publication_failures.append(f"table count {publication_tables} != {table_count}")
        if not isinstance(publication_units, (int, float)) or float(publication_units) > float(publication_limit):
            publication_failures.append(f"publication units {publication_units} exceed limit {publication_limit}")
        if not isinstance(publication_word_count, int) or publication_word_count <= 0:
            publication_failures.append("word_count_estimate is not positive")
        if isinstance(publication_checks, dict) and any(value is False for value in publication_checks.values()):
            publication_failures.append("one or more publication-unit component checks failed")
    add(
        checks,
        "Publication-unit audit",
        not publication_failures,
        publication_unit_error if publication_unit_error else (
            f"{args.publication_unit_audit} loaded; {publication_word_count} words, {publication_figures} figures, {publication_tables} tables, {publication_units:.3f} PU; limit {publication_limit:.1f}."
            if not publication_failures
            else "Problems: " + "; ".join(publication_failures)
        ),
    )
    add(
        checks,
        "Inference benchmark artifact",
        benchmark is not None,
        benchmark_error if benchmark_error else (
            f"{args.benchmark} loaded; median {benchmark['median_seconds']:.2f} s."
            if benchmark is not None
            else f"Missing or unreadable {args.benchmark}."
        ),
    )
    add(
        checks,
        "Sampling sensitivity artifact",
        sensitivity is not None,
        sensitivity_error if sensitivity_error else (
            f"{args.sensitivity} loaded; {len(sensitivity['results'])} configurations."
            if sensitivity is not None
            else f"Missing or unreadable {args.sensitivity}."
        ),
    )
    posterior_predictive_cases = posterior_predictive.get("cases", {}) if isinstance(posterior_predictive, dict) else {}
    posterior_predictive_failures = []
    if isinstance(posterior_predictive_cases, dict):
        for case_name in ("synthetic", "ak135"):
            case = posterior_predictive_cases.get(case_name, {})
            if not isinstance(case, dict):
                posterior_predictive_failures.append(f"missing {case_name}")
                continue
            if case.get("failed_sample_indices"):
                posterior_predictive_failures.append(f"{case_name} failed samples")
            if case.get("posterior_samples_forward_modeled", 0) <= 0:
                posterior_predictive_failures.append(f"{case_name} no forward-modeled samples")
    add(
        checks,
        "Posterior-predictive artifact",
        posterior_predictive is not None and not posterior_predictive_failures,
        posterior_predictive_error if posterior_predictive_error else (
            f"{args.posterior_predictive} loaded; synthetic and ak135 posterior samples forward-modeled with no failures."
            if posterior_predictive is not None and not posterior_predictive_failures
            else (
                "Failures: " + ", ".join(posterior_predictive_failures)
                if posterior_predictive_failures
                else f"Missing or unreadable {args.posterior_predictive}."
            )
        ),
    )
    observation_noise_levels = []
    if isinstance(observation_noise, dict) and isinstance(observation_noise.get("results"), list):
        observation_noise_levels = [item.get("noise_sigma_km_s") for item in observation_noise["results"] if isinstance(item, dict)]
    expected_noise_levels = {0.0, 0.02, 0.05, 0.1}
    found_noise_levels = {float(level) for level in observation_noise_levels if isinstance(level, (int, float))}
    add(
        checks,
        "Observation-noise sensitivity artifact",
        observation_noise is not None and expected_noise_levels.issubset(found_noise_levels),
        observation_noise_error if observation_noise_error else (
            f"{args.observation_noise} loaded with expected 0.00/0.02/0.05/0.10 km/s noise levels."
            if observation_noise is not None and expected_noise_levels.issubset(found_noise_levels)
            else (
                "Missing noise levels: " + ", ".join(f"{level:.2f}" for level in sorted(expected_noise_levels - found_noise_levels))
                if observation_noise is not None
                else f"Missing or unreadable {args.observation_noise}."
            )
        ),
    )
    calibration_splits = calibration_split.get("splits", []) if isinstance(calibration_split, dict) else []
    calibration_summary = calibration_split.get("summary", {}) if isinstance(calibration_split, dict) else {}
    add(
        checks,
        "Calibration split sensitivity artifact",
        (
            calibration_split is not None
            and isinstance(calibration_splits, list)
            and len(calibration_splits) >= 5
            and isinstance(calibration_summary.get("temperature_scale"), dict)
            and isinstance(calibration_summary.get("scaled_test_coverage_16_84_mean"), dict)
        ),
        calibration_split_error if calibration_split_error else (
            f"{args.calibration_split} loaded with {len(calibration_splits)} calibration/test splits."
            if calibration_split is not None
            else f"Missing or unreadable {args.calibration_split}."
        ),
    )
    required_npz_keys = {
        "depth_km.npy",
        "channel_names.npy",
        "channel_units.npy",
        "synthetic_target.npy",
        "synthetic_dispersion.npy",
        "synthetic_mask.npy",
        "synthetic_posterior_samples.npy",
        "ak135_target.npy",
        "ak135_dispersion.npy",
        "ak135_mask.npy",
        "ak135_posterior_samples.npy",
        "posterior_samples.npy",
        "sampling_steps.npy",
        "evaluation_seed.npy",
        "checkpoint_epoch.npy",
        "checkpoint_global_step.npy",
    }
    figure_sample_error = ""
    figure_sample_keys: set[str] = set()
    if figure_samples_path.exists():
        try:
            with zipfile.ZipFile(figure_samples_path) as handle:
                figure_sample_keys = set(handle.namelist())
        except zipfile.BadZipFile as exc:
            figure_sample_error = str(exc)
    missing_npz_keys = sorted(required_npz_keys - figure_sample_keys)
    add(
        checks,
        "Figure posterior-sample artifact",
        figure_samples_path.exists() and not figure_sample_error and not missing_npz_keys,
        (
            figure_sample_error
            if figure_sample_error
            else (
                f"{args.figure_samples} present with Figure 4--6 posterior samples and metadata."
                if figure_samples_path.exists() and not missing_npz_keys
                else (
                    "Missing NPZ keys: " + ", ".join(missing_npz_keys)
                    if figure_samples_path.exists()
                    else f"Missing {args.figure_samples}."
                )
            )
        ),
    )
    required_environment_missing = (
        environment.get("required_missing", []) if isinstance(environment, dict) else []
    )
    required_environment_import_failures = (
        environment.get("required_import_failures", []) if isinstance(environment, dict) else []
    )
    add(
        checks,
        "Environment report artifact",
        environment is not None and not required_environment_missing,
        environment_error if environment_error else (
            f"{args.environment} loaded; {len(environment.get('packages', []))} package records; no missing required packages."
            if environment is not None and not required_environment_missing
            else (
                "Required environment packages missing: " + ", ".join(required_environment_missing)
                if required_environment_missing
                else f"Missing or unreadable {args.environment}."
            )
        ),
    )
    add(
        checks,
        "Environment import smoke tests",
        not required_environment_import_failures,
        "Required packages imported successfully."
        if not required_environment_import_failures
        else "Import failures: " + ", ".join(required_environment_import_failures),
        warn=True,
    )
    environment_spec_text = environment_spec_path.read_text(encoding="utf-8") if environment_spec_path.exists() else ""
    required_spec_terms = ["python=3.11", "torch==", "numpy=", "matplotlib=", "disba==", "obspy="]
    missing_spec_terms = [term for term in required_spec_terms if term not in environment_spec_text]
    add(
        checks,
        "Environment specification",
        environment_spec_path.exists() and not missing_spec_terms,
        (
            f"{args.environment_spec} present with core runtime terms."
            if environment_spec_path.exists() and not missing_spec_terms
            else (
                "Missing terms: " + ", ".join(missing_spec_terms)
                if environment_spec_path.exists()
                else f"Missing {args.environment_spec}."
            )
        ),
    )
    environment_validation = None
    environment_validation_error = ""
    if environment_validation_path.exists():
        try:
            environment_validation = json.loads(environment_validation_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            environment_validation_error = str(exc)
    environment_validation_failures = []
    if isinstance(environment_validation, dict):
        if not environment_validation.get("overall_ok"):
            if not environment_validation.get("conda_dry_run", {}).get("ok"):
                environment_validation_failures.append("conda dry-run failed")
            if not environment_validation.get("conda_dry_run", {}).get("summary", {}).get("parse_ok"):
                environment_validation_failures.append("conda dry-run JSON parse failed")
            missing_terms = environment_validation.get("missing_terms", [])
            if missing_terms:
                environment_validation_failures.append("missing terms: " + ", ".join(missing_terms))
            pip_unavailable = environment_validation.get("pip_unavailable", [])
            if pip_unavailable:
                environment_validation_failures.append("pip unavailable: " + ", ".join(pip_unavailable))
    add(
        checks,
        "Environment spec validation",
        isinstance(environment_validation, dict) and bool(environment_validation.get("overall_ok")),
        environment_validation_error if environment_validation_error else (
            f"{args.environment_validation} reports conda dry-run success and pip pins available."
            if isinstance(environment_validation, dict) and environment_validation.get("overall_ok")
            else (
                "Validation failures: " + "; ".join(environment_validation_failures or ["unknown"])
                if isinstance(environment_validation, dict)
                else f"Missing or unreadable {args.environment_validation}."
            )
        ),
    )
    inventory_missing = inventory.get("required_missing", []) if isinstance(inventory, dict) else []
    inventory_records = len(inventory.get("files", [])) if isinstance(inventory, dict) else 0
    add(
        checks,
        "Archive inventory artifact",
        inventory is not None and not inventory_missing and inventory_records >= args.min_inventory_records,
        inventory_error if inventory_error else (
            f"{args.inventory} loaded; {inventory_records} records; no required missing files."
            if inventory is not None and not inventory_missing and inventory_records >= args.min_inventory_records
            else (
                "Required missing files: " + ", ".join(inventory_missing)
                if inventory_missing
                else (
                    f"{args.inventory} has {inventory_records} records; expected at least {args.min_inventory_records}."
                    if inventory is not None
                    else f"Missing or unreadable {args.inventory}."
                )
            )
        ),
    )
    bundle_plan = None
    bundle_plan_error = ""
    if bundle_plan_path.exists():
        try:
            bundle_plan = json.loads(bundle_plan_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            bundle_plan_error = str(exc)
    bundle_ready = isinstance(bundle_plan, dict) and bundle_plan.get("ready_for_archive")
    bundle_records = bundle_plan.get("records_total", 0) if isinstance(bundle_plan, dict) else 0
    add(
        checks,
        "Archive bundle plan",
        bundle_ready and bundle_records >= args.min_inventory_records,
        bundle_plan_error if bundle_plan_error else (
            f"{args.bundle_plan} ready; {bundle_records} records assigned to archive bundles."
            if bundle_ready and bundle_records >= args.min_inventory_records
            else (
                f"{args.bundle_plan} has {bundle_records} records; expected at least {args.min_inventory_records}."
                if isinstance(bundle_plan, dict)
                else f"Missing or unreadable {args.bundle_plan}."
            )
        ),
    )
    archive_readme_text = archive_readme_path.read_text(encoding="utf-8") if archive_readme_path.exists() else ""
    required_archive_readme_sections = [
        "## Citation Status",
        "## License Status",
        "## File Map",
        "## Reproduction Commands",
        "## Evaluation Seeds and Sampling Settings",
        "## Checksum Verification",
        "## Known Scope Limits",
    ]
    missing_archive_readme_sections = [
        section for section in required_archive_readme_sections if section not in archive_readme_text
    ]
    add(
        checks,
        "Archive README packet",
        archive_readme_path.exists() and not missing_archive_readme_sections,
        (
            f"{args.archive_readme} present with archive file map, reproduction, checksum, license, and scope sections."
            if archive_readme_path.exists() and not missing_archive_readme_sections
            else (
                "Missing sections: " + ", ".join(missing_archive_readme_sections)
                if archive_readme_path.exists()
                else f"Missing {args.archive_readme}."
            )
        ),
    )
    reproducibility_notes_text = reproducibility_notes_path.read_text(encoding="utf-8") if reproducibility_notes_path.exists() else ""
    required_reproducibility_terms = [
        "figure-production preflight",
        "publication-unit audit synchronization",
        "claim-evidence matrix coverage",
        "editorial scope brief coverage",
        "publication_unit_audit.json",
        "figures/figure_preflight_audit.json",
        "validate_claim_evidence_matrix.py",
        "central_posterior_framing_audit.py",
        "framework_scope_audit.py",
        "reviewer_response_playbook.md",
        "agu_guidance_snapshot.md",
        "editorial_scope_brief.md",
        "archive_inventory.json` records 80 archive file entries",
        "archive_bundle_plan.json` assigns 80 records",
    ]
    missing_reproducibility_terms = [
        term for term in required_reproducibility_terms if term not in reproducibility_notes_text
    ]
    stale_reproducibility_terms = [
        term
        for term in [
            "archive_inventory.json` records 64 archive file entries",
            "archive_bundle_plan.json` assigns 64 records",
            "assigns 64 records",
            "records 72 archive file entries",
            "assigns 72 records",
            "records 73 archive file entries",
            "assigns 73 records",
            "records 74 archive file entries",
            "assigns 74 records",
            "records 75 archive file entries",
            "assigns 75 records",
            "records 76 archive file entries",
            "assigns 76 records",
        ]
        if term in reproducibility_notes_text
    ]
    add(
        checks,
        "Reproducibility notes packet",
        reproducibility_notes_path.exists() and not missing_reproducibility_terms and not stale_reproducibility_terms,
        (
            f"{args.reproducibility_notes} present with current QA, figure-preflight, publication-unit, claim-evidence, editorial-scope, and archive-count terms."
            if reproducibility_notes_path.exists() and not missing_reproducibility_terms and not stale_reproducibility_terms
            else (
                "Problems: "
                + "; ".join(
                    item
                    for item in [
                        "missing terms: " + ", ".join(missing_reproducibility_terms) if missing_reproducibility_terms else "",
                        "stale terms: " + ", ".join(stale_reproducibility_terms) if stale_reproducibility_terms else "",
                    ]
                    if item
                )
                if reproducibility_notes_path.exists()
                else f"Missing {args.reproducibility_notes}."
            )
        ),
    )
    required_zenodo_top_keys = [
        "schema_version",
        "source_final_values_template",
        "public_records",
        "staged_support_bundles",
        "pre_minting_checks",
    ]
    required_zenodo_record_fields = [
        "planned_upload_type",
        "title",
        "version",
        "doi",
        "license",
        "creators",
        "description",
        "keywords",
        "related_identifiers",
        "expected_bundle",
        "required_file_categories",
        "scope_note",
    ]
    expected_zenodo_records = {
        "software_record": ("software_record", "software"),
        "data_output_record": ("data_output_record", "data_output"),
    }
    missing_zenodo_top_keys = [
        key
        for key in required_zenodo_top_keys
        if not isinstance(zenodo_metadata, dict) or key not in zenodo_metadata
    ]
    public_records = zenodo_metadata.get("public_records", {}) if isinstance(zenodo_metadata, dict) else {}
    missing_public_records = [
        name for name in expected_zenodo_records if not isinstance(public_records.get(name), dict)
    ]
    missing_zenodo_fields = []
    mismatched_zenodo_bundles = []
    mismatched_zenodo_categories = []
    for name, (expected_bundle, expected_category) in expected_zenodo_records.items():
        record = public_records.get(name, {}) if isinstance(public_records, dict) else {}
        if not isinstance(record, dict):
            continue
        for field in required_zenodo_record_fields:
            value = record.get(field)
            if value in (None, "", []):
                missing_zenodo_fields.append(f"{name}.{field}")
        if record.get("expected_bundle") != expected_bundle:
            mismatched_zenodo_bundles.append(f"{name}.expected_bundle")
        categories = record.get("required_file_categories", [])
        if not isinstance(categories, list) or expected_category not in categories:
            mismatched_zenodo_categories.append(f"{name}.required_file_categories")
    zenodo_metadata_ok = (
        isinstance(zenodo_metadata, dict)
        and not missing_zenodo_top_keys
        and not missing_public_records
        and not missing_zenodo_fields
        and not mismatched_zenodo_bundles
        and not mismatched_zenodo_categories
    )
    add(
        checks,
        "Zenodo metadata templates",
        zenodo_metadata_ok,
        zenodo_metadata_error if zenodo_metadata_error else (
            f"{args.zenodo_metadata} present with software/data record templates, bundle mappings, and pre-minting checks."
            if zenodo_metadata_ok
            else (
                "Missing or mismatched: "
                + ", ".join(
                    missing_zenodo_top_keys
                    + missing_public_records
                    + missing_zenodo_fields
                    + mismatched_zenodo_bundles
                    + mismatched_zenodo_categories
                )
                if zenodo_metadata_path.exists()
                else f"Missing {args.zenodo_metadata}."
            )
        ),
    )
    required_claim_categories = {
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
    claims = claim_evidence_matrix.get("claims", []) if isinstance(claim_evidence_matrix, dict) else []
    claim_categories = {
        claim.get("category")
        for claim in claims
        if isinstance(claim, dict) and isinstance(claim.get("category"), str)
    }
    evidence_paths = []
    missing_claim_terms = []
    malformed_claims = []
    if isinstance(claims, list):
        for claim in claims:
            if not isinstance(claim, dict):
                malformed_claims.append("<non-object claim>")
                continue
            claim_id = claim.get("id", "<missing id>")
            if not all(isinstance(claim.get(key), str) and claim.get(key).strip() for key in ("id", "category", "short_claim", "scope_limit")):
                malformed_claims.append(str(claim_id))
            terms = claim.get("manuscript_terms", [])
            if not isinstance(terms, list) or not terms:
                malformed_claims.append(f"{claim_id}.manuscript_terms")
            else:
                for term in terms:
                    if not isinstance(term, str) or term.casefold() not in tex_casefold:
                        missing_claim_terms.append(f"{claim_id}:{term}")
            evidence = claim.get("evidence", [])
            if not isinstance(evidence, list) or not evidence:
                malformed_claims.append(f"{claim_id}.evidence")
            else:
                for item in evidence:
                    if isinstance(item, dict) and isinstance(item.get("path"), str):
                        evidence_paths.append(item["path"])
                    else:
                        malformed_claims.append(f"{claim_id}.evidence_path")
    missing_claim_categories = sorted(required_claim_categories - claim_categories)
    missing_evidence_files = sorted({path for path in evidence_paths if not (base / path).exists()})
    claim_evidence_ok = (
        isinstance(claim_evidence_matrix, dict)
        and isinstance(claims, list)
        and len(claims) >= 10
        and not missing_claim_categories
        and not malformed_claims
        and not missing_claim_terms
        and not missing_evidence_files
    )
    add(
        checks,
        "Claim-evidence matrix",
        claim_evidence_ok,
        claim_evidence_matrix_error if claim_evidence_matrix_error else (
            f"{args.claim_evidence_matrix} present with {len(claims) if isinstance(claims, list) else 0} claims and {len(evidence_paths)} evidence links."
            if claim_evidence_ok
            else (
                "Problems: "
                + "; ".join(
                    item
                    for item in [
                        "missing categories: " + ", ".join(missing_claim_categories) if missing_claim_categories else "",
                        "malformed claims: " + ", ".join(malformed_claims) if malformed_claims else "",
                        "missing manuscript terms: " + ", ".join(missing_claim_terms) if missing_claim_terms else "",
                        "missing evidence files: " + ", ".join(missing_evidence_files) if missing_evidence_files else "",
                    ]
                    if item
                )
                if claim_evidence_matrix_path.exists()
                else f"Missing {args.claim_evidence_matrix}."
            )
        ),
    )
    required_editorial_scope_sections = [
        "## Central Posterior Claim",
        "## Claim Boundaries",
        "## Evidence for the Editor",
        "## Likely Reviewer Questions",
        "## Scope Decision for Submission",
    ]
    required_editorial_scope_terms = [
        "amortized posterior sampler for surface-wave dispersion inversion",
        "learned conditional posterior surrogate",
        r"q_{\theta}",
        "not a universally calibrated field-data posterior",
        "ak135 example is a standard-model stress test",
        "scalar posterior-temperature scaling is a calibration diagnostic",
        "depth-control basis is an implementation and parameterization choice",
        "Amortized neural posterior sampling for surface-wave dispersion",
    ]
    missing_editorial_scope_sections = [
        section for section in required_editorial_scope_sections if section not in editorial_scope_text
    ]
    editorial_scope_casefold = editorial_scope_text.casefold()
    missing_editorial_scope_terms = [
        term for term in required_editorial_scope_terms if term.casefold() not in editorial_scope_casefold
    ]
    add(
        checks,
        "Editorial scope brief",
        editorial_scope_path.exists() and not missing_editorial_scope_sections and not missing_editorial_scope_terms,
        (
            f"{args.editorial_scope_brief} present with central posterior claim, claim boundaries, editor evidence, reviewer questions, and scope decision sections."
            if editorial_scope_path.exists() and not missing_editorial_scope_sections and not missing_editorial_scope_terms
            else (
                "Missing sections/terms: " + ", ".join(missing_editorial_scope_sections + missing_editorial_scope_terms)
                if editorial_scope_path.exists()
                else f"Missing {args.editorial_scope_brief}."
            )
        ),
    )
    reviewer_playbook_text = (
        reviewer_response_playbook_path.read_text(encoding="utf-8")
        if reviewer_response_playbook_path.exists()
        else ""
    )
    required_reviewer_playbook_sections = [
        "## Scope Discipline",
        "## Response Matrix",
        "## Draft Response Language",
        "## Decision-Dependent Revision Branches",
        "## Response Quality Gate",
    ]
    required_reviewer_playbook_terms = [
        "amortized posterior sampling",
        "prior-dependent posterior surrogate",
        "Bayesian posterior semantics",
        "field-data validation",
        "moderately under-dispersed",
        "not a universally calibrated posterior",
        "validate_claim_evidence_matrix.py",
        "central_posterior_framing_audit.py",
        "framework_scope_audit.py",
        "submission_gate_report.py",
    ]
    missing_reviewer_playbook_sections = [
        section for section in required_reviewer_playbook_sections if section not in reviewer_playbook_text
    ]
    reviewer_playbook_casefold = reviewer_playbook_text.casefold()
    missing_reviewer_playbook_terms = [
        term for term in required_reviewer_playbook_terms if term.casefold() not in reviewer_playbook_casefold
    ]
    add(
        checks,
        "Reviewer response playbook",
        reviewer_response_playbook_path.exists()
        and not missing_reviewer_playbook_sections
        and not missing_reviewer_playbook_terms,
        (
            f"{args.reviewer_response_playbook} present with scope discipline, response matrix, draft language, revision branches, and response QA gate."
            if reviewer_response_playbook_path.exists()
            and not missing_reviewer_playbook_sections
            and not missing_reviewer_playbook_terms
            else (
                "Missing sections/terms: "
                + ", ".join(missing_reviewer_playbook_sections + missing_reviewer_playbook_terms)
                if reviewer_response_playbook_path.exists()
                else f"Missing {args.reviewer_response_playbook}."
            )
        ),
    )
    agu_guidance_text = (
        agu_guidance_snapshot_path.read_text(encoding="utf-8")
        if agu_guidance_snapshot_path.exists()
        else ""
    )
    required_agu_guidance_sections = [
        "## Official Sources Checked",
        "## Requirement-to-Evidence Map",
        "## Current Machine Checks",
        "## Current Non-Machine Decisions",
    ]
    required_agu_guidance_terms = [
        "Accessed: 2026-06-04",
        "https://www.agu.org/Publications/Authors/Journals",
        "https://www.agu.org/publications/authors/journals/submission-checklists",
        "https://www.agu.org/publications/authors/journals/text-graphics-requirements",
        "https://www.agu.org/publications/authors/journals/data-software-for-authors",
        "publication-unit",
        "agujournal2025",
        "Open Research",
        "Plain Language Summary",
        "amortized posterior sampling",
        "central_posterior_framing_audit.py",
        "framework_scope_audit.py",
        "reviewer_response_playbook.md",
    ]
    missing_agu_guidance_sections = [
        section for section in required_agu_guidance_sections if section not in agu_guidance_text
    ]
    missing_agu_guidance_terms = [
        term for term in required_agu_guidance_terms if term not in agu_guidance_text
    ]
    add(
        checks,
        "AGU guidance snapshot",
        agu_guidance_snapshot_path.exists()
        and not missing_agu_guidance_sections
        and not missing_agu_guidance_terms,
        (
            f"{args.agu_guidance_snapshot} present with official-source URLs, requirement-evidence map, machine checks, and non-machine decisions."
            if agu_guidance_snapshot_path.exists()
            and not missing_agu_guidance_sections
            and not missing_agu_guidance_terms
            else (
                "Missing sections/terms: "
                + ", ".join(missing_agu_guidance_sections + missing_agu_guidance_terms)
                if agu_guidance_snapshot_path.exists()
                else f"Missing {args.agu_guidance_snapshot}."
            )
        ),
    )
    bundle_prep_text = bundle_prep_path.read_text(encoding="utf-8") if bundle_prep_path.exists() else ""
    required_bundle_prep_terms = ["--dry-run", "--output", "archive_bundle_plan.json", "shutil.copy2"]
    missing_bundle_prep_terms = [term for term in required_bundle_prep_terms if term not in bundle_prep_text]
    add(
        checks,
        "Archive staging script",
        bundle_prep_path.exists() and not missing_bundle_prep_terms,
        (
            f"{args.bundle_prep} present with dry-run, output, plan, and copy support."
            if bundle_prep_path.exists() and not missing_bundle_prep_terms
            else (
                "Missing terms: " + ", ".join(missing_bundle_prep_terms)
                if bundle_prep_path.exists()
                else f"Missing {args.bundle_prep}."
            )
        ),
    )
    gate_report_text = gate_report_path.read_text(encoding="utf-8") if gate_report_path.exists() else ""
    required_gate_report_terms = [
        "submission_qa.py",
        "apply_final_submission_values.py",
        "validate_final_submission_values.py",
        "validate_final_submission_sync.py",
        "zenodo_metadata_templates.json",
        "validate_claim_evidence_matrix.py",
        "central_posterior_framing_audit.py",
        "framework_scope_audit.py",
        "validate_field_dispersion_input.py",
        "validate_claim_evidence_matrix.py",
        "figure_integrity_audit.py",
        "figure_preflight_audit.py",
        "publication_unit_audit.py",
        "posterior_predictive_check.py",
        "observation_noise_sensitivity.py",
        "calibration_split_sensitivity.py",
        "prepare_archive_bundles.py",
        "latexmk",
        "external_submission_ready",
        "human_blockers",
    ]
    missing_gate_report_terms = [term for term in required_gate_report_terms if term not in gate_report_text]
    add(
        checks,
        "Submission gate report runner",
        gate_report_path.exists() and not missing_gate_report_terms,
        (
            f"{args.gate_report} present with QA, final-values, field-input, archive, LaTeX, and blocker summary gates."
            if gate_report_path.exists() and not missing_gate_report_terms
            else (
                "Missing terms: " + ", ".join(missing_gate_report_terms)
                if gate_report_path.exists()
                else f"Missing {args.gate_report}."
            )
        ),
    )
    final_values_applier_text = final_values_applier_path.read_text(encoding="utf-8") if final_values_applier_path.exists() else ""
    required_applier_terms = [
        "validate_final_submission_values",
        "agujournaltemplate.tex",
        "references.bib",
        "cover_letter_jgrse.md",
        "jgrse_submission_metadata.md",
        "archive_manifest_zenodo.md",
        "zenodo_metadata_templates.json",
        "--dry-run",
        "--template",
    ]
    missing_applier_terms = [term for term in required_applier_terms if term not in final_values_applier_text]
    add(
        checks,
        "Final submission values applier",
        final_values_applier_path.exists() and not missing_applier_terms,
        (
            f"{args.final_values_applier} present with validation, dry-run, template guard, and target-file update support."
            if final_values_applier_path.exists() and not missing_applier_terms
            else (
                "Missing terms: " + ", ".join(missing_applier_terms)
                if final_values_applier_path.exists()
                else f"Missing {args.final_values_applier}."
            )
        ),
    )
    claim_evidence_validator_text = (
        claim_evidence_validator_path.read_text(encoding="utf-8")
        if claim_evidence_validator_path.exists()
        else ""
    )
    required_claim_evidence_validator_terms = [
        "REQUIRED_CATEGORIES",
        "required_json_paths",
        "required_npz_keys",
        "manuscript_terms",
        "claim_evidence_matrix.json",
    ]
    missing_claim_evidence_validator_terms = [
        term
        for term in required_claim_evidence_validator_terms
        if term not in claim_evidence_validator_text
    ]
    add(
        checks,
        "Claim-evidence validator",
        claim_evidence_validator_path.exists() and not missing_claim_evidence_validator_terms,
        (
            f"{args.claim_evidence_validator} present with category, manuscript-term, JSON-path, and NPZ-key checks."
            if claim_evidence_validator_path.exists() and not missing_claim_evidence_validator_terms
            else (
                "Missing terms: " + ", ".join(missing_claim_evidence_validator_terms)
                if claim_evidence_validator_path.exists()
                else f"Missing {args.claim_evidence_validator}."
            )
        ),
    )
    central_framing_text = (
        central_framing_audit_path.read_text(encoding="utf-8")
        if central_framing_audit_path.exists()
        else ""
    )
    required_central_framing_terms = [
        "title_posterior_focus",
        "abstract_posterior_precedes_control",
        "plain_language_central_contribution",
        "no_control_point_inversion_framing",
        "posterior_term_dominance",
        "control-point inversion",
        "amortized posterior sampling",
    ]
    missing_central_framing_terms = [
        term
        for term in required_central_framing_terms
        if term not in central_framing_text
    ]
    add(
        checks,
        "Central posterior framing audit",
        central_framing_audit_path.exists() and not missing_central_framing_terms,
        (
            f"{args.central_framing_audit} present with posterior-focus, control-point-framing, and term-dominance checks."
            if central_framing_audit_path.exists() and not missing_central_framing_terms
            else (
                "Missing terms: " + ", ".join(missing_central_framing_terms)
                if central_framing_audit_path.exists()
                else f"Missing {args.central_framing_audit}."
            )
        ),
    )
    framework_scope_text = (
        framework_scope_audit_path.read_text(encoding="utf-8")
        if framework_scope_audit_path.exists()
        else ""
    )
    required_framework_scope_terms = [
        "title_framework_with_demonstration",
        "abstract_framework_boundary",
        "methods_application_contract",
        "discussion_field_boundary",
        "no_unbounded_generalization",
        "boundary_language_density",
        "amortized posterior sampling for a surface-wave demonstration",
        "synthetic prior-predictive simulations",
    ]
    missing_framework_scope_terms = [
        term
        for term in required_framework_scope_terms
        if term not in framework_scope_text
    ]
    add(
        checks,
        "Framework scope audit",
        framework_scope_audit_path.exists() and not missing_framework_scope_terms,
        (
            f"{args.framework_scope_audit} present with framework-scope, application-contract, field-boundary, and unbounded-generalization checks."
            if framework_scope_audit_path.exists() and not missing_framework_scope_terms
            else (
                "Missing terms: " + ", ".join(missing_framework_scope_terms)
                if framework_scope_audit_path.exists()
                else f"Missing {args.framework_scope_audit}."
            )
        ),
    )

    add(checks, "Table count", table_count == 2, f"{table_count} main-text tables found; target is 2.")

    required_sections = [
        "Open Research Statement",
        "Inclusion in Global Research",
        "Author Contributions",
        "AI Assistance Disclosure",
        "Conflict of Interest Statement",
    ]
    missing_sections = [
        section
        for section in required_sections
        if not re.search(rf"\\section\*\{{{re.escape(section)}\}}", tex)
    ]
    if not extract_env(tex, "plainlanguagesummary") and not re.search(r"\\section\*\{Plain Language Summary\}", tex):
        missing_sections.insert(0, "Plain Language Summary")
    add(checks, "Required AGU sections", not missing_sections, "Missing sections: " + ", ".join(missing_sections) if missing_sections else "Required front/back-matter sections are present.")

    posterior_boundary_terms = [
        "prior-dependent posterior surrogate",
        "should not be read as a universally calibrated field-data posterior",
        "not be read as a calibrated posterior for arbitrary field dispersion picks",
        "not a substitute for an observational likelihood model",
        "not a retrained noise-aware posterior",
        "field application requires an explicit observational error model and independent calibration",
    ]
    missing_posterior_boundary_terms = [
        term for term in posterior_boundary_terms if term.casefold() not in tex_casefold
    ]
    add(
        checks,
        "Posterior-claim boundary language",
        not missing_posterior_boundary_terms,
        (
            "Manuscript retains prior-dependent posterior, field-calibration, likelihood-model, and noise-aware-posterior boundary language."
            if not missing_posterior_boundary_terms
            else "Missing boundary terms: " + ", ".join(missing_posterior_boundary_terms)
        ),
    )
    cover_letter_scope_terms = [
        "synthetic, ak135, and Bayan Obo stress-test demonstration of amortized posterior sampling",
        "not as calibrated field-data validation",
        "universally calibrated field-data posterior",
        "prior-dependent posterior surrogate",
        "field deployment requires observational-error modeling and independent calibration",
    ]
    cover_letter_casefold = cover_letter_text.casefold()
    missing_cover_letter_scope_terms = [
        term for term in cover_letter_scope_terms if term.casefold() not in cover_letter_casefold
    ]
    add(
        checks,
        "Cover letter scope language",
        cover_letter_path.exists() and not missing_cover_letter_scope_terms,
        (
            "Cover letter frames the work as a bounded amortized posterior-sampling demonstration and preserves posterior-calibration limitations."
            if cover_letter_path.exists() and not missing_cover_letter_scope_terms
            else (
                "Missing terms: " + ", ".join(missing_cover_letter_scope_terms)
                if cover_letter_path.exists()
                else f"Missing {args.cover_letter}."
            )
        ),
    )

    metric_needles = {
        "Evaluation examples": f"{metrics['n_eval']:,}",
        "Calibration examples": f"{metrics['split_temperature_calibration']['calibration_examples']:,}",
        "Test examples": f"{metrics['split_temperature_calibration']['test_examples']:,}",
        "MAE Vp": rounded(metrics["mae"]["Vp"]),
        "MAE Vs": rounded(metrics["mae"]["Vs"]),
        "MAE density": rounded(metrics["mae"]["rho"]),
        "Mean coverage": rounded(metrics["coverage_16_84"]["mean"]),
        "Split raw coverage": rounded(metrics["split_temperature_calibration"]["raw_test_coverage_16_84"]["mean"]),
        "Split scaled coverage": rounded(metrics["split_temperature_calibration"]["scaled_test_coverage_16_84"]["mean"]),
        "Temperature scale": f"{metrics['split_temperature_calibration']['temperature_scale']:.2f}",
        "ak135 Vs MAE": rounded(metrics["ak135_standard_model"]["mae"]["Vs"]),
        "ak135 mean MAE": rounded(metrics["ak135_standard_model"]["mae"]["mean"]),
        "ak135 coverage": rounded(metrics["ak135_standard_model"]["coverage_16_84"]["mean"]),
        "Dense mean MAE": rounded(metrics["dense_control_ablation"]["dense_mae"]["mean"]),
        "Control mean MAE": rounded(metrics["dense_control_ablation"]["control_mae"]["mean"]),
        "Dense roughness": rounded(metrics["dense_control_ablation"]["dense_roughness"]["mean"]),
        "Control roughness": f"{metrics['dense_control_ablation']['control_roughness']['mean']:.4f}",
    }
    missing_metrics = [name for name, needle in metric_needles.items() if needle not in tex]
    add(
        checks,
        "Metric-text consistency",
        not missing_metrics,
        "Missing rounded values: " + ", ".join(missing_metrics) if missing_metrics else "Core rounded metrics from figures/metrics.json appear in the manuscript.",
    )
    if benchmark is not None:
        benchmark_needles = {
            "Benchmark examples": f"{benchmark['n_examples']:,}",
            "Benchmark posterior samples": f"{benchmark['posterior_samples']:,}",
            "Benchmark sampling steps": f"{benchmark['sampling_steps']:,}",
            "Benchmark median seconds": f"{benchmark['median_seconds']:.2f}",
            "Benchmark curves/s": f"{benchmark['curves_per_second']:.0f}",
            "Benchmark posterior profiles/s": f"{benchmark['posterior_profiles_per_second']:,.0f}",
        }
        missing_benchmark = [name for name, needle in benchmark_needles.items() if needle not in tex]
        add(
            checks,
            "Benchmark-text consistency",
            not missing_benchmark,
            "Missing rounded values: " + ", ".join(missing_benchmark) if missing_benchmark else "Core rounded benchmark values appear in the manuscript.",
        )
    if sensitivity is not None:
        results = sensitivity["results"]
        sample_8 = next(item for item in results if item["posterior_samples"] == 8 and item["sampling_steps"] == 24)
        sample_32 = next(item for item in results if item["posterior_samples"] == 32 and item["sampling_steps"] == 24)
        baseline = next(item for item in results if item["is_baseline"])
        step_items = [item for item in results if item["posterior_samples"] == baseline["posterior_samples"]]
        max_step_mae_delta = max(abs(item["mae"]["mean"] - baseline["mae"]["mean"]) for item in step_items)
        max_step_cov_delta = max(abs(item["coverage_16_84"]["mean"] - baseline["coverage_16_84"]["mean"]) for item in step_items)
        sensitivity_needles = {
            "Sensitivity examples": f"{sensitivity['n_eval']:,}",
            "Sensitivity sample-8 MAE": rounded(sample_8["mae"]["mean"]),
            "Sensitivity sample-32 MAE": rounded(sample_32["mae"]["mean"]),
            "Sensitivity sample-8 coverage": rounded(sample_8["coverage_16_84"]["mean"]),
            "Sensitivity sample-32 coverage": rounded(sample_32["coverage_16_84"]["mean"]),
            "Sensitivity step MAE delta": rounded(max_step_mae_delta),
            "Sensitivity step coverage delta": rounded(max_step_cov_delta),
        }
        missing_sensitivity = [name for name, needle in sensitivity_needles.items() if needle not in tex]
        add(
            checks,
            "Sensitivity-text consistency",
            not missing_sensitivity,
            "Missing rounded values: " + ", ".join(missing_sensitivity) if missing_sensitivity else "Core rounded sensitivity values appear in the manuscript.",
        )
    if posterior_predictive is not None and isinstance(posterior_predictive.get("cases"), dict):
        pp_cases = posterior_predictive["cases"]
        pp_needles = {}
        if isinstance(pp_cases.get("synthetic"), dict):
            pp_needles["Synthetic posterior-predictive residual"] = (
                f"{pp_cases['synthetic']['weighted_summary']['median_abs_residual_km_s']:.3f}"
            )
        if isinstance(pp_cases.get("ak135"), dict):
            pp_needles["ak135 posterior-predictive residual"] = (
                f"{pp_cases['ak135']['weighted_summary']['mean_abs_residual_km_s']:.3f}"
            )
        missing_pp = [name for name, needle in pp_needles.items() if needle not in tex]
        add(
            checks,
            "Posterior-predictive text consistency",
            not missing_pp,
            "Missing rounded values: " + ", ".join(missing_pp) if missing_pp else "Core rounded posterior-predictive values appear in the manuscript.",
        )
    if observation_noise is not None and isinstance(observation_noise.get("results"), list):
        noise_by_level = {
            float(item["noise_sigma_km_s"]): item
            for item in observation_noise["results"]
            if isinstance(item, dict) and isinstance(item.get("noise_sigma_km_s"), (int, float))
        }
        if 0.0 in noise_by_level and 0.05 in noise_by_level:
            noise_needles = {
                "Noise baseline MAE": f"{noise_by_level[0.0]['mae']['mean']:.3f}",
                "Noise 0.05 MAE": f"{noise_by_level[0.05]['mae']['mean']:.3f}",
                "Noise baseline coverage": f"{noise_by_level[0.0]['coverage_16_84']['mean']:.3f}",
                "Noise 0.05 raw coverage": f"{noise_by_level[0.05]['coverage_16_84']['mean']:.3f}",
                "Noise 0.05 scaled coverage": f"{noise_by_level[0.05]['split_temperature']['scaled_test_coverage_16_84_mean']:.3f}",
            }
            missing_noise = [name for name, needle in noise_needles.items() if needle not in tex]
            add(
                checks,
                "Observation-noise text consistency",
                not missing_noise,
                "Missing rounded values: " + ", ".join(missing_noise) if missing_noise else "Core rounded observation-noise values appear in the manuscript.",
            )
    if calibration_split is not None and isinstance(calibration_split.get("summary"), dict):
        split_summary = calibration_split["summary"]
        if all(isinstance(split_summary.get(key), dict) for key in ("temperature_scale", "raw_test_coverage_16_84_mean", "scaled_test_coverage_16_84_mean")):
            split_needles = {
                "Split-stability temperature min": f"{split_summary['temperature_scale']['min']:.3f}",
                "Split-stability temperature max": f"{split_summary['temperature_scale']['max']:.3f}",
                "Split-stability raw coverage mean": f"{split_summary['raw_test_coverage_16_84_mean']['mean']:.3f}",
                "Split-stability scaled coverage mean": f"{split_summary['scaled_test_coverage_16_84_mean']['mean']:.3f}",
            }
            missing_split_stability = [name for name, needle in split_needles.items() if needle not in tex]
            add(
                checks,
                "Calibration split text consistency",
                not missing_split_stability,
                (
                    "Missing rounded values: " + ", ".join(missing_split_stability)
                    if missing_split_stability
                    else "Core rounded calibration split-stability values appear in the manuscript."
                ),
            )

    placeholders = [
        "Author Name",
        "Institution, City, Country",
        "email@example.com",
        "software DOI pending",
        "synthetic evaluation DOI pending",
        "Funding sources",
        "Author names, order, affiliations",
    ]
    found_placeholders = [p for p in placeholders if p in tex]
    add(
        checks,
        "Human-input placeholders",
        not found_placeholders,
        "Pending placeholders: " + ", ".join(found_placeholders) if found_placeholders else "No tracked placeholders remain.",
        warn=not args.strict_placeholders,
    )
    human_inputs_text = human_inputs_path.read_text(encoding="utf-8") if human_inputs_path.exists() else ""
    required_human_input_sections = [
        "## Manuscript Identity Inputs",
        "## Contribution and Declaration Inputs",
        "## Open Research and Archive Inputs",
        "## Reviewer Inputs",
        "## Scientific Scope Decision",
        "## Final Update Sequence",
    ]
    missing_human_sections = [
        section for section in required_human_input_sections if section not in human_inputs_text
    ]
    add(
        checks,
        "Human-input completion packet",
        human_inputs_path.exists() and not missing_human_sections,
        (
            f"{args.human_inputs} present with final-submission input sections."
            if human_inputs_path.exists() and not missing_human_sections
            else (
                "Missing sections: " + ", ".join(missing_human_sections)
                if human_inputs_path.exists()
                else f"Missing {args.human_inputs}."
            )
        ),
    )

    final_values_template = None
    final_values_template_error = ""
    if final_values_template_path.exists():
        try:
            final_values_template = json.loads(final_values_template_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            final_values_template_error = str(exc)
    required_final_value_sections = [
        "schema_version",
        "manuscript_identity",
        "contributions_and_declarations",
        "open_research",
        "reviewers",
        "scope_decision",
    ]
    missing_final_value_sections = [
        section for section in required_final_value_sections
        if not isinstance(final_values_template, dict) or section not in final_values_template
    ]
    reviewer_template_count = 0
    if isinstance(final_values_template, dict):
        reviewers = final_values_template.get("reviewers", {})
        if isinstance(reviewers, dict) and isinstance(reviewers.get("suggested"), list):
            reviewer_template_count = len(reviewers["suggested"])
    final_template_ok = (
        isinstance(final_values_template, dict)
        and not missing_final_value_sections
        and reviewer_template_count >= 3
    )
    add(
        checks,
        "Final submission values template",
        final_template_ok,
        final_values_template_error if final_values_template_error else (
            f"{args.final_values_template} present with required sections and {reviewer_template_count} reviewer slots."
            if final_template_ok
            else (
                "Missing sections: " + ", ".join(missing_final_value_sections)
                if final_values_template_path.exists()
                else f"Missing {args.final_values_template}."
            )
        ),
    )
    final_values_validator_text = (
        final_values_validator_path.read_text(encoding="utf-8")
        if final_values_validator_path.exists()
        else ""
    )
    required_final_values_validator_terms = [
        "--template",
        "final_submission_values.json",
        "ORCID_RE",
        "DOI_RE",
        "reviewers",
        "scope_decision",
    ]
    missing_final_values_validator_terms = [
        term for term in required_final_values_validator_terms
        if term not in final_values_validator_text
    ]
    add(
        checks,
        "Final submission values validator",
        final_values_validator_path.exists() and not missing_final_values_validator_terms,
        (
            f"{args.final_values_validator} present with template, DOI, ORCID, reviewer, and scope checks."
            if final_values_validator_path.exists() and not missing_final_values_validator_terms
            else (
                "Missing terms: " + ", ".join(missing_final_values_validator_terms)
                if final_values_validator_path.exists()
                else f"Missing {args.final_values_validator}."
            )
        ),
    )
    final_sync_validator_text = (
        final_sync_validator_path.read_text(encoding="utf-8")
        if final_sync_validator_path.exists()
        else ""
    )
    required_final_sync_terms = [
        "final_submission_values.json",
        "agujournaltemplate.tex",
        "references.bib",
        "cover_letter_jgrse.md",
        "jgrse_submission_metadata.md",
        "zenodo_metadata_templates.json",
        "software_doi",
        "reviewers",
    ]
    missing_final_sync_terms = [
        term for term in required_final_sync_terms
        if term not in final_sync_validator_text
    ]
    add(
        checks,
        "Final submission sync validator",
        final_sync_validator_path.exists() and not missing_final_sync_terms,
        (
            f"{args.final_sync_validator} present with manuscript, reference, cover letter, metadata, Zenodo, DOI, and reviewer sync checks."
            if final_sync_validator_path.exists() and not missing_final_sync_terms
            else (
                "Missing terms: " + ", ".join(missing_final_sync_terms)
                if final_sync_validator_path.exists()
                else f"Missing {args.final_sync_validator}."
            )
        ),
    )
    field_calibration_plan_text = (
        field_calibration_plan_path.read_text(encoding="utf-8")
        if field_calibration_plan_path.exists()
        else ""
    )
    required_field_calibration_sections = [
        "## Submission Decision Tree",
        "## Minimum Field Example Requirements",
        "## Calibration Experiment Requirements",
        "## Acceptance Gates",
        "## Manuscript Integration Plan",
        "## Archive Additions Needed for an Upgrade",
    ]
    missing_field_calibration_sections = [
        section for section in required_field_calibration_sections
        if section not in field_calibration_plan_text
    ]
    add(
        checks,
        "Field calibration upgrade plan",
        field_calibration_plan_path.exists() and not missing_field_calibration_sections,
        (
            f"{args.field_calibration_plan} present with decision, field-example, calibration, acceptance, manuscript, and archive sections."
            if field_calibration_plan_path.exists() and not missing_field_calibration_sections
            else (
                "Missing sections: " + ", ".join(missing_field_calibration_sections)
                if field_calibration_plan_path.exists()
                else f"Missing {args.field_calibration_plan}."
            )
        ),
    )
    field_dispersion_validator_text = (
        field_dispersion_validator_path.read_text(encoding="utf-8")
        if field_dispersion_validator_path.exists()
        else ""
    )
    required_field_dispersion_validator_terms = [
        "--output-npz",
        "field_dispersion_input.csv",
        "Rayleigh",
        "Love",
        "period_s",
        "uncertainty_km_s",
        "dispersion",
        "mask",
    ]
    missing_field_dispersion_validator_terms = [
        term for term in required_field_dispersion_validator_terms
        if term not in field_dispersion_validator_text
    ]
    add(
        checks,
        "Field dispersion validator",
        field_dispersion_validator_path.exists() and not missing_field_dispersion_validator_terms,
        (
            f"{args.field_dispersion_validator} present with CSV validation and NPZ tensor output support."
            if field_dispersion_validator_path.exists() and not missing_field_dispersion_validator_terms
            else (
                "Missing terms: " + ", ".join(missing_field_dispersion_validator_terms)
                if field_dispersion_validator_path.exists()
                else f"Missing {args.field_dispersion_validator}."
            )
        ),
    )
    field_dispersion_template_text = (
        field_dispersion_template_path.read_text(encoding="utf-8")
        if field_dispersion_template_path.exists()
        else ""
    )
    field_dispersion_template_columns = (
        field_dispersion_template_text.splitlines()[0].split(",")
        if field_dispersion_template_text.strip()
        else []
    )
    required_field_dispersion_columns = [
        "wave",
        "mode",
        "period_s",
        "phase_velocity_km_s",
        "uncertainty_km_s",
        "source",
        "notes",
    ]
    missing_field_dispersion_columns = [
        column for column in required_field_dispersion_columns
        if column not in field_dispersion_template_columns
    ]
    field_template_has_waves = (
        "Rayleigh" in field_dispersion_template_text
        and "Love" in field_dispersion_template_text
    )
    add(
        checks,
        "Field dispersion input template",
        field_dispersion_template_path.exists() and not missing_field_dispersion_columns and field_template_has_waves,
        (
            f"{args.field_dispersion_template} present with required columns and Rayleigh/Love example rows."
            if field_dispersion_template_path.exists() and not missing_field_dispersion_columns and field_template_has_waves
            else (
                "Missing columns: " + ", ".join(missing_field_dispersion_columns)
                if field_dispersion_template_path.exists()
                else f"Missing {args.field_dispersion_template}."
            )
        ),
    )

    metadata_path = base / "jgrse_submission_metadata.md"
    if metadata_path.exists():
        metadata_text = metadata_path.read_text(encoding="utf-8")
        required_metadata_sections = [
            "## Submission Form Fields",
            "## Title",
            "## Key Points",
            "## Abstract",
            "## Plain Language Summary",
            "## Keywords",
            "## Suggested Reviewers",
            "## Open Research Statement Status",
            "## Final Go or No-Go Items",
        ]
        missing_metadata = [
            section for section in required_metadata_sections if section not in metadata_text
        ]
        add(
            checks,
            "Submission metadata packet",
            not missing_metadata,
            "Missing sections: " + ", ".join(missing_metadata) if missing_metadata else "Submission metadata packet is present with required sections.",
        )
        stale_support_terms = [
            "Current sensitivity diagnostic uses 128 held-out examples",
            "Decide whether to extend the diagnostic to 1,024 examples",
            "Current QA result: 29 pass",
            "about 15.7 PU",
            "Current rough publication units",
        ]
        stale_support_hits = [
            term for term in stale_support_terms if term in human_inputs_text or term in metadata_text
        ]
        add(
            checks,
            "Support packet freshness",
            not stale_support_hits,
            "No stale 128-example sensitivity or old QA-count text remains."
            if not stale_support_hits
            else "Stale terms: " + ", ".join(stale_support_hits),
        )
        publication_metadata_needles = {}
        if isinstance(publication_unit, dict) and isinstance(publication_counts, dict):
            publication_metadata_needles = {
                "publication-unit command": "python disp_inv_scripts/publication_unit_audit.py",
                "publication-unit words": f"{publication_word_count:,}",
                "publication-unit value": f"{float(publication_units):.3f} PU" if isinstance(publication_units, (int, float)) else "",
                "publication-unit margin": f"{float(publication_counts.get('publication_unit_margin')):.3f} PU"
                if isinstance(publication_counts.get("publication_unit_margin"), (int, float))
                else "",
                "publication-unit threshold": f"{float(publication_limit):.0f} PU" if isinstance(publication_limit, (int, float)) else "",
            }
        missing_publication_metadata = [
            name for name, needle in publication_metadata_needles.items() if not needle or needle not in metadata_text
        ]
        add(
            checks,
            "Metadata publication-unit sync",
            bool(publication_metadata_needles) and not missing_publication_metadata,
            (
                "Metadata packet reports the publication-unit audit command, counted words, PU value, margin, and threshold."
                if publication_metadata_needles and not missing_publication_metadata
                else (
                    "Missing publication-unit metadata: " + ", ".join(missing_publication_metadata)
                    if publication_metadata_needles
                    else f"Missing or unreadable {args.publication_unit_audit} for metadata sync."
                )
            ),
        )
        sync_mismatches = []
        title_groups = brace_groups_after_command(tex, "title", 1)
        manuscript_title = title_groups[0] if title_groups else ""
        if normalize_submission_text(manuscript_title) != normalize_submission_text(markdown_section(metadata_text, "Title")):
            sync_mismatches.append("title")

        metadata_keypoints = markdown_numbered_items(markdown_section(metadata_text, "Key Points"))
        if normalized_sequence(keypoints) != normalized_sequence(metadata_keypoints):
            sync_mismatches.append("Key Points")

        if normalize_submission_text(abstract) != normalize_submission_text(markdown_section(metadata_text, "Abstract")):
            sync_mismatches.append("abstract")

        if normalize_submission_text(pls) != normalize_submission_text(markdown_section(metadata_text, "Plain Language Summary")):
            sync_mismatches.append("Plain Language Summary")

        manuscript_keywords = split_keywords(extract_star_section(tex, "Keywords"))
        metadata_keywords = split_keywords(markdown_section(metadata_text, "Keywords"))
        if normalized_sequence(manuscript_keywords) != normalized_sequence(metadata_keywords):
            sync_mismatches.append("keywords")

        add(
            checks,
            "Metadata-manuscript sync",
            not sync_mismatches,
            "Out of sync: " + ", ".join(sync_mismatches) if sync_mismatches else "Title, Key Points, abstract, Plain Language Summary, and keywords match the manuscript.",
        )
    else:
        add(
            checks,
            "Submission metadata packet",
            False,
            "Missing jgrse_submission_metadata.md.",
        )

    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic pre-submission QA checks.")
    parser.add_argument("--root", default=".", help="Repository root containing the manuscript.")
    parser.add_argument("--manuscript", default="agujournaltemplate.tex")
    parser.add_argument("--bib", default="references.bib")
    parser.add_argument("--cover-letter", default="cover_letter_jgrse.md")
    parser.add_argument("--reference-audit", default="reference_integrity_audit.json")
    parser.add_argument("--figures", default="figures")
    parser.add_argument("--figure-generator", default="disp_inv_scripts/make_prior_boundary_workflow_figure.py")
    parser.add_argument("--figure-integrity", default="figures/figure_integrity_audit.json")
    parser.add_argument("--figure-preflight", default="figures/figure_preflight_audit.json")
    parser.add_argument("--publication-unit-audit", default="publication_unit_audit.json")
    parser.add_argument("--metrics", default="figures/metrics.json")
    parser.add_argument("--benchmark", default="figures/inference_benchmark.json")
    parser.add_argument("--sensitivity", default="figures/sampling_sensitivity.json")
    parser.add_argument("--posterior-predictive", default="figures/posterior_predictive_check.json")
    parser.add_argument("--observation-noise", default="figures/observation_noise_sensitivity.json")
    parser.add_argument("--calibration-split", default="figures/calibration_split_sensitivity.json")
    parser.add_argument("--figure-samples", default="figures/posterior_figure_samples.npz")
    parser.add_argument("--environment", default="environment_report.json")
    parser.add_argument("--environment-spec", default="environment-repro.yml")
    parser.add_argument("--environment-validation", default="environment_validation.json")
    parser.add_argument("--inventory", default="archive_inventory.json")
    parser.add_argument("--bundle-plan", default="archive_bundle_plan.json")
    parser.add_argument("--bundle-prep", default="disp_inv_scripts/prepare_archive_bundles.py")
    parser.add_argument("--gate-report", default="disp_inv_scripts/submission_gate_report.py")
    parser.add_argument("--archive-readme", default="archive_readme_zenodo.md")
    parser.add_argument("--reproducibility-notes", default="reproducibility_notes.md")
    parser.add_argument("--zenodo-metadata", default="zenodo_metadata_templates.json")
    parser.add_argument("--claim-evidence-matrix", default="claim_evidence_matrix.json")
    parser.add_argument("--claim-evidence-validator", default="disp_inv_scripts/validate_claim_evidence_matrix.py")
    parser.add_argument("--central-framing-audit", default="disp_inv_scripts/central_posterior_framing_audit.py")
    parser.add_argument("--framework-scope-audit", default="disp_inv_scripts/framework_scope_audit.py")
    parser.add_argument("--editorial-scope-brief", default="editorial_scope_brief.md")
    parser.add_argument("--reviewer-response-playbook", default="reviewer_response_playbook.md")
    parser.add_argument("--agu-guidance-snapshot", default="agu_guidance_snapshot.md")
    parser.add_argument("--human-inputs", default="jgrse_human_input_packet.md")
    parser.add_argument("--final-values-template", default="final_submission_values.template.json")
    parser.add_argument("--final-values-applier", default="disp_inv_scripts/apply_final_submission_values.py")
    parser.add_argument("--final-values-validator", default="disp_inv_scripts/validate_final_submission_values.py")
    parser.add_argument("--final-sync-validator", default="disp_inv_scripts/validate_final_submission_sync.py")
    parser.add_argument("--field-calibration-plan", default="field_calibration_upgrade_plan.md")
    parser.add_argument("--field-dispersion-template", default="field_dispersion_input.template.csv")
    parser.add_argument("--field-dispersion-validator", default="disp_inv_scripts/validate_field_dispersion_input.py")
    parser.add_argument("--max-publication-units", type=float, default=25.0)
    parser.add_argument("--min-inventory-records", type=int, default=76)
    parser.add_argument("--strict-placeholders", action="store_true", help="Treat tracked author/DOI placeholders as failures.")
    args = parser.parse_args()

    checks = run_checks(args)
    width = max(len(c.name) for c in checks)
    for check in checks:
        print(f"[{check.status}] {check.name:<{width}}  {check.detail}")

    failed = [c for c in checks if c.status == "FAIL"]
    warned = [c for c in checks if c.status == "WARN"]
    print(f"\nSummary: {len(checks) - len(failed) - len(warned)} pass, {len(warned)} warn, {len(failed)} fail.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
