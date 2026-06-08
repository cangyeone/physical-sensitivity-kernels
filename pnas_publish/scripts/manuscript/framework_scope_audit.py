#!/usr/bin/env python3
"""Audit the posterior-surrogate claim boundary for the manuscript.

This deterministic text audit protects the revised theme: the paper presents an
amortized posterior sampler for surface-wave dispersion, while the evidence in
the current manuscript comes from synthetic prior-predictive simulations, an
ak135 standard-model stress test, and an uncalibrated Bayan Obo field-data
stress test.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ScopeCheck:
    name: str
    ok: bool
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


def section_after(text: str, needle: str, max_chars: int = 2200) -> str:
    index = text.find(needle)
    return "" if index < 0 else text[index : index + max_chars]


def contains_all(text: str, needles: list[str]) -> tuple[bool, list[str]]:
    lowered = text.casefold()
    missing = [needle for needle in needles if needle.casefold() not in lowered]
    return not missing, missing


def add(checks: list[ScopeCheck], name: str, ok: bool, detail: str) -> None:
    checks.append(ScopeCheck(name=name, ok=ok, detail=detail))


def count_terms(text: str, terms: list[str]) -> dict[str, int]:
    lowered = text.casefold()
    return {term: lowered.count(term.casefold()) for term in terms}


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


def read_claims(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []
    claims = parsed.get("claims") if isinstance(parsed, dict) else None
    return [claim for claim in claims if isinstance(claim, dict)] if isinstance(claims, list) else []


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    manuscript_path = root / args.manuscript
    cover_letter_path = root / args.cover_letter
    scope_brief_path = root / args.editorial_scope_brief
    reviewer_playbook_path = root / args.reviewer_response_playbook
    metadata_path = root / args.metadata
    claim_matrix_path = root / args.claim_evidence_matrix

    tex = strip_comments(read_text(manuscript_path))
    cover_letter = read_text(cover_letter_path)
    scope_brief = read_text(scope_brief_path)
    reviewer_playbook = read_text(reviewer_playbook_path)
    metadata = read_text(metadata_path)
    claims = read_claims(claim_matrix_path)
    combined = "\n".join([tex, cover_letter, scope_brief, reviewer_playbook, metadata])

    title_groups = brace_groups_after_command(tex, "title", 1)
    title = title_groups[0] if title_groups else ""
    abstract = extract_env(tex, "abstract")
    methods_framework = section_after(tex, "The method starts from the same objects used in Bayesian inverse theory", 3200)
    contribution_paragraph = section_after(tex, "The study makes three contributions.", 1600)
    discussion_scope = section_after(tex, "The method converts a simulator-defined Bayesian inverse problem", 3000)

    checks: list[ScopeCheck] = []
    add(
        checks,
        "title_framework_with_demonstration",
        "Amortized Posterior Sampling" in title
        and "Surface-Wave Dispersion" in title
        and "Conditional Rectified Flow" in title,
        "Title presents amortized posterior sampling for the demonstrated surface-wave inverse problem."
        if "Amortized Posterior Sampling" in title
        and "Surface-Wave Dispersion" in title
        and "Conditional Rectified Flow" in title
        else "Title does not clearly combine amortized posterior sampling and the surface-wave demonstration.",
    )

    ok, missing = contains_all(
        abstract,
        [
            "amortized posterior sampler",
            "posterior surrogate",
            "not a universally calibrated field-data posterior",
            "field application requires an observational error model and independent calibration",
        ],
    )
    add(
        checks,
        "abstract_framework_boundary",
        ok,
        "Abstract pairs amortized posterior-sampling language with field-calibration boundaries."
        if ok
        else "Missing abstract terms: " + ", ".join(missing),
    )

    ok, missing = contains_all(
        contribution_paragraph,
        [
            "surface-wave dispersion inversion as amortized posterior sampling",
            "prior-dependent posterior surrogate",
            "synthetic validation, standard-model transfer, and uncalibrated field-transfer stress testing",
            "field deployment still requires observational-error modeling and independent calibration",
        ],
    )
    add(
        checks,
        "contribution_scope_boundary",
        ok,
        "Contribution paragraph separates the posterior-sampling claim from the surface-wave evidence."
        if ok
        else "Missing contribution-boundary terms: " + ", ".join(missing),
    )

    ok, missing = contains_all(
        methods_framework,
        [
            "model vector",
            "observation vector",
            "prior",
            "simulator or likelihood model",
            "training joint distribution",
            "conditional posterior surrogate",
            "masking or observation process",
            "noise model",
            "specification-and-training procedure",
            "probability object remains",
            "field use requires an observational error model and independent calibration",
        ],
    )
    add(
        checks,
        "methods_application_contract",
        ok,
        "Methods define the reusable Bayesian objects and require application-specific priors, simulators, masks, and noise assumptions."
        if ok
        else "Missing methods contract terms: " + ", ".join(missing),
    )

    ok, missing = contains_all(
        discussion_scope,
        [
            "prior-dependent posterior surrogate",
            "not replace a physical observational error model",
            "evidence is therefore strongest for synthetic validation, standard-model transfer, and uncalibrated field-data stress testing",
            "field application requires an explicit observational error model and independent calibration",
        ],
    )
    add(
        checks,
        "discussion_field_boundary",
        ok,
        "Discussion states the learned posterior boundary and field-deployment requirements."
        if ok
        else "Missing discussion-boundary terms: " + ", ".join(missing),
    )

    ok, missing = contains_all(
        scope_brief,
        [
            "amortized posterior sampler for surface-wave dispersion inversion",
            "Surface-wave dispersion is the principal worked example",
            "not a universally calibrated field-data posterior",
            "nor does one surface-wave demonstration validate every possible geophysical inverse problem",
            "Amortized neural posterior sampling for surface-wave dispersion",
            "specification-and-training procedure",
            "not a claim that one trained network transfers unchanged to every data type",
        ],
    )
    add(
        checks,
        "editorial_scope_boundary",
        ok,
        "Editorial scope brief protects the posterior-surrogate claim boundary."
        if ok
        else "Missing scope-brief terms: " + ", ".join(missing),
    )

    ok, missing = contains_all(
        reviewer_playbook,
        [
            "present submission is a synthetic, ak135, and Bayan Obo stress-test demonstration of amortized posterior sampling",
            "Do not claim that the current network is a universally calibrated posterior for arbitrary field data",
            "specification-and-training procedure for a simulator-defined posterior sampler",
            "not an unconstrained neural ensemble",
            "not calibrated field-data validation",
        ],
    )
    add(
        checks,
        "reviewer_response_boundary",
        ok,
        "Reviewer playbook keeps later responses from overstating framework generality."
        if ok
        else "Missing reviewer-playbook terms: " + ", ".join(missing),
    )

    ok, missing = contains_all(
        cover_letter,
        [
            "synthetic, ak135, and Bayan Obo stress-test demonstration of amortized posterior sampling",
            "prior-dependent posterior surrogate",
            "each deployment requires an application-specific prior, simulator, observational-error model, and independent calibration",
            "not as calibrated field-data validation or a universally calibrated field-data posterior",
        ],
    )
    add(
        checks,
        "cover_letter_scope_boundary",
        ok,
        "Cover letter frames the submission as a bounded amortized posterior-sampling demonstration."
        if ok
        else "Missing cover-letter terms: " + ", ".join(missing),
    )

    c01 = next((claim for claim in claims if claim.get("id") == "C01"), {})
    c01_text = "\n".join(
        [
            str(c01.get("short_claim", "")),
            str(c01.get("scope_limit", "")),
            "\n".join(str(term) for term in c01.get("manuscript_terms", []) if isinstance(term, str)),
        ]
    )
    ok, missing = contains_all(
        c01_text,
        [
            "conditional rectified-flow amortized posterior sampler",
            "prior-dependent posterior surrogate",
            "not a universally calibrated field-data posterior",
        ],
    )
    add(
        checks,
        "claim_matrix_scope_limit",
        ok,
        "Claim-evidence matrix binds the central posterior-sampling claim to a scope limit."
        if ok
        else "Missing C01 claim-matrix terms: " + ", ".join(missing),
    )

    forbidden_phrases = [
        "validated for all geophysical inverse problems",
        "validated for every possible geophysical inverse problem",
        "guarantees calibrated posterior",
        "calibrated for arbitrary field observations",
        "field-data posterior without calibration",
        "universal solution to geophysical inversion",
    ]
    forbidden_hits = [phrase for phrase in forbidden_phrases if phrase.casefold() in combined.casefold()]
    add(
        checks,
        "no_unbounded_generalization",
        not forbidden_hits,
        "No unsupported universal framework claims found."
        if not forbidden_hits
        else "Unsupported claims found: " + ", ".join(forbidden_hits),
    )

    boundary_terms = [
        "principal demonstration",
        "prior-dependent posterior surrogate",
        "not a universally calibrated field-data posterior",
        "Bayan Obo",
        "field application requires",
        "independent calibration",
        "not calibrated field-data validation",
    ]
    framework_terms = ["amortized posterior", "posterior sampler", "posterior surrogate", "conditional posterior"]
    boundary_counts = count_terms(combined, boundary_terms)
    framework_counts = count_terms(combined, framework_terms)
    add(
        checks,
        "boundary_language_density",
        sum(boundary_counts.values()) >= 12 and sum(framework_counts.values()) >= 20,
        f"Boundary terms total {sum(boundary_counts.values())}; posterior-surrogate terms total {sum(framework_counts.values())}.",
    )

    failures = [check for check in checks if not check.ok]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Audit that the manuscript is framed as amortized posterior sampling for a surface-wave demonstration, without unsupported universal field claims.",
        "manuscript": str(manuscript_path),
        "cover_letter": str(cover_letter_path),
        "editorial_scope_brief": str(scope_brief_path),
        "reviewer_response_playbook": str(reviewer_playbook_path),
        "metadata": str(metadata_path),
        "claim_evidence_matrix": str(claim_matrix_path),
        "counts": {
            "framework_terms": framework_counts,
            "boundary_terms": boundary_counts,
        },
        "checks": [
            {
                "name": check.name,
                "status": "PASS" if check.ok else "FAIL",
                "detail": check.detail,
            }
            for check in checks
        ],
        "failure_reasons": [check.detail for check in failures],
        "overall_status": "PASS" if not failures else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit posterior-surrogate scope boundaries.")
    parser.add_argument("--root", default=".", help="Overleaf project root.")
    parser.add_argument("--manuscript", default="agujournaltemplate.tex")
    parser.add_argument("--cover-letter", default="cover_letter_jgrse.md")
    parser.add_argument("--editorial-scope-brief", default="editorial_scope_brief.md")
    parser.add_argument("--reviewer-response-playbook", default="reviewer_response_playbook.md")
    parser.add_argument("--metadata", default="jgrse_submission_metadata.md")
    parser.add_argument("--claim-evidence-matrix", default="claim_evidence_matrix.json")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    report = build_report(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
