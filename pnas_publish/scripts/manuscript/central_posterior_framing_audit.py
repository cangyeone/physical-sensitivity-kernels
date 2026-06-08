#!/usr/bin/env python3
"""Audit whether the manuscript framing keeps amortized posterior sampling central.

This is a deterministic text audit, not a scientific review. It protects the
intended manuscript message: the paper is about a conditional rectified-flow
posterior surrogate for surface-wave dispersion, and depth controls are an
implementation choice for regular samples.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class TextCheck:
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


def section_after(text: str, needle: str, max_chars: int = 2000) -> str:
    index = text.find(needle)
    return "" if index < 0 else text[index : index + max_chars]


def add(checks: list[TextCheck], name: str, ok: bool, detail: str) -> None:
    checks.append(TextCheck(name=name, ok=ok, detail=detail))


def contains_all(text: str, needles: list[str]) -> tuple[bool, list[str]]:
    lowered = text.casefold()
    missing = [needle for needle in needles if needle.casefold() not in lowered]
    return not missing, missing


def first_index(text: str, needle: str) -> int | None:
    index = text.casefold().find(needle.casefold())
    return index if index >= 0 else None


def count_terms(text: str, terms: list[str]) -> int:
    lowered = text.casefold()
    return sum(lowered.count(term.casefold()) for term in terms)


def build_report(args: argparse.Namespace) -> dict:
    root = Path(args.root)
    manuscript_path = root / args.manuscript
    cover_letter_path = root / args.cover_letter
    scope_brief_path = root / args.editorial_scope_brief
    metadata_path = root / args.metadata

    tex = strip_comments(manuscript_path.read_text(encoding="utf-8"))
    cover_letter = cover_letter_path.read_text(encoding="utf-8") if cover_letter_path.exists() else ""
    scope_brief = scope_brief_path.read_text(encoding="utf-8") if scope_brief_path.exists() else ""
    metadata = metadata_path.read_text(encoding="utf-8") if metadata_path.exists() else ""
    combined = "\n".join([tex, cover_letter, scope_brief, metadata])

    title = brace_groups_after_command(tex, "title", 1)[0] if brace_groups_after_command(tex, "title", 1) else ""
    abstract = extract_env(tex, "abstract")
    plain_language = extract_env(tex, "plainlanguagesummary")
    contributions = section_after(tex, "The study makes three contributions.", 1400)
    discussion = section_after(tex, "The method converts a simulator-defined Bayesian inverse problem", 3000)
    conclusion = section_after(tex, "\\section{Conclusions}", 1800)

    checks: list[TextCheck] = []
    add(
        checks,
        "title_posterior_focus",
        "Amortized Posterior Sampling" in title
        and "Surface-Wave Dispersion" in title
        and "Conditional Rectified Flow" in title,
        "Title foregrounds amortized posterior sampling for surface-wave dispersion."
        if "Amortized Posterior Sampling" in title
        and "Surface-Wave Dispersion" in title
        and "Conditional Rectified Flow" in title
        else "Title does not foreground amortized posterior sampling, surface-wave dispersion, and conditional rectified flow.",
    )

    ok, missing = contains_all(
        abstract,
        [
            "amortized posterior sampler",
            "posterior surrogate",
            "not a universally calibrated field-data posterior",
            "draws ensembles",
        ],
    )
    posterior_idx = first_index(abstract, "amortized posterior sampler")
    control_idx = first_index(abstract, "depth-control")
    add(
        checks,
        "abstract_posterior_precedes_control",
        ok and posterior_idx is not None and control_idx is not None and posterior_idx < control_idx,
        "Abstract states the amortized posterior-sampling objective before introducing the depth-control basis."
        if ok and posterior_idx is not None and control_idx is not None and posterior_idx < control_idx
        else "Missing terms or ordering problem: " + ", ".join(missing),
    )

    ok, missing = contains_all(
        plain_language,
        [
            "deep neural network can sample a posterior distribution",
            "synthetic model prior",
            "central contribution is an amortized posterior sampler",
        ],
    )
    add(
        checks,
        "plain_language_central_contribution",
        ok,
        "Plain Language Summary identifies amortized posterior sampling as the central contribution."
        if ok
        else "Missing PLS terms: " + ", ".join(missing),
    )

    ok, missing = contains_all(
        tex,
        [
            "The central methodological question of this study is whether a deep neural network can be trained as an amortized posterior sampler",
            "The depth-control basis is not the scientific endpoint",
            "The central object remains the learned conditional posterior surrogate",
        ],
    )
    add(
        checks,
        "introduction_scope_hierarchy",
        ok,
        "Introduction frames depth controls below the conditional posterior distribution."
        if ok
        else "Missing introduction hierarchy terms: " + ", ".join(missing),
    )

    posterior_contrib_idx = first_index(contributions, "amortized posterior sampling")
    control_contrib_idx = first_index(contributions, "depth-aware sampling basis")
    add(
        checks,
        "contribution_order",
        posterior_contrib_idx is not None
        and control_contrib_idx is not None
        and posterior_contrib_idx < control_contrib_idx,
        "Contribution paragraph lists amortized posterior sampling before the depth-control basis."
        if posterior_contrib_idx is not None
        and control_contrib_idx is not None
        and posterior_contrib_idx < control_contrib_idx
        else "Contribution paragraph ordering does not clearly put amortized posterior sampling first.",
    )

    ok, missing = contains_all(
        discussion,
        [
            "method converts a simulator-defined Bayesian inverse problem",
            "prior-dependent posterior surrogate",
            "field application requires an explicit observational error model",
            "posterior representation is valuable",
        ],
    )
    add(
        checks,
        "discussion_posterior_boundary",
        ok,
        "Discussion keeps posterior value and calibration boundary language together."
        if ok
        else "Missing discussion terms: " + ", ".join(missing),
    )

    ok, missing = contains_all(
        conclusion,
        [
            "amortized posterior sampler",
            "conditional rectified flow can draw ensembles",
            "Bayan Obo",
            "observational noise models",
        ],
    )
    add(
        checks,
        "conclusion_posterior_message",
        ok,
        "Conclusion states amortized posterior sampling and field-calibration limits."
        if ok
        else "Missing conclusion terms: " + ", ".join(missing),
    )

    ok, missing = contains_all(
        cover_letter,
        [
            "main methodological contribution is the simulator-defined posterior surrogate",
            "scientific focus is the learned conditional posterior",
            "synthetic, ak135, and Bayan Obo stress-test demonstration of amortized posterior sampling",
        ],
    )
    add(
        checks,
        "cover_letter_central_claim",
        ok,
        "Cover letter foregrounds neural posterior inversion and limits field-data claims."
        if ok
        else "Missing cover-letter terms: " + ", ".join(missing),
    )

    ok, missing = contains_all(
        scope_brief,
        [
            "Central Posterior Claim",
            "amortized posterior sampler for surface-wave dispersion inversion",
            "depth-control basis is an implementation and parameterization choice",
            "not the main scientific contribution",
        ],
    )
    add(
        checks,
        "scope_brief_central_claim",
        ok,
        "Editorial scope brief preserves the central posterior claim and depth-control boundary."
        if ok
        else "Missing scope-brief terms: " + ", ".join(missing),
    )

    forbidden_terms = [
        "control-point inversion",
        "control point inversion",
        "depth-control inversion",
        "control-depth inversion",
    ]
    forbidden_hits = [term for term in forbidden_terms if term in combined.casefold()]
    add(
        checks,
        "no_control_point_inversion_framing",
        not forbidden_hits,
        "No text frames the work as control-point inversion."
        if not forbidden_hits
        else "Forbidden framing terms: " + ", ".join(forbidden_hits),
    )

    posterior_terms = [
        "posterior",
        "bayesian",
        "conditional rectified flow",
        "probability-aware",
        "generative",
    ]
    control_terms = [
        "depth-control",
        "control depths",
        "sampling basis",
        "dense-output",
    ]
    posterior_mentions = count_terms(combined, posterior_terms)
    control_mentions = count_terms(combined, control_terms)
    add(
        checks,
        "posterior_term_dominance",
        posterior_mentions > control_mentions,
        f"Posterior-family terms ({posterior_mentions}) exceed control-basis terms ({control_mentions}).",
    )

    failures = [check for check in checks if not check.ok]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Audit central framing: amortized posterior sampling is the paper's scientific focus; surface-wave dispersion is the main demonstration and depth controls are an implementation choice.",
        "manuscript": str(manuscript_path),
        "cover_letter": str(cover_letter_path),
        "editorial_scope_brief": str(scope_brief_path),
        "metadata": str(metadata_path),
        "counts": {
            "posterior_family_mentions": posterior_mentions,
            "control_basis_mentions": control_mentions,
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
    parser = argparse.ArgumentParser(description="Audit central neural-posterior framing.")
    parser.add_argument("--root", default=".", help="Overleaf project root.")
    parser.add_argument("--manuscript", default="agujournaltemplate.tex")
    parser.add_argument("--cover-letter", default="cover_letter_jgrse.md")
    parser.add_argument("--editorial-scope-brief", default="editorial_scope_brief.md")
    parser.add_argument("--metadata", default="jgrse_submission_metadata.md")
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
