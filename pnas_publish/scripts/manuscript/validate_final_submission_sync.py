#!/usr/bin/env python3
"""Check that final submission values were propagated into support files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


PLACEHOLDER_RE = re.compile(
    r"(\[[^\]]+\]|"
    r"\b(?:required|tbd|todo|placeholder|pending|example\.com)\b|"
    r"zenodo\.[XY]+|"
    r"0000-0000-0000-0000|"
    r"/\[owner\]/\[repo\])",
    flags=re.I,
)

TARGET_FILES = {
    "manuscript": "agujournaltemplate.tex",
    "references": "references.bib",
    "cover_letter": "cover_letter_jgrse.md",
    "metadata": "jgrse_submission_metadata.md",
    "archive_manifest": "archive_manifest_zenodo.md",
    "zenodo_metadata": "zenodo_metadata_templates.json",
}


class Collector:
    def __init__(self, template_mode: bool) -> None:
        self.template_mode = template_mode
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def add(self, message: str) -> None:
        if self.template_mode:
            self.warnings.append(message)
        else:
            self.errors.append(message)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)


def is_placeholder(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if not value.strip():
        return True
    return bool(PLACEHOLDER_RE.search(value))


def clean(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def normalize(text: str) -> str:
    text = text.replace("\\%", "%")
    text = re.sub(r"\\([%&#_$])", r"\1", text)
    text = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\[A-Za-z]+\*?", " ", text)
    text = re.sub(r"[{}]", " ", text)
    text = text.replace("--", " ").replace("–", " ").replace("—", " ")
    text = re.sub(r"[^A-Za-z0-9]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip().casefold()


def value_in_text(value: str, text: str) -> bool:
    if not value:
        return False
    return normalize(value) in normalize(text)


def load_values(path: Path, collector: Collector) -> dict[str, Any]:
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        collector.error(f"Missing {path}.")
        return {}
    except json.JSONDecodeError as exc:
        collector.error(f"Invalid JSON in {path}: {exc}")
        return {}
    if not isinstance(values, dict):
        collector.error("Final submission values must be a JSON object.")
        return {}
    return values


def read_targets(root: Path, collector: Collector) -> dict[str, str]:
    targets: dict[str, str] = {}
    for key, filename in TARGET_FILES.items():
        path = root / filename
        try:
            targets[key] = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            collector.error(f"Missing target file {filename}.")
            targets[key] = ""
    return targets


def require_value(
    value: str,
    label: str,
    targets: dict[str, str],
    target_keys: list[str],
    collector: Collector,
) -> None:
    if is_placeholder(value):
        collector.add(f"{label} is still a placeholder in final values.")
        return
    missing = [key for key in target_keys if not value_in_text(value, targets.get(key, ""))]
    if missing:
        collector.add(f"{label} not found in " + ", ".join(TARGET_FILES[key] for key in missing) + f": {value}")


def validate_identity(values: dict[str, Any], targets: dict[str, str], collector: Collector) -> None:
    identity = values.get("manuscript_identity", {})
    if not isinstance(identity, dict):
        collector.error("manuscript_identity must be an object.")
        return
    title = clean(identity.get("title"))
    short_title = clean(identity.get("short_title"))
    require_value(title, "manuscript title", targets, ["manuscript", "metadata", "cover_letter", "zenodo_metadata"], collector)
    require_value(short_title, "short title/running head", targets, ["manuscript"], collector)

    corresponding = identity.get("corresponding_author", {})
    if not isinstance(corresponding, dict):
        collector.error("manuscript_identity.corresponding_author must be an object.")
        corresponding = {}
    corr_name = clean(corresponding.get("name"))
    corr_email = clean(corresponding.get("email"))
    corr_orcid = clean(corresponding.get("orcid"))
    require_value(corr_name, "corresponding author name", targets, ["manuscript", "cover_letter", "metadata"], collector)
    require_value(corr_email, "corresponding author email", targets, ["manuscript", "cover_letter", "metadata"], collector)
    require_value(corr_orcid, "corresponding author ORCID", targets, ["cover_letter", "metadata"], collector)

    authors = identity.get("authors", [])
    if not isinstance(authors, list) or not authors:
        collector.error("manuscript_identity.authors must contain at least one author.")
        return
    for index, author in enumerate(authors, start=1):
        if not isinstance(author, dict):
            collector.error(f"manuscript_identity.authors[{index}] must be an object.")
            continue
        name = clean(author.get("name"))
        email = clean(author.get("email"))
        require_value(name, f"author {index} name", targets, ["manuscript", "metadata", "zenodo_metadata"], collector)
        require_value(email, f"author {index} email", targets, ["metadata"], collector)

    affiliations = identity.get("affiliations", [])
    if not isinstance(affiliations, list) or not affiliations:
        collector.error("manuscript_identity.affiliations must contain at least one affiliation.")
        return
    for index, affiliation in enumerate(affiliations, start=1):
        if not isinstance(affiliation, dict):
            collector.error(f"manuscript_identity.affiliations[{index}] must be an object.")
            continue
        for key in ("institution", "city", "country"):
            value = clean(affiliation.get(key))
            require_value(value, f"affiliation {index} {key}", targets, ["manuscript", "metadata", "zenodo_metadata"], collector)


def validate_declarations(values: dict[str, Any], targets: dict[str, str], collector: Collector) -> None:
    declarations = values.get("contributions_and_declarations", {})
    if not isinstance(declarations, dict):
        collector.error("contributions_and_declarations must be an object.")
        return
    funding = clean(declarations.get("funding_statement"))
    acknowledgments = clean(declarations.get("acknowledgments"))
    require_value(funding, "funding statement", targets, ["manuscript", "metadata"], collector)
    require_value(acknowledgments, "acknowledgments", targets, ["manuscript"], collector)

    roles = declarations.get("credit_roles", [])
    if not isinstance(roles, list) or not roles:
        collector.error("contributions_and_declarations.credit_roles must contain at least one entry.")
        return
    for index, entry in enumerate(roles, start=1):
        if not isinstance(entry, dict):
            collector.error(f"credit_roles[{index}] must be an object.")
            continue
        author = clean(entry.get("author"))
        require_value(author, f"CRediT author {index}", targets, ["manuscript"], collector)
        for role in entry.get("roles", []):
            require_value(clean(role), f"CRediT role for {author or index}", targets, ["manuscript"], collector)


def validate_open_research(values: dict[str, Any], targets: dict[str, str], collector: Collector) -> None:
    open_research = values.get("open_research", {})
    if not isinstance(open_research, dict):
        collector.error("open_research must be an object.")
        return
    software_doi = clean(open_research.get("software_doi"))
    data_doi = clean(open_research.get("data_output_doi"))
    repo_url = clean(open_research.get("public_repository_url"))
    code_license = clean(open_research.get("code_license"))
    data_license = clean(open_research.get("data_output_license"))
    require_value(software_doi, "software DOI", targets, ["manuscript", "references", "metadata", "zenodo_metadata"], collector)
    require_value(data_doi, "data/output DOI", targets, ["manuscript", "references", "metadata", "zenodo_metadata"], collector)
    require_value(repo_url, "public repository URL", targets, ["manuscript", "metadata", "zenodo_metadata"], collector)
    require_value(code_license, "code license", targets, ["metadata", "archive_manifest", "zenodo_metadata"], collector)
    require_value(data_license, "data/output license", targets, ["metadata", "archive_manifest", "zenodo_metadata"], collector)


def validate_reviewers(values: dict[str, Any], targets: dict[str, str], collector: Collector) -> None:
    reviewers = values.get("reviewers", {})
    if not isinstance(reviewers, dict):
        collector.error("reviewers must be an object.")
        return
    suggested = reviewers.get("suggested", [])
    if not isinstance(suggested, list) or len(suggested) < 3:
        collector.error("reviewers.suggested must contain at least three reviewers.")
        return
    for index, reviewer in enumerate(suggested, start=1):
        if not isinstance(reviewer, dict):
            collector.error(f"reviewers.suggested[{index}] must be an object.")
            continue
        for key in ("name", "institution", "email", "expertise_fit", "conflict_check"):
            require_value(clean(reviewer.get(key)), f"suggested reviewer {index} {key}", targets, ["metadata"], collector)

    excluded = reviewers.get("excluded", [])
    if isinstance(excluded, list):
        for index, reviewer in enumerate(excluded, start=1):
            if not isinstance(reviewer, dict):
                collector.error(f"reviewers.excluded[{index}] must be an object.")
                continue
            values_to_check = [clean(reviewer.get(key)) for key in ("name", "institution", "reason")]
            if any(value and not is_placeholder(value) for value in values_to_check):
                for key, value in zip(("name", "institution", "reason"), values_to_check):
                    require_value(value, f"excluded reviewer {index} {key}", targets, ["metadata"], collector)


def validate_scope(values: dict[str, Any], targets: dict[str, str], collector: Collector) -> None:
    scope = values.get("scope_decision", {})
    if not isinstance(scope, dict):
        collector.error("scope_decision must be an object.")
        return
    calibration = clean(scope.get("calibration_decision"))
    field_description = clean(scope.get("field_example_description"))
    require_value(calibration, "calibration decision", targets, ["metadata", "cover_letter"], collector)
    if scope.get("add_field_example") is True:
        require_value(field_description, "field example description", targets, ["manuscript", "cover_letter", "metadata"], collector)
    if scope.get("submit_current_synthetic_plus_ak135_scope") is True:
        for needle in ("synthetic", "ak135", "Bayan Obo"):
            if not value_in_text(needle, targets.get("cover_letter", "")):
                collector.add(f"Cover letter should state the current {needle} scope decision.")


def validate(values: dict[str, Any], targets: dict[str, str], template_mode: bool) -> dict[str, Any]:
    collector = Collector(template_mode=template_mode)
    validate_identity(values, targets, collector)
    validate_declarations(values, targets, collector)
    validate_open_research(values, targets, collector)
    validate_reviewers(values, targets, collector)
    validate_scope(values, targets, collector)
    return {
        "ok": not collector.errors,
        "mode": "template" if template_mode else "final",
        "errors": collector.errors,
        "warnings": collector.warnings,
        "target_files": TARGET_FILES,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether final values are synchronized into submission files.")
    parser.add_argument("path", nargs="?", help="Final values JSON file.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Overleaf project root.")
    parser.add_argument(
        "--template",
        action="store_true",
        help="Run against the committed template and report missing sync as warnings.",
    )
    args = parser.parse_args()

    values_path = Path(args.path or ("final_submission_values.template.json" if args.template else "final_submission_values.json"))
    collector = Collector(template_mode=args.template)
    values = load_values(args.root / values_path if not values_path.is_absolute() else values_path, collector)
    targets = read_targets(args.root, collector)
    if collector.errors:
        result = {
            "ok": False,
            "mode": "template" if args.template else "final",
            "errors": collector.errors,
            "warnings": collector.warnings,
            "target_files": TARGET_FILES,
        }
    else:
        result = validate(values, targets, template_mode=args.template)
        result["path"] = str(values_path)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
