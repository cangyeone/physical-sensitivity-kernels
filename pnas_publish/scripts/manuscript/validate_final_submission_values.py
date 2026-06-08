#!/usr/bin/env python3
"""Validate final human-provided values before JGR:SE submission."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


REQUIRED_SECTIONS = [
    "schema_version",
    "manuscript_identity",
    "contributions_and_declarations",
    "open_research",
    "reviewers",
    "scope_decision",
]

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
ORCID_RE = re.compile(r"^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$")
DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$")
URL_RE = re.compile(r"^https?://\S+$")
PLACEHOLDER_RE = re.compile(
    r"(\[[^\]]+\]|"
    r"\b(?:required|tbd|todo|placeholder|pending|example\.com)\b|"
    r"zenodo\.[XY]+|"
    r"0000-0000-0000-0000|"
    r"/\[owner\]/\[repo\])",
    flags=re.I,
)


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


def get_mapping(parent: dict[str, Any], key: str, collector: Collector) -> dict[str, Any]:
    value = parent.get(key)
    if isinstance(value, dict):
        return value
    collector.error(f"{key} must be an object.")
    return {}


def get_list(parent: dict[str, Any], key: str, collector: Collector) -> list[Any]:
    value = parent.get(key)
    if isinstance(value, list):
        return value
    collector.error(f"{key} must be a list.")
    return []


def require_text(data: dict[str, Any], key: str, path: str, collector: Collector) -> str:
    value = data.get(key)
    if not isinstance(value, str) or is_placeholder(value):
        collector.add(f"{path}.{key} must be filled with a non-placeholder value.")
        return "" if value is None else str(value)
    return value.strip()


def optional_text(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    return value.strip() if isinstance(value, str) else ""


def require_true(data: dict[str, Any], key: str, path: str, collector: Collector) -> None:
    if data.get(key) is not True:
        collector.add(f"{path}.{key} must be true before submission.")


def validate_email(value: str, path: str, collector: Collector) -> None:
    if value and not EMAIL_RE.match(value):
        collector.add(f"{path} must be a valid email address.")


def validate_orcid(value: str, path: str, collector: Collector) -> None:
    if value and not ORCID_RE.match(value):
        collector.add(f"{path} must use the ORCID form 0000-0000-0000-0000.")


def validate_doi(value: str, path: str, collector: Collector) -> None:
    if value and not DOI_RE.match(value):
        collector.add(f"{path} must look like a DOI, for example 10.5281/zenodo.1234567.")


def validate_url(value: str, path: str, collector: Collector) -> None:
    if value and not URL_RE.match(value):
        collector.add(f"{path} must start with http:// or https://.")


def validate_identity(values: dict[str, Any], collector: Collector) -> None:
    identity = get_mapping(values, "manuscript_identity", collector)
    require_text(identity, "target_journal", "manuscript_identity", collector)
    require_text(identity, "manuscript_type", "manuscript_identity", collector)
    require_text(identity, "title", "manuscript_identity", collector)
    require_text(identity, "short_title", "manuscript_identity", collector)

    corresponding = get_mapping(identity, "corresponding_author", collector)
    corresponding_name = require_text(corresponding, "name", "manuscript_identity.corresponding_author", collector)
    corresponding_email = require_text(corresponding, "email", "manuscript_identity.corresponding_author", collector)
    corresponding_orcid = require_text(corresponding, "orcid", "manuscript_identity.corresponding_author", collector)
    validate_email(corresponding_email, "manuscript_identity.corresponding_author.email", collector)
    validate_orcid(corresponding_orcid, "manuscript_identity.corresponding_author.orcid", collector)
    if not get_list(corresponding, "affiliation_ids", collector):
        collector.add("manuscript_identity.corresponding_author.affiliation_ids must list at least one affiliation id.")

    authors = get_list(identity, "authors", collector)
    if not authors:
        collector.error("manuscript_identity.authors must list at least one author.")
    author_names = []
    for index, item in enumerate(authors, start=1):
        if not isinstance(item, dict):
            collector.error(f"manuscript_identity.authors[{index}] must be an object.")
            continue
        path = f"manuscript_identity.authors[{index}]"
        name = require_text(item, "name", path, collector)
        if name:
            author_names.append(name)
        email = require_text(item, "email", path, collector)
        validate_email(email, f"{path}.email", collector)
        orcid = optional_text(item, "orcid")
        if orcid and is_placeholder(orcid):
            collector.add(f"{path}.orcid must be replaced or removed if unavailable.")
        elif orcid:
            validate_orcid(orcid, f"{path}.orcid", collector)
        if not get_list(item, "affiliation_ids", collector):
            collector.add(f"{path}.affiliation_ids must list at least one affiliation id.")

    if corresponding_name and author_names and corresponding_name not in author_names:
        collector.warn("Corresponding author name is not an exact match to any manuscript_identity.authors entry.")

    affiliations = get_list(identity, "affiliations", collector)
    if not affiliations:
        collector.error("manuscript_identity.affiliations must list at least one affiliation.")
    for index, item in enumerate(affiliations, start=1):
        if not isinstance(item, dict):
            collector.error(f"manuscript_identity.affiliations[{index}] must be an object.")
            continue
        path = f"manuscript_identity.affiliations[{index}]"
        require_text(item, "id", path, collector)
        require_text(item, "institution", path, collector)
        require_text(item, "city", path, collector)
        require_text(item, "country", path, collector)


def validate_contributions(values: dict[str, Any], collector: Collector) -> None:
    section = get_mapping(values, "contributions_and_declarations", collector)
    roles = get_list(section, "credit_roles", collector)
    if not roles:
        collector.add("contributions_and_declarations.credit_roles must list author CRediT roles.")
    for index, item in enumerate(roles, start=1):
        if not isinstance(item, dict):
            collector.error(f"contributions_and_declarations.credit_roles[{index}] must be an object.")
            continue
        path = f"contributions_and_declarations.credit_roles[{index}]"
        require_text(item, "author", path, collector)
        role_values = get_list(item, "roles", collector)
        if not role_values:
            collector.add(f"{path}.roles must contain at least one CRediT role.")
        for role_index, role in enumerate(role_values, start=1):
            if not isinstance(role, str) or is_placeholder(role):
                collector.add(f"{path}.roles[{role_index}] must be a non-placeholder CRediT role.")

    require_text(section, "funding_statement", "contributions_and_declarations", collector)
    require_text(section, "acknowledgments", "contributions_and_declarations", collector)
    require_true(section, "conflict_statement_confirmed", "contributions_and_declarations", collector)
    require_true(section, "ai_disclosure_confirmed", "contributions_and_declarations", collector)
    require_true(section, "originality_confirmed", "contributions_and_declarations", collector)
    require_true(section, "all_authors_approved", "contributions_and_declarations", collector)


def validate_open_research(values: dict[str, Any], collector: Collector) -> None:
    section = get_mapping(values, "open_research", collector)
    software_doi = require_text(section, "software_doi", "open_research", collector)
    data_doi = require_text(section, "data_output_doi", "open_research", collector)
    repo_url = require_text(section, "public_repository_url", "open_research", collector)
    validate_doi(software_doi, "open_research.software_doi", collector)
    validate_doi(data_doi, "open_research.data_output_doi", collector)
    validate_url(repo_url, "open_research.public_repository_url", collector)
    require_text(section, "code_license", "open_research", collector)
    require_text(section, "data_output_license", "open_research", collector)
    require_text(section, "software_record_version", "open_research", collector)
    require_text(section, "data_output_record_version", "open_research", collector)


def validate_reviewers(values: dict[str, Any], collector: Collector) -> None:
    section = get_mapping(values, "reviewers", collector)
    suggested = get_list(section, "suggested", collector)
    if len(suggested) < 3:
        collector.add("reviewers.suggested must contain at least three suggested reviewers.")
    for index, item in enumerate(suggested, start=1):
        if not isinstance(item, dict):
            collector.error(f"reviewers.suggested[{index}] must be an object.")
            continue
        path = f"reviewers.suggested[{index}]"
        require_text(item, "name", path, collector)
        require_text(item, "institution", path, collector)
        email = require_text(item, "email", path, collector)
        validate_email(email, f"{path}.email", collector)
        require_text(item, "expertise_fit", path, collector)
        require_text(item, "conflict_check", path, collector)

    excluded = get_list(section, "excluded", collector)
    for index, item in enumerate(excluded, start=1):
        if not isinstance(item, dict):
            collector.error(f"reviewers.excluded[{index}] must be an object.")
            continue
        path = f"reviewers.excluded[{index}]"
        name = optional_text(item, "name")
        institution = optional_text(item, "institution")
        reason = optional_text(item, "reason")
        if any([name, institution, reason]) and not all([name, institution, reason]):
            collector.add(f"{path} must include name, institution, and reason, or be removed.")
        if any(is_placeholder(value) for value in [name, institution, reason]):
            collector.add(f"{path} must remove optional placeholder text or provide a factual exclusion.")


def validate_scope(values: dict[str, Any], collector: Collector) -> None:
    section = get_mapping(values, "scope_decision", collector)
    current_scope = section.get("submit_current_synthetic_plus_ak135_scope")
    add_field = section.get("add_field_example")
    if current_scope is not True and add_field is not True:
        collector.add(
            "scope_decision must either confirm the current synthetic, ak135, and Bayan Obo stress-test scope or mark add_field_example true for stronger field calibration."
        )
    if add_field is True:
        require_text(section, "field_example_description", "scope_decision", collector)
    require_text(section, "calibration_decision", "scope_decision", collector)
    require_true(section, "known_scope_limitations_acknowledged", "scope_decision", collector)


def validate(values: dict[str, Any], template_mode: bool) -> dict[str, Any]:
    collector = Collector(template_mode=template_mode)
    if not isinstance(values, dict):
        collector.error("Top-level JSON value must be an object.")
        return {"ok": False, "errors": collector.errors, "warnings": collector.warnings}

    missing_sections = [section for section in REQUIRED_SECTIONS if section not in values]
    if missing_sections:
        collector.error("Missing required sections: " + ", ".join(missing_sections))
    for section in REQUIRED_SECTIONS:
        if section == "schema_version":
            continue
        if section in values and not isinstance(values[section], dict):
            collector.error(f"{section} must be an object.")

    validate_identity(values, collector)
    validate_contributions(values, collector)
    validate_open_research(values, collector)
    validate_reviewers(values, collector)
    validate_scope(values, collector)

    return {
        "ok": not collector.errors,
        "mode": "template" if template_mode else "final",
        "required_sections": REQUIRED_SECTIONS,
        "errors": collector.errors,
        "warnings": collector.warnings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate final JGR:SE submission values.")
    parser.add_argument("path", nargs="?", help="JSON file to validate.")
    parser.add_argument(
        "--template",
        action="store_true",
        help="Validate template shape while allowing placeholder values as warnings.",
    )
    args = parser.parse_args()

    path = Path(args.path or ("final_submission_values.template.json" if args.template else "final_submission_values.json"))
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        result = {
            "ok": False,
            "mode": "template" if args.template else "final",
            "errors": [f"Missing {path}."],
            "warnings": [],
        }
        print(json.dumps(result, indent=2))
        return 1
    except json.JSONDecodeError as exc:
        result = {
            "ok": False,
            "mode": "template" if args.template else "final",
            "errors": [f"Invalid JSON in {path}: {exc}"],
            "warnings": [],
        }
        print(json.dumps(result, indent=2))
        return 1

    result = validate(values, template_mode=args.template)
    result["path"] = str(path)
    print(json.dumps(result, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
