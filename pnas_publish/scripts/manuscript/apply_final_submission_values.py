#!/usr/bin/env python3
"""Apply final JGR:SE submission values to manuscript support files."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from validate_final_submission_values import validate as validate_final_values


TARGET_FILES = [
    "agujournaltemplate.tex",
    "references.bib",
    "cover_letter_jgrse.md",
    "jgrse_submission_metadata.md",
    "archive_manifest_zenodo.md",
    "zenodo_metadata_templates.json",
]


@dataclass
class FileUpdate:
    path: Path
    before: str
    after: str

    @property
    def changed(self) -> bool:
        return self.before != self.after


def load_values(path: Path) -> dict[str, Any]:
    try:
        values = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"Missing {path}.")
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(values, dict):
        raise SystemExit("Final submission values must be a JSON object.")
    return values


def latex_escape(value: Any) -> str:
    text = "" if value is None else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def md(value: Any) -> str:
    return "" if value is None else str(value).strip()


def doi_url(doi: str) -> str:
    doi = doi.strip()
    return doi if doi.startswith(("http://", "https://")) else f"https://doi.org/{doi}"


def author_names(values: dict[str, Any]) -> list[str]:
    authors = values["manuscript_identity"]["authors"]
    return [md(author.get("name")) for author in authors if isinstance(author, dict)]


def bib_author_list(values: dict[str, Any]) -> str:
    return " and ".join(author_names(values))


def compact_affiliation(affiliation: dict[str, Any]) -> str:
    parts = [md(affiliation.get(key)) for key in ("institution", "city", "country")]
    return ", ".join(part for part in parts if part)


def find_command_span(text: str, command: str, arg_count: int) -> tuple[int, int]:
    match = re.search(rf"(?<![A-Za-z{{])\\{re.escape(command)}\b", text)
    if not match:
        raise ValueError(f"Missing LaTeX command \\{command}.")
    pos = match.end()
    for _ in range(arg_count):
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text) or text[pos] != "{":
            raise ValueError(f"Command \\{command} does not have {arg_count} brace argument(s).")
        depth = 0
        pos += 1
        while pos < len(text):
            char = text[pos]
            previous = text[pos - 1] if pos else ""
            if char == "{" and previous != "\\":
                depth += 1
            elif char == "}" and previous != "\\":
                if depth == 0:
                    pos += 1
                    break
                depth -= 1
            pos += 1
        else:
            raise ValueError(f"Unclosed brace group in \\{command}.")
    return match.start(), pos


def replace_latex_command(text: str, command: str, replacement: str, arg_count: int = 1) -> str:
    start, end = find_command_span(text, command, arg_count)
    return text[:start] + replacement + text[end:]


def replace_section(text: str, heading: str, body: str) -> str:
    pattern = re.compile(
        rf"(\\section\*\{{{re.escape(heading)}\}}\n)(.*?)(?=\n\\section\*?\{{|\n\\acknowledgments|\n\\bibliography|\n\\end\{{document\}})",
        flags=re.S,
    )
    new_text, count = pattern.subn(lambda match: match.group(1) + body.strip() + "\n", text, count=1)
    if count != 1:
        raise ValueError(f"Could not replace section {heading}.")
    return new_text


def replace_acknowledgments(text: str, acknowledgments: str) -> str:
    pattern = re.compile(r"(\\acknowledgments\n)(.*?)(?=\n\\bibliography)", flags=re.S)
    new_text, count = pattern.subn(
        lambda match: match.group(1) + latex_escape(acknowledgments).strip() + "\n",
        text,
        count=1,
    )
    if count != 1:
        raise ValueError("Could not replace acknowledgments block.")
    return new_text


def render_authors(values: dict[str, Any]) -> tuple[str, str, str, str]:
    identity = values["manuscript_identity"]
    authors = identity["authors"]
    affiliations = identity["affiliations"]
    corresponding = identity["corresponding_author"]

    author_items = []
    for author in authors:
        affils = "".join(rf"\affil{{{latex_escape(affil_id)}}}" for affil_id in author.get("affiliation_ids", []))
        author_items.append(f"{latex_escape(author['name'])}{affils}")
    authors_command = r"\authors{" + ", ".join(author_items) + "}"

    affiliation_lines = [
        rf"\affiliation{{{latex_escape(item['id'])}}}{{{latex_escape(compact_affiliation(item))}}}"
        for item in affiliations
    ]

    corr_name = latex_escape(corresponding["name"])
    corr_email = latex_escape(corresponding["email"])
    corr_command = rf"\correspondingauthor{{{corr_name}}}{{{corr_email}}}"

    affiliation_lookup = {md(item.get("id")): compact_affiliation(item) for item in affiliations}
    corr_affiliations = [
        affiliation_lookup.get(md(affil_id), "")
        for affil_id in corresponding.get("affiliation_ids", [])
    ]
    corr_affiliation_text = "; ".join(item for item in corr_affiliations if item)
    author_addr = rf"\authoraddr{{{corr_name}, {latex_escape(corr_affiliation_text)}. ({corr_email})}}"
    return authors_command, "\n".join(affiliation_lines), corr_command, author_addr


def update_manuscript(text: str, values: dict[str, Any]) -> str:
    identity = values["manuscript_identity"]
    declarations = values["contributions_and_declarations"]
    open_research = values["open_research"]
    authors_command, affiliation_block, corr_command, author_addr = render_authors(values)

    text = replace_latex_command(text, "title", rf"\title{{{latex_escape(identity['title'])}}}")
    text = replace_latex_command(text, "authors", authors_command)
    _, authors_end = find_command_span(text, "authors", 1)
    corr_start = text.index(r"\correspondingauthor", authors_end)
    text = text[:authors_end] + "\n\n" + affiliation_block + "\n\n" + text[corr_start:]
    text = replace_latex_command(text, "correspondingauthor", corr_command, arg_count=2)
    text = replace_latex_command(text, "authoraddr", author_addr)
    names = author_names(values)
    author_running = names[0] if len(names) == 1 else f"{names[0]} et al."
    text = replace_latex_command(text, "authorrunninghead", rf"\authorrunninghead{{{latex_escape(author_running)}}}")
    text = replace_latex_command(text, "titlerunninghead", rf"\titlerunninghead{{{latex_escape(identity['short_title'])}}}")

    software_doi = md(open_research["software_doi"])
    data_doi = md(open_research["data_output_doi"])
    repo_url = md(open_research["public_repository_url"])
    open_research_body = (
        "The software, trained checkpoints, training logs, and figure-generation scripts used in this study "
        rf"are preserved at Zenodo with DOI \texttt{{{latex_escape(software_doi)}}} "
        rf"\cite{{PosteriorDispersionSoftware2026}} and developed at \texttt{{{latex_escape(repo_url)}}}. "
        "Synthetic evaluation outputs, manuscript figures, posterior-sample artifacts, and numerical metrics "
        rf"are preserved at Zenodo with DOI \texttt{{{latex_escape(data_doi)}}} "
        r"\cite{PosteriorDispersionData2026}. "
        "The evaluation set is generated from the synthetic model prior described in the Methods section using seed 2026. "
        "No field observations or restricted third-party data were used in the current validation."
    )
    text = replace_section(text, "Open Research Statement", open_research_body)

    roles = []
    for entry in declarations["credit_roles"]:
        role_text = ", ".join(md(role) for role in entry.get("roles", []))
        roles.append(f"{latex_escape(entry['author'])}: {latex_escape(role_text)}.")
    text = replace_section(text, "Author Contributions", " ".join(roles))
    acknowledgment_parts = [md(declarations["funding_statement"]), md(declarations["acknowledgments"])]
    text = replace_acknowledgments(text, " ".join(part for part in acknowledgment_parts if part))
    return text


def bib_entry(key: str, values: dict[str, Any], doi: str, title: str, kind: str, version: str) -> str:
    return (
        f"@misc{{{key},\n"
        f"  author = {{{bib_author_list(values)}}},\n"
        "  year = {2026},\n"
        f"  title = {{{title} [{kind}]}},\n"
        f"  version = {{{version}}},\n"
        "  publisher = {Zenodo},\n"
        f"  doi = {{{doi}}},\n"
        f"  url = {{{doi_url(doi)}}}\n"
        "}\n"
    )


def upsert_bib_entry(text: str, key: str, entry: str) -> str:
    pattern = re.compile(rf"@misc\{{{re.escape(key)},.*?\n\}}\n?", flags=re.S)
    if pattern.search(text):
        return pattern.sub(entry + "\n", text, count=1)
    return text.rstrip() + "\n\n" + entry


def update_references(text: str, values: dict[str, Any]) -> str:
    open_research = values["open_research"]
    software_entry = bib_entry(
        "PosteriorDispersionSoftware2026",
        values,
        md(open_research["software_doi"]),
        "Amortized posterior sampling for surface-wave dispersion with conditional rectified flow",
        "Software",
        md(open_research["software_record_version"]),
    )
    data_entry = bib_entry(
        "PosteriorDispersionData2026",
        values,
        md(open_research["data_output_doi"]),
        "Synthetic surface-wave evaluation outputs for amortized posterior sampling",
        "Dataset",
        md(open_research["data_output_record_version"]),
    )
    text = upsert_bib_entry(text, "PosteriorDispersionSoftware2026", software_entry)
    text = upsert_bib_entry(text, "PosteriorDispersionData2026", data_entry)
    return text


def scope_sentence(values: dict[str, Any]) -> str:
    scope = values["scope_decision"]
    if scope.get("add_field_example") is True:
        return "The manuscript includes the selected field example: " + md(scope.get("field_example_description"))
    return "The authors confirm that the submitted manuscript is framed as amortized posterior sampling demonstrated with synthetic validation, an ak135 standard-model stress test, and an uncalibrated Bayan Obo field-data stress test."


def update_cover_letter(text: str, values: dict[str, Any]) -> str:
    identity = values["manuscript_identity"]
    declarations = values["contributions_and_declarations"]
    corresponding = identity["corresponding_author"]
    affiliations = {md(item.get("id")): compact_affiliation(item) for item in identity["affiliations"]}
    corr_affiliation = "; ".join(
        affiliations.get(md(affil_id), "") for affil_id in corresponding.get("affiliation_ids", [])
    ).strip("; ")
    final_paragraph = (
        "All authors have approved the manuscript and agree with its submission to *JGR: Solid Earth*. "
        f"{scope_sentence(values)} "
        f"Calibration decision: {md(values['scope_decision']['calibration_decision'])}. "
        f"Funding and acknowledgments are finalized as: {md(declarations['funding_statement'])}"
    )
    text = re.sub(
        r"The final submission should include complete author information.*?or have added a real or semi-real example\.",
        final_paragraph,
        text,
        count=1,
        flags=re.S,
    )
    signature = (
        f"{md(corresponding['name'])}  \n"
        f"{corr_affiliation}  \n"
        f"{md(corresponding['email'])}  \n"
        f"{md(corresponding['orcid'])}"
    )
    text = re.sub(
        r"\[Corresponding Author Name\]\s*\n\[Affiliation\]\s*\n\[Email\]\s*\n\[ORCID\]",
        signature,
        text,
        count=1,
    )
    return text


def render_final_values_snapshot(values: dict[str, Any]) -> str:
    identity = values["manuscript_identity"]
    declarations = values["contributions_and_declarations"]
    open_research = values["open_research"]
    reviewers = values["reviewers"]
    scope = values["scope_decision"]
    lines = ["## Final Values Snapshot", ""]
    lines.extend(
        [
            f"- Target journal: {md(identity['target_journal'])}",
            f"- Manuscript type: {md(identity['manuscript_type'])}",
            f"- Corresponding author: {md(identity['corresponding_author']['name'])}",
            f"- Corresponding author email: {md(identity['corresponding_author']['email'])}",
            f"- Corresponding author ORCID: {md(identity['corresponding_author']['orcid'])}",
            f"- Funding statement: {md(declarations['funding_statement'])}",
            f"- Acknowledgments: {md(declarations['acknowledgments'])}",
            f"- Software DOI: {md(open_research['software_doi'])}",
            f"- Data/output DOI: {md(open_research['data_output_doi'])}",
            f"- Public repository URL: {md(open_research['public_repository_url'])}",
            f"- Code license: {md(open_research['code_license'])}",
            f"- Data/output license: {md(open_research['data_output_license'])}",
            f"- Software record version: {md(open_research['software_record_version'])}",
            f"- Data/output record version: {md(open_research['data_output_record_version'])}",
            f"- Calibration decision: {md(scope['calibration_decision'])}",
            f"- Scope decision: {scope_sentence(values)}",
            "",
            "Authors:",
        ]
    )
    for author in identity["authors"]:
        lines.append(
            f"- {md(author['name'])}; email: {md(author['email'])}; ORCID: {md(author.get('orcid', ''))}; affiliations: {', '.join(author.get('affiliation_ids', []))}"
        )
    lines.extend(["", "Affiliations:"])
    for affiliation in identity["affiliations"]:
        lines.append(f"- {md(affiliation['id'])}: {compact_affiliation(affiliation)}")
    lines.extend(["", "CRediT roles:"])
    for entry in declarations["credit_roles"]:
        lines.append(f"- {md(entry['author'])}: {', '.join(md(role) for role in entry.get('roles', []))}")
    lines.extend(["", "Suggested reviewers:"])
    for reviewer in reviewers["suggested"]:
        lines.append(
            f"- {md(reviewer['name'])}; {md(reviewer['institution'])}; {md(reviewer['email'])}; "
            f"{md(reviewer['expertise_fit'])}; {md(reviewer['conflict_check'])}"
        )
    excluded = reviewers.get("excluded", [])
    if excluded:
        lines.extend(["", "Excluded reviewers:"])
        for reviewer in excluded:
            if any(md(reviewer.get(key)) for key in ("name", "institution", "reason")):
                lines.append(f"- {md(reviewer.get('name'))}; {md(reviewer.get('institution'))}; {md(reviewer.get('reason'))}")
    return "\n".join(lines).rstrip() + "\n"


def upsert_markdown_section(text: str, heading: str, body: str, before_heading: str | None = None) -> str:
    pattern = re.compile(rf"^## {re.escape(heading)}\s*$\n.*?(?=^## |\Z)", flags=re.M | re.S)
    if pattern.search(text):
        return pattern.sub(body + "\n", text, count=1)
    if before_heading:
        marker = f"## {before_heading}"
        if marker in text:
            return text.replace(marker, body + "\n" + marker, 1)
    return text.rstrip() + "\n\n" + body


def update_metadata(text: str, values: dict[str, Any]) -> str:
    return upsert_markdown_section(text, "Final Values Snapshot", render_final_values_snapshot(values), before_heading="Title")


def update_archive_manifest(text: str, values: dict[str, Any]) -> str:
    open_research = values["open_research"]
    license_sentence = (
        f"- Selected code license: {md(open_research['code_license'])}. "
        f"Selected data/output license: {md(open_research['data_output_license'])}."
    )
    if "Selected code license:" in text:
        text = re.sub(r"^- Selected code license:.*$", license_sentence, text, count=1, flags=re.M)
    else:
        text = text.replace(
            "- Add a license before archiving. A permissive code license and a clear data license should be selected by the author team.",
            "- Add a license before archiving. A permissive code license and a clear data license should be selected by the author team.\n" + license_sentence,
            1,
        )
    return text


def creator_records(values: dict[str, Any]) -> list[dict[str, str]]:
    identity = values["manuscript_identity"]
    affiliations = {md(item.get("id")): compact_affiliation(item) for item in identity["affiliations"]}
    creators = []
    for author in identity["authors"]:
        author_affiliations = [
            affiliations.get(md(affil_id), "")
            for affil_id in author.get("affiliation_ids", [])
        ]
        creators.append(
            {
                "name": md(author.get("name")),
                "orcid": md(author.get("orcid")),
                "affiliation": "; ".join(item for item in author_affiliations if item),
            }
        )
    return creators


def upsert_related_identifier(record: dict[str, Any], relation: str, identifier: str, scheme: str) -> None:
    related = record.setdefault("related_identifiers", [])
    if not isinstance(related, list):
        record["related_identifiers"] = []
        related = record["related_identifiers"]
    for item in related:
        if isinstance(item, dict) and item.get("relation") == relation and item.get("scheme") == scheme:
            item["identifier"] = identifier
            return
    related.append({"relation": relation, "identifier": identifier, "scheme": scheme})


def update_zenodo_metadata(text: str, values: dict[str, Any]) -> str:
    try:
        metadata = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse zenodo_metadata_templates.json: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError("zenodo_metadata_templates.json must contain a JSON object.")

    open_research = values["open_research"]
    creators = creator_records(values)
    public_records = metadata.setdefault("public_records", {})
    if not isinstance(public_records, dict):
        raise ValueError("zenodo_metadata_templates.json public_records must be an object.")

    software = public_records.setdefault("software_record", {})
    data_output = public_records.setdefault("data_output_record", {})
    if not isinstance(software, dict) or not isinstance(data_output, dict):
        raise ValueError("software_record and data_output_record must be objects.")

    software.update(
        {
            "version": md(open_research["software_record_version"]),
            "doi": md(open_research["software_doi"]),
            "license": md(open_research["code_license"]),
            "creators": creators,
        }
    )
    data_output.update(
        {
            "version": md(open_research["data_output_record_version"]),
            "doi": md(open_research["data_output_doi"]),
            "license": md(open_research["data_output_license"]),
            "creators": creators,
        }
    )
    repo_url = md(open_research["public_repository_url"])
    upsert_related_identifier(software, "isSourceOf", md(open_research["data_output_doi"]), "doi")
    upsert_related_identifier(software, "isDocumentedBy", repo_url, "url")
    upsert_related_identifier(data_output, "isDerivedFrom", md(open_research["software_doi"]), "doi")
    upsert_related_identifier(data_output, "isDocumentedBy", repo_url, "url")
    return json.dumps(metadata, indent=2) + "\n"


def build_updates(root: Path, values: dict[str, Any]) -> list[FileUpdate]:
    transforms = {
        "agujournaltemplate.tex": update_manuscript,
        "references.bib": update_references,
        "cover_letter_jgrse.md": update_cover_letter,
        "jgrse_submission_metadata.md": update_metadata,
        "archive_manifest_zenodo.md": update_archive_manifest,
        "zenodo_metadata_templates.json": update_zenodo_metadata,
    }
    updates: list[FileUpdate] = []
    for filename in TARGET_FILES:
        path = root / filename
        before = path.read_text(encoding="utf-8")
        after = transforms[filename](before, values)
        updates.append(FileUpdate(path=path, before=before, after=after))
    return updates


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply final JGR:SE submission values to target files.")
    parser.add_argument("path", nargs="?", help="Final values JSON file.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Overleaf project root.")
    parser.add_argument("--template", action="store_true", help="Load the committed template; dry-run only.")
    parser.add_argument("--dry-run", action="store_true", help="Report planned updates without writing files.")
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    args = parser.parse_args()

    if args.template and not args.dry_run:
        raise SystemExit("--template can only be used with --dry-run.")

    root = args.root.resolve()
    values_path = Path(args.path or ("final_submission_values.template.json" if args.template else "final_submission_values.json"))
    values = load_values(root / values_path if not values_path.is_absolute() else values_path)
    validation = validate_final_values(values, template_mode=args.template)
    if not validation["ok"]:
        print(json.dumps(validation, indent=2))
        return 1

    try:
        updates = build_updates(root, values)
    except (KeyError, ValueError) as exc:
        raise SystemExit(f"Could not apply final values: {exc}") from exc

    changed = [update for update in updates if update.changed]
    if not args.dry_run:
        for update in changed:
            update.path.write_text(update.after, encoding="utf-8")

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "values_path": str(values_path),
        "dry_run": args.dry_run,
        "template_mode": args.template,
        "ok": True,
        "changed_files": [str(update.path) for update in changed],
        "unchanged_files": [str(update.path) for update in updates if not update.changed],
        "validation_warnings": validation.get("warnings", []),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
