#!/usr/bin/env python3
"""Estimate AGU publication units for the manuscript.

This is a deterministic local preflight for the AGU submission-system count.
It follows AGU's publication-unit rule as closely as possible from LaTeX
source: publication units = words / 500 + figures + tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SCRIPTS = Path(__file__).resolve().parent
OVERLEAF = SCRIPTS.parent

GUIDANCE_URLS = [
    "https://www.agu.org/Publications/Authors/Journals",
    "https://www.agu.org/publications/authors/journals/text-graphics-requirements",
    "https://www.agu.org/publications/authors/journals/submission-checklists",
]


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        cut = None
        for index, char in enumerate(line):
            if char == "%" and (index == 0 or line[index - 1] != "\\"):
                cut = index
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


def remove_envs(text: str, envs: Iterable[str]) -> str:
    for env in envs:
        text = remove_env(text, env)
    return text


def matching_delimiter(text: str, start: int, opener: str, closer: str) -> int | None:
    if start >= len(text) or text[start] != opener:
        return None
    depth = 0
    pos = start + 1
    while pos < len(text):
        char = text[pos]
        escaped = pos > 0 and text[pos - 1] == "\\"
        if char == opener and not escaped:
            depth += 1
        elif char == closer and not escaped:
            if depth == 0:
                return pos
            depth -= 1
        pos += 1
    return None


def skip_optional_arguments(text: str, pos: int) -> int:
    while True:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text) or text[pos] != "[":
            return pos
        end = matching_delimiter(text, pos, "[", "]")
        if end is None:
            return pos
        pos = end + 1


def command_arguments(text: str, command: str) -> list[str]:
    bodies: list[str] = []
    for match in re.finditer(rf"\\{re.escape(command)}\b", text):
        pos = skip_optional_arguments(text, match.end())
        if pos >= len(text) or text[pos] != "{":
            continue
        end = matching_delimiter(text, pos, "{", "}")
        if end is None:
            continue
        bodies.append(text[pos + 1 : end])
    return bodies


def main_body(tex: str) -> str:
    match = re.search(r"\\end\{abstract\}", tex)
    start = match.end() if match else 0
    tail = tex[start:]
    end_match = re.search(
        r"\\begin\{acknowledgments\}|\\section\*\{Open Research Statement\}|\\bibliographystyle|\\bibliography|\\end\{document\}",
        tail,
        flags=re.S,
    )
    body = tail[: end_match.start()] if end_match else tail
    return body


def keyword_body(body: str) -> str:
    match = re.search(
        r"\\section\*\{Keywords\}(?P<body>.*?)(?=\\section\{)",
        body,
        flags=re.S,
    )
    return match.group("body") if match else ""


def expand_citations(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        keys = [key.strip() for key in match.group(1).split(",") if key.strip()]
        return " " + " ".join("citation" for _ in keys or ["citation"]) + " "

    return re.sub(r"\\cite[A-Za-z*]*(?:\[[^\]]*\])*\{([^{}]+)\}", repl, text)


def latex_to_text(text: str) -> str:
    text = expand_citations(text)
    text = re.sub(r"\\(?:ref|pageref|autoref|eqref)\{[^{}]+\}", " reference ", text)
    text = re.sub(
        r"\\begin\{(?:equation|equation\*|align|align\*|gather|gather\*|linenomath\*)\}.*?\\end\{(?:equation|equation\*|align|align\*|gather|gather\*|linenomath\*)\}",
        " equation ",
        text,
        flags=re.S,
    )
    text = re.sub(r"\$.*?\$", " equation ", text, flags=re.S)
    text = re.sub(r"\\\((.*?)\\\)", " equation ", text, flags=re.S)
    text = re.sub(r"\\\[(.*?)\\\]", " equation ", text, flags=re.S)
    text = text.replace("\\%", " percent ")
    text = re.sub(r"\\([%&#_$])", r"\1", text)
    text = re.sub(r"\\[A-Za-z]+\*?(?:\[[^\]]*\])?", " ", text)
    text = text.replace("~", " ")
    text = text.replace("--", " ")
    text = re.sub(r"[{}_^$&]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def count_words(text: str) -> int:
    plain = latex_to_text(text)
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", plain))


def figure_paths(tex: str) -> list[str]:
    return re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)


def audit(args: argparse.Namespace) -> dict:
    manuscript = args.manuscript.resolve()
    tex = strip_comments(manuscript.read_text(encoding="utf-8"))

    abstract = remove_env(extract_env(tex, "abstract"), "plainlanguagesummary")
    plain_language_summary = extract_env(tex, "plainlanguagesummary")
    body = main_body(tex)
    keywords = keyword_body(body)
    body_for_count = re.sub(
        r"\\section\*\{Keywords\}.*?(?=\\section\{)",
        " ",
        body,
        flags=re.S,
    )
    captions = command_arguments(body_for_count, "caption")
    figure_captions = [
        caption
        for env_body in re.findall(r"\\begin\{figure\}(.*?)\\end\{figure\}", body_for_count, flags=re.S)
        for caption in command_arguments(env_body, "caption")
    ]
    table_captions = [
        caption
        for env_body in re.findall(r"\\begin\{table\}(.*?)\\end\{table\}", body_for_count, flags=re.S)
        for caption in command_arguments(env_body, "caption")
    ]

    body_without_floats = remove_envs(body_for_count, ["figure", "table"])
    body_without_tables = remove_envs(body_without_floats, ["tabular", "tabular*", "array"])

    components = {
        "abstract_words": count_words(abstract),
        "main_text_words": count_words(body_without_tables),
        "figure_caption_words": count_words(" ".join(figure_captions)),
        "table_caption_words": count_words(" ".join(table_captions)),
    }
    word_count = sum(components.values())
    excluded = {
        "plain_language_summary_words": count_words(plain_language_summary),
        "keywords_words": count_words(keywords),
        "table_body_words": count_words(" ".join(remove_env(env_body, "caption") for env_body in re.findall(r"\\begin\{table\}(.*?)\\end\{table\}", body_for_count, flags=re.S))),
    }
    figures = len(figure_paths(tex))
    tables = len(re.findall(r"\\begin\{table\}", tex))
    publication_units = word_count / 500.0 + figures + tables

    checks = {
        "word_count_positive": word_count > 0,
        "figure_count_matches_expected": figures == args.expected_figures,
        "table_count_matches_expected": tables == args.expected_tables,
        "publication_units_within_limit": publication_units <= args.max_publication_units,
    }
    failures = [name for name, ok in checks.items() if not ok]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Local AGU publication-unit audit for the manuscript source.",
        "manuscript": str(manuscript),
        "manuscript_sha256": sha256_file(manuscript),
        "source_guidance": {
            "rule": "publication_units = words / 500 + figures + tables",
            "word_count_includes": "abstract, text, in-text citations, figure captions, table captions, and appendices",
            "word_count_excludes": "title, author list, affiliations, key points, keywords, plain language summary, table body text, Open Research section, references, and supporting information",
            "guidance_urls": GUIDANCE_URLS,
        },
        "thresholds": {
            "max_publication_units": float(args.max_publication_units),
            "expected_figures": int(args.expected_figures),
            "expected_tables": int(args.expected_tables),
        },
        "counts": {
            "word_count_estimate": int(word_count),
            "figures": int(figures),
            "tables": int(tables),
            "publication_units": round(publication_units, 3),
            "publication_unit_margin": round(args.max_publication_units - publication_units, 3),
        },
        "word_count_components": components,
        "excluded_word_count_estimates": excluded,
        "caption_count": len(captions),
        "checks": checks,
        "failure_reasons": failures,
        "overall_status": "PASS" if not failures else "FAIL",
        "caveat": "This LaTeX-source estimate is intended for local QA; the AGU submission system remains authoritative.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Estimate AGU publication units from a LaTeX manuscript.")
    parser.add_argument("--manuscript", type=Path, default=OVERLEAF / "agujournaltemplate.tex")
    parser.add_argument("--output", type=Path, default=OVERLEAF / "publication_unit_audit.json")
    parser.add_argument("--max-publication-units", type=float, default=25.0)
    parser.add_argument("--expected-figures", type=int, default=8)
    parser.add_argument("--expected-tables", type=int, default=2)
    args = parser.parse_args()

    result = audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["overall_status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
