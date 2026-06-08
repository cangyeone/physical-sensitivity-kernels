#!/usr/bin/env python3
"""Validate the clean runtime environment specification without creating it."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent
OVERLEAF = SCRIPTS.parent
DEFAULT_SPEC = OVERLEAF / "environment-repro.yml"
DEFAULT_OUTPUT = OVERLEAF / "environment_validation.json"


def read_pip_pins(spec_text: str) -> dict[str, str]:
    pins: dict[str, str] = {}
    for match in re.finditer(r"^\s*-\s*([A-Za-z0-9_.-]+)==([A-Za-z0-9_.!+-]+)\s*$", spec_text, flags=re.M):
        pins[match.group(1)] = match.group(2)
    return pins


def command_result(command: list[str], timeout: int = 180) -> dict:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:  # pragma: no cover - diagnostic output only
        return {
            "command": command,
            "returncode": None,
            "ok": False,
            "stdout": "",
            "stderr": "",
            "error": f"{type(exc).__name__}: {exc}",
        }
    return {
        "command": command,
        "returncode": completed.returncode,
        "ok": completed.returncode == 0,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "error": None,
    }


def parse_conda_dry_run(stdout: str) -> dict:
    start = stdout.find("{")
    end = stdout.rfind("}")
    if start < 0 or end < start:
        return {"parse_ok": False, "package_count": 0, "name": None, "channels": []}
    try:
        payload = json.loads(stdout[start : end + 1])
    except json.JSONDecodeError:
        return {"parse_ok": False, "package_count": 0, "name": None, "channels": []}
    dependencies = payload.get("dependencies", [])
    return {
        "parse_ok": True,
        "name": payload.get("name"),
        "channels": payload.get("channels", []),
        "package_count": len(dependencies),
        "python_package": next((item for item in dependencies if "::python==" in item), None),
        "numpy_package": next((item for item in dependencies if "::numpy==" in item), None),
        "matplotlib_package": next((item for item in dependencies if "::matplotlib==" in item), None),
        "obspy_package": next((item for item in dependencies if "::obspy==" in item), None),
    }


def parse_pip_index(stdout: str, package: str, pinned: str) -> dict:
    available_line = next((line for line in stdout.splitlines() if line.startswith("Available versions:")), "")
    available = [part.strip() for part in available_line.replace("Available versions:", "").split(",") if part.strip()]
    first_line = stdout.splitlines()[0].strip() if stdout.splitlines() else ""
    return {
        "package": package,
        "pinned_version": pinned,
        "available": pinned in available,
        "latest_line": first_line,
        "available_versions_checked": available[:20],
    }


def validate(spec: Path) -> dict:
    spec_text = spec.read_text(encoding="utf-8")
    pip_pins = read_pip_pins(spec_text)

    conda_command = ["conda", "env", "create", "-f", str(spec), "--dry-run", "--json"]
    conda_run = command_result(conda_command)
    conda_summary = parse_conda_dry_run(conda_run["stdout"]) if conda_run["ok"] else {
        "parse_ok": False,
        "package_count": 0,
        "name": None,
        "channels": [],
    }

    pip_checks = []
    for package, pinned in sorted(pip_pins.items()):
        run = command_result([sys.executable, "-m", "pip", "index", "versions", package], timeout=60)
        summary = parse_pip_index(run["stdout"], package, pinned) if run["ok"] else {
            "package": package,
            "pinned_version": pinned,
            "available": False,
            "latest_line": "",
            "available_versions_checked": [],
        }
        summary.update(
            {
                "returncode": run["returncode"],
                "ok": run["ok"],
                "stderr_first_line": run["stderr"].splitlines()[0] if run["stderr"].splitlines() else "",
            }
        )
        pip_checks.append(summary)

    required_terms = ["python=3.11", "numpy=", "matplotlib=", "obspy=", "torch==", "disba=="]
    missing_terms = [term for term in required_terms if term not in spec_text]
    pip_unavailable = [
        item["package"]
        for item in pip_checks
        if not item["ok"] or not item["available"]
    ]
    overall_ok = (
        spec.exists()
        and not missing_terms
        and conda_run["ok"]
        and conda_summary["parse_ok"]
        and not pip_unavailable
    )

    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "spec_path": str(spec),
        "purpose": "Dry-run validation of the clean runtime environment specification for manuscript archiving.",
        "required_terms": required_terms,
        "missing_terms": missing_terms,
        "conda_dry_run": {
            "ok": conda_run["ok"],
            "returncode": conda_run["returncode"],
            "stderr_first_line": conda_run["stderr"].splitlines()[0] if conda_run["stderr"].splitlines() else "",
            "summary": conda_summary,
        },
        "pip_index_checks": pip_checks,
        "pip_unavailable": pip_unavailable,
        "overall_ok": overall_ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate environment-repro.yml without creating an environment.")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = validate(args.spec)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "overall_ok": report["overall_ok"],
                "conda_dry_run_ok": report["conda_dry_run"]["ok"],
                "pip_unavailable": report["pip_unavailable"],
            },
            indent=2,
        )
    )
    return 0 if report["overall_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
