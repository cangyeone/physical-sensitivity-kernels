#!/usr/bin/env python3
"""Write a reproducible runtime-environment report for the manuscript archive."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent
OVERLEAF = SCRIPTS.parent
ROOT = OVERLEAF.parent


PACKAGE_SPECS = [
    ("torch", "torch", True),
    ("numpy", "numpy", True),
    ("matplotlib", "matplotlib", True),
    ("disba", "disba", True),
    ("obspy", "obspy", True),
    ("scipy", "scipy", False),
    ("Pillow", "PIL", False),
]

COMMAND_SPECS = [
    ("python", [sys.executable, "--version"]),
    ("git", ["git", "--version"]),
    ("latexmk", ["latexmk", "--version"]),
    ("pdflatex", ["pdflatex", "--version"]),
]

THREAD_ENV_VARS = [
    "KMP_DUPLICATE_LIB_OK",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYTHONHASHSEED",
    "CONDA_DEFAULT_ENV",
    "VIRTUAL_ENV",
]

# The local macOS/PyTorch/disba stack used for this manuscript needs the same
# OpenMP duplicate-runtime override and conservative thread limits as the
# figure-generation commands.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
for thread_var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(thread_var, "1")


def package_record(distribution: str, import_name: str, required: bool) -> dict:
    record = {
        "distribution": distribution,
        "import_name": import_name,
        "required": required,
        "installed_version": None,
        "importable": False,
        "import_error": None,
        "import_returncode": None,
    }
    try:
        record["installed_version"] = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        record["import_error"] = "distribution not found"
        return record

    code = (
        "import importlib, json; "
        f"module = importlib.import_module({import_name!r}); "
        "print(json.dumps({'module_version': getattr(module, '__version__', None)}))"
    )
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
            env=env,
        )
    except Exception as exc:  # pragma: no cover - diagnostic output only
        record["import_error"] = f"{type(exc).__name__}: {exc}"
        return record

    record["import_returncode"] = completed.returncode
    if completed.returncode == 0:
        record["importable"] = True
        try:
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError):
            payload = {}
        module_version = payload.get("module_version")
        if module_version and record["installed_version"] is None:
            record["installed_version"] = str(module_version)
    else:
        output = (completed.stderr or completed.stdout).strip()
        record["import_error"] = output or f"import subprocess exited with code {completed.returncode}"

    return record


def command_record(name: str, command: list[str]) -> dict:
    executable = command[0]
    resolved = shutil.which(executable) if executable != sys.executable else sys.executable
    record = {
        "name": name,
        "command": command,
        "resolved_path": resolved,
        "available": bool(resolved),
        "version_line": None,
        "error": None,
    }
    if not resolved:
        return record
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except Exception as exc:  # pragma: no cover - diagnostic output only
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    output = (completed.stdout or completed.stderr).strip().splitlines()
    record["returncode"] = completed.returncode
    record["version_line"] = output[0].strip() if output else ""
    return record


def git_record(path: Path) -> dict:
    record = {"path": str(path), "available": False, "commit": None, "branch": None, "dirty": None}
    if not (path / ".git").exists():
        return record

    def run_git(args: list[str]) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(path), *args],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except Exception:
            return None
        if completed.returncode != 0:
            return None
        return completed.stdout.strip()

    commit = run_git(["rev-parse", "HEAD"])
    branch = run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    status = run_git(["status", "--short"])
    record.update(
        {
            "available": commit is not None,
            "commit": commit,
            "branch": branch,
            "dirty": bool(status),
        }
    )
    return record


def torch_record() -> dict | None:
    code = """
import json
import torch
record = {
    "version": getattr(torch, "__version__", None),
    "num_threads": int(torch.get_num_threads()),
    "num_interop_threads": int(torch.get_num_interop_threads()),
    "cuda_available": bool(torch.cuda.is_available()),
    "cuda_version": getattr(torch.version, "cuda", None),
}
if hasattr(torch.backends, "mps"):
    record["mps_available"] = bool(torch.backends.mps.is_available())
print(json.dumps(record))
"""
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
            env=env,
        )
    except Exception as exc:  # pragma: no cover - diagnostic output only
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    if completed.returncode != 0:
        output = (completed.stderr or completed.stdout).strip()
        return {
            "available": False,
            "returncode": completed.returncode,
            "error": output or f"torch subprocess exited with code {completed.returncode}",
        }
    try:
        record = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    record["available"] = True
    return record


def build_report() -> dict:
    packages = [package_record(*spec) for spec in PACKAGE_SPECS]
    required_missing = [
        item["distribution"]
        for item in packages
        if item["required"] and not item["installed_version"]
    ]
    required_import_failures = [
        item["distribution"]
        for item in packages
        if item["required"] and item["installed_version"] and not item["importable"]
    ]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Runtime environment report for reproducing the amortized posterior sampling training, evaluation, and manuscript figures.",
        "project_root": str(ROOT),
        "overleaf_project": str(OVERLEAF),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "packages": packages,
        "required_missing": required_missing,
        "required_import_failures": required_import_failures,
        "torch_runtime": torch_record(),
        "commands": [command_record(*spec) for spec in COMMAND_SPECS],
        "git": {
            "main_repository": git_record(ROOT),
            "overleaf_repository": git_record(OVERLEAF),
        },
        "environment_variables": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
        "environment_overrides": {
            "KMP_DUPLICATE_LIB_OK": "set to TRUE by default inside make_environment_report.py if not already defined",
            "thread_limits": "OMP/MKL/OpenBLAS/VecLib/NumExpr thread counts default to 1 inside make_environment_report.py if not already defined",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a JSON runtime-environment report.")
    parser.add_argument("--output", type=Path, default=OVERLEAF / "environment_report.json")
    args = parser.parse_args()

    report = build_report()
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "packages": len(report["packages"]),
                "required_missing": report["required_missing"],
                "required_import_failures": report["required_import_failures"],
            },
            indent=2,
        )
    )
    return 1 if report["required_missing"] else 0


if __name__ == "__main__":
    sys.exit(main())
