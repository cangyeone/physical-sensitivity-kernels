#!/usr/bin/env python3
"""Create a file-level archive bundle plan from archive_inventory.json."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent
OVERLEAF = SCRIPTS.parent
DEFAULT_INVENTORY = OVERLEAF / "archive_inventory.json"
DEFAULT_OUTPUT = OVERLEAF / "archive_bundle_plan.json"

BUNDLE_BY_CATEGORY = {
    "software": "software_record",
    "software_optional": "software_record",
    "data_output": "data_output_record",
    "manuscript_source": "manuscript_source_record",
    "submission_support": "submission_support_record",
}


def archive_path(record: dict) -> str:
    category = record["category"]
    relative = record.get("relative_to_project_root")
    path = Path(record["path"])
    if relative:
        return f"{category}/{relative}"
    return f"{category}/external/{path.name}"


def load_inventory(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def plan_bundles(inventory: dict) -> dict:
    bundle_files: dict[str, list[dict]] = defaultdict(list)
    unknown_categories = []
    for record in inventory["files"]:
        bundle = BUNDLE_BY_CATEGORY.get(record["category"])
        if bundle is None:
            unknown_categories.append(record["category"])
            bundle = "unassigned_record"
        planned = {
            "archive_path": archive_path(record),
            "source_path": record["path"],
            "category": record["category"],
            "required": bool(record["required"]),
            "exists": bool(record["exists"]),
            "size_bytes": int(record.get("size_bytes", 0)),
            "sha256": record.get("sha256"),
            "note": record.get("note", ""),
        }
        bundle_files[bundle].append(planned)

    bundles = {}
    for name, files in sorted(bundle_files.items()):
        required_files = [item for item in files if item["required"]]
        missing_required = [item for item in required_files if not item["exists"]]
        bundles[name] = {
            "file_count": len(files),
            "required_file_count": len(required_files),
            "total_size_bytes": sum(item["size_bytes"] for item in files),
            "required_size_bytes": sum(item["size_bytes"] for item in required_files),
            "missing_required": missing_required,
            "files": sorted(files, key=lambda item: item["archive_path"]),
        }

    required_missing = inventory.get("required_missing", [])
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "File-level plan for preparing Zenodo/AGU Open Research archive records from archive_inventory.json.",
        "source_inventory": str(DEFAULT_INVENTORY),
        "inventory_created_utc": inventory.get("created_utc"),
        "project_root": inventory.get("project_root"),
        "overleaf_project": inventory.get("overleaf_project"),
        "records_total": len(inventory.get("files", [])),
        "required_records_total": sum(1 for item in inventory.get("files", []) if item.get("required")),
        "total_size_bytes": sum(int(item.get("size_bytes", 0)) for item in inventory.get("files", [])),
        "required_size_bytes": sum(int(item.get("size_bytes", 0)) for item in inventory.get("files", []) if item.get("required")),
        "required_missing": required_missing,
        "unknown_categories": sorted(set(unknown_categories)),
        "ready_for_archive": not required_missing and not unknown_categories,
        "bundles": bundles,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a bundle plan from archive_inventory.json.")
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    inventory = load_inventory(args.inventory)
    plan = plan_bundles(inventory)
    plan["source_inventory"] = str(args.inventory)
    args.output.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "ready_for_archive": plan["ready_for_archive"],
                "records_total": plan["records_total"],
                "required_size_bytes": plan["required_size_bytes"],
                "bundles": sorted(plan["bundles"]),
            },
            indent=2,
        )
    )
    return 0 if plan["ready_for_archive"] else 1


if __name__ == "__main__":
    sys.exit(main())
