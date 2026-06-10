#!/usr/bin/env python3
"""Generate the published LLM attack taxonomy JSON from taxonomy.yaml.

Usage:
    python tools/build_taxonomy.py [--bdc-path PATH] [--no-sync] [--source-base-url URL]
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

import yaml

STATUS_LABELS = {"covered": "Covered", "in_depth": "In-depth", "expanding": "Expanding"}
REPO_ROOT = Path(__file__).resolve().parent.parent


def load_taxonomy(path: Path) -> list[dict]:
    """Parse taxonomy.yaml into a list of category dicts."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def to_render_json(taxonomy: list[dict], *, source_base_url: str = "") -> dict:
    """Transform the curated taxonomy into render-ready JSON with counts."""
    summary = {"covered": 0, "in_depth": 0, "expanding": 0, "total": 0}
    frameworks: set[str] = set()
    categories: list[dict] = []
    base = source_base_url.rstrip("/") + "/" if source_base_url else ""

    for cat in taxonomy:
        methods_out = []
        for m in cat["methods"]:
            status = m["status"]
            if status in STATUS_LABELS:
                summary[status] += 1
            summary["total"] += 1
            fws = m.get("frameworks", [])
            frameworks.update(fws)
            tests = m.get("tests", [])
            methods_out.append({
                "name": m["name"],
                "status": status,
                "status_label": STATUS_LABELS.get(status, status),
                "frameworks": fws,
                "blurb": m.get("blurb", ""),
                "tests": tests,
                "source_links": [base + t for t in tests] if base else [],
            })
        categories.append({
            "category": cat["category"],
            "description": cat.get("description", ""),
            "methods": methods_out,
        })

    return {
        "generated": datetime.now(UTC).isoformat(),
        "summary": summary,
        "frameworks": sorted(frameworks),
        "categories": categories,
    }


def validate(taxonomy: list[dict], *, repo_root: Path = REPO_ROOT) -> list[str]:
    """Return non-fatal drift warnings comparing the taxonomy to the live suite."""
    warnings: list[str] = []
    for cat in taxonomy:
        for m in cat["methods"]:
            name, status = m["name"], m["status"]
            tests = m.get("tests", [])
            if status == "expanding" and tests:
                warnings.append(f"'{name}': status is expanding but lists tests {tests}")
            if status in ("covered", "in_depth"):
                if not tests:
                    warnings.append(f"'{name}': status {status} but no tests listed")
                for rel in tests:
                    path = repo_root / rel
                    if not path.exists():
                        warnings.append(f"'{name}': test file missing: {rel}")
                        continue
                    text = path.read_text(encoding="utf-8", errors="ignore")
                    for fw in m.get("frameworks", []):
                        if fw not in text:
                            warnings.append(
                                f"'{name}': marker '{fw}' not found in {rel}"
                            )
    return warnings
