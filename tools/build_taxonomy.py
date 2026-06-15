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
DETECTABILITY_LABELS = {
    "io": "I/O signal",
    "session": "Session-level",
    "behavioral": "Baseline-only",
}
REPO_ROOT = Path(__file__).resolve().parent.parent


def load_taxonomy(path: Path) -> list[dict]:
    """Parse taxonomy.yaml into a list of category dicts."""
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def to_render_json(taxonomy: list[dict], *, source_base_url: str = "") -> dict:
    """Transform the curated taxonomy into render-ready JSON with counts."""
    summary = {"covered": 0, "in_depth": 0, "expanding": 0, "total": 0}
    detectability_summary = {"io": 0, "session": 0, "behavioral": 0}
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
            detect = m.get("detectability", "")
            if detect in detectability_summary:
                detectability_summary[detect] += 1
            fws = m.get("frameworks", [])
            frameworks.update(fws)
            tests = m.get("tests", [])
            methods_out.append({
                "name": m["name"],
                "status": status,
                "status_label": STATUS_LABELS.get(status, status),
                "detectability": detect,
                "detectability_label": DETECTABILITY_LABELS.get(detect, detect),
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
        "detectability_summary": detectability_summary,
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
            detect = m.get("detectability")
            if not detect:
                warnings.append(f"'{name}': missing detectability")
            elif detect not in DETECTABILITY_LABELS:
                warnings.append(f"'{name}': unknown detectability '{detect}'")
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


def write_artifact(taxonomy: list[dict], *, dest: Path) -> Path:
    """Render taxonomy to JSON and write to dest. Returns dest path."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    rendered = to_render_json(taxonomy)
    dest.write_text(json.dumps(rendered, indent=2), encoding="utf-8")
    return dest


def sync_to_bdc(artifact: Path, *, bdc_path: Path) -> Path:
    """Copy artifact JSON into bdc_path/data/. Creates data/ if needed."""
    data_dir = Path(bdc_path) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / Path(artifact).name
    shutil.copy2(artifact, dest)
    return dest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the published LLM attack taxonomy JSON.")
    parser.add_argument("--taxonomy", default=str(REPO_ROOT / "taxonomy.yaml"),
                        help="Path to taxonomy.yaml (default: repo root)")
    parser.add_argument("--artifact-dest",
                        default=str(REPO_ROOT / "published" / "taxonomy.json"),
                        help="Where to write taxonomy.json in the eval repo")
    parser.add_argument("--bdc-path", default=str(REPO_ROOT.parent / "blackdiamondconsulting.ai"),
                        help="Path to the BDC repo (default: ../blackdiamondconsulting.ai)")
    parser.add_argument("--no-sync", action="store_true",
                        help="Skip copying to the BDC repo")
    parser.add_argument("--source-base-url", default="",
                        help="Base URL for source links in the rendered page (off by default)")
    args = parser.parse_args(argv)

    taxonomy = load_taxonomy(Path(args.taxonomy))
    warnings = validate(taxonomy)
    if warnings:
        print("DRIFT WARNINGS:")
        for w in warnings:
            print(f"  DRIFT: {w}")
        print()

    artifact = write_artifact(taxonomy, dest=Path(args.artifact_dest))
    print(f"Wrote {artifact}")

    if not args.no_sync:
        dest = sync_to_bdc(artifact, bdc_path=Path(args.bdc_path))
        print(f"Copied to {dest}")
        print()
        print("Next steps:")
        print(f"  cd {args.bdc_path}")
        print("  hugo server")
        print("  # verify the taxonomy page at http://localhost:1313")
        print("  git add data/taxonomy.json")
        print("  git commit -m 'update: taxonomy matrix'")
        print("  git push")
    else:
        print("  (--no-sync: skipped BDC copy)")


if __name__ == "__main__":
    main()
