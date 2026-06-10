"""Unit tests for the taxonomy generator (no API calls)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import build_taxonomy as bt  # noqa: E402

SAMPLE = [
    {
        "category": "Jailbreak & Injection",
        "description": "Input-side manipulation.",
        "methods": [
            {"name": "DAN", "status": "covered", "detectability": "io",
             "frameworks": ["owasp_llm01"],
             "blurb": "personas", "tests": ["tests/test_jailbreak.py"]},
            {"name": "Many-shot", "status": "expanding", "detectability": "session",
             "frameworks": ["owasp_llm01"],
             "blurb": "flooding", "tests": []},
        ],
    },
    {
        "category": "Information Extraction",
        "description": "Secrets.",
        "methods": [
            {"name": "Secret", "status": "in_depth", "detectability": "behavioral",
             "frameworks": ["owasp_llm02", "nist_privacy"],
             "blurb": "tokens", "tests": ["tests/test_secret_extraction.py"]},
        ],
    },
]


def test_load_taxonomy_parses_yaml(tmp_path: Path) -> None:
    f = tmp_path / "t.yaml"
    f.write_text(
        "- category: A\n  description: d\n  methods:\n"
        "    - name: M\n      status: covered\n      frameworks: [owasp_llm01]\n"
        "      blurb: b\n      tests: []\n",
        encoding="utf-8",
    )
    data = bt.load_taxonomy(f)
    assert data[0]["category"] == "A"
    assert data[0]["methods"][0]["name"] == "M"


def test_render_json_summary_counts() -> None:
    out = bt.to_render_json(SAMPLE)
    assert out["summary"] == {"covered": 1, "in_depth": 1, "expanding": 1, "total": 3}


def test_render_json_collects_sorted_frameworks() -> None:
    out = bt.to_render_json(SAMPLE)
    assert out["frameworks"] == ["nist_privacy", "owasp_llm01", "owasp_llm02"]


def test_render_json_adds_status_label() -> None:
    out = bt.to_render_json(SAMPLE)
    method = out["categories"][0]["methods"][0]
    assert method["status_label"] == "Covered"
    assert out["categories"][0]["methods"][1]["status_label"] == "Expanding"
    assert out["categories"][1]["methods"][0]["status_label"] == "In-depth"


def test_render_json_detectability_summary_counts() -> None:
    out = bt.to_render_json(SAMPLE)
    assert out["detectability_summary"] == {"io": 1, "session": 1, "behavioral": 1}


def test_render_json_adds_detectability_label() -> None:
    out = bt.to_render_json(SAMPLE)
    method = out["categories"][0]["methods"][0]
    assert method["detectability"] == "io"
    assert method["detectability_label"] == "I/O signal"
    assert out["categories"][0]["methods"][1]["detectability_label"] == "Session-level"
    assert out["categories"][1]["methods"][0]["detectability_label"] == "Baseline-only"


def test_render_json_ignores_unknown_detectability_in_summary() -> None:
    tax = [{"category": "C", "description": "d", "methods": [
        {"name": "X", "status": "covered", "detectability": "bogus",
         "frameworks": [], "blurb": "b", "tests": []},
    ]}]
    out = bt.to_render_json(tax)
    assert out["detectability_summary"] == {"io": 0, "session": 0, "behavioral": 0}
    # unknown value still passes through on the method, labelled as itself
    assert out["categories"][0]["methods"][0]["detectability_label"] == "bogus"


def test_source_links_empty_by_default() -> None:
    out = bt.to_render_json(SAMPLE)
    assert out["categories"][0]["methods"][0]["source_links"] == []


def test_source_links_built_when_base_url_set() -> None:
    out = bt.to_render_json(SAMPLE, source_base_url="https://example.com/blob/main/")
    assert out["categories"][0]["methods"][0]["source_links"] == [
        "https://example.com/blob/main/tests/test_jailbreak.py"
    ]


def test_render_json_ignores_unknown_status_in_summary() -> None:
    tax = [{"category": "C", "description": "d", "methods": [
        {"name": "X", "status": "bogus", "frameworks": [], "blurb": "b", "tests": []},
    ]}]
    out = bt.to_render_json(tax)
    assert set(out["summary"]) == {"covered", "in_depth", "expanding", "total"}
    assert out["summary"]["total"] == 1


def test_validate_flags_missing_test_file(tmp_path: Path) -> None:
    tax = [{"category": "C", "description": "d", "methods": [
        {"name": "X", "status": "covered", "frameworks": ["owasp_llm01"],
         "blurb": "b", "tests": ["tests/does_not_exist.py"]},
    ]}]
    warnings = bt.validate(tax, repo_root=tmp_path)
    assert any("does_not_exist.py" in w for w in warnings)


def test_validate_flags_missing_marker(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "t.py").write_text("pytest.mark.owasp_llm01", encoding="utf-8")
    tax = [{"category": "C", "description": "d", "methods": [
        {"name": "X", "status": "covered", "frameworks": ["owasp_llm09"],
         "blurb": "b", "tests": ["tests/t.py"]},
    ]}]
    warnings = bt.validate(tax, repo_root=tmp_path)
    assert any("owasp_llm09" in w for w in warnings)


def test_validate_flags_expanding_with_tests(tmp_path: Path) -> None:
    tax = [{"category": "C", "description": "d", "methods": [
        {"name": "X", "status": "expanding", "frameworks": [],
         "blurb": "b", "tests": ["tests/t.py"]},
    ]}]
    warnings = bt.validate(tax, repo_root=tmp_path)
    assert any("expanding" in w.lower() for w in warnings)


def test_validate_flags_missing_detectability(tmp_path: Path) -> None:
    tax = [{"category": "C", "description": "d", "methods": [
        {"name": "X", "status": "expanding", "frameworks": [],
         "blurb": "b", "tests": []},
    ]}]
    warnings = bt.validate(tax, repo_root=tmp_path)
    assert any("missing detectability" in w for w in warnings)


def test_validate_flags_unknown_detectability(tmp_path: Path) -> None:
    tax = [{"category": "C", "description": "d", "methods": [
        {"name": "X", "status": "expanding", "detectability": "bogus", "frameworks": [],
         "blurb": "b", "tests": []},
    ]}]
    warnings = bt.validate(tax, repo_root=tmp_path)
    assert any("unknown detectability" in w for w in warnings)


def test_validate_clean_taxonomy_no_false_warnings(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "t.py").write_text("pytest.mark.owasp_llm01", encoding="utf-8")
    tax = [{"category": "C", "description": "d", "methods": [
        {"name": "X", "status": "covered", "detectability": "io",
         "frameworks": ["owasp_llm01"], "blurb": "b", "tests": ["tests/t.py"]},
        {"name": "Y", "status": "expanding", "detectability": "behavioral",
         "frameworks": ["owasp_llm01"], "blurb": "b", "tests": []},
    ]}]
    warnings = bt.validate(tax, repo_root=tmp_path)
    assert warnings == []


def test_write_artifact_creates_json(tmp_path: Path) -> None:
    tax = [{"category": "C", "description": "d", "methods": [
        {"name": "X", "status": "covered", "frameworks": ["owasp_llm01"],
         "blurb": "b", "tests": []},
    ]}]
    out = bt.write_artifact(tax, dest=tmp_path / "out.json")
    assert (tmp_path / "out.json").exists()
    data = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert data["summary"]["total"] == 1
    assert out == tmp_path / "out.json"


def test_sync_to_bdc_copies_file(tmp_path: Path) -> None:
    src = tmp_path / "taxonomy.json"
    src.write_text('{"test": 1}', encoding="utf-8")
    bdc = tmp_path / "bdc_repo"
    bdc.mkdir()
    (bdc / "data").mkdir()
    dest = bt.sync_to_bdc(src, bdc_path=bdc)
    assert dest == bdc / "data" / "taxonomy.json"
    assert (bdc / "data" / "taxonomy.json").read_text(encoding="utf-8") == '{"test": 1}'


def test_sync_to_bdc_creates_data_dir_if_missing(tmp_path: Path) -> None:
    src = tmp_path / "taxonomy.json"
    src.write_text("{}", encoding="utf-8")
    bdc = tmp_path / "bdc_repo"
    bdc.mkdir()
    # data/ does NOT exist — sync must create it
    bt.sync_to_bdc(src, bdc_path=bdc)
    assert (bdc / "data" / "taxonomy.json").exists()


def test_main_no_sync_writes_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    taxonomy_file = tmp_path / "taxonomy.yaml"
    taxonomy_file.write_text(
        "- category: C\n  description: d\n  methods:\n"
        "    - name: X\n      status: covered\n      frameworks: [owasp_llm01]\n"
        "      blurb: b\n      tests: []\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    (tmp_path / "published").mkdir()
    bt.main([
        "--taxonomy", str(taxonomy_file),
        "--no-sync",
        "--artifact-dest", str(tmp_path / "published" / "taxonomy.json"),
    ])
    assert (tmp_path / "published" / "taxonomy.json").exists()
