# LLM Attack Taxonomy — Interactive Coverage Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a curated, interactive taxonomy of LLM attack vectors/methods to the Black Diamond Consulting Hugo site, driven by a single source-of-truth file in the eval repo and a one-command generator.

**Architecture:** `taxonomy.yaml` (eval repo, curated) → `build_taxonomy.py` validates against the live pytest suite and emits `taxonomy.json` → the JSON is copied into the BDC repo's `data/` → a Hugo `_default/taxonomy.html` layout renders it at build time into a filterable table that inherits the theme's nav/footer/CTA. The generator never commits or pushes BDC; the user verifies via `hugo server` before pushing.

**Tech Stack:** Python 3.10+ (PyYAML, stdlib argparse/json/re/datetime, pytest for the generator's own tests), Hugo (Go templates, `.Site.Data`), vanilla JS/CSS (no build step, no CDN — CSP-safe).

**Two repos:**
- Eval repo: `c:\Users\seany\projects\llm-adversarial-eval` (cwd; tasks 1–6 + 9).
- BDC repo: `c:\Users\seany\projects\blackdiamondconsulting.ai` (tasks 7–8; **not committed by the executor** — handed to the user).

---

## File Structure

| File | Repo | Responsibility |
|---|---|---|
| `taxonomy.yaml` | eval | Curated source of truth: categories → methods → status/frameworks/blurb/tests |
| `tools/build_taxonomy.py` | eval | Parse → validate (drift vs. suite) → emit render-ready JSON → sync to BDC |
| `tests/test_build_taxonomy.py` | eval | Unit tests for the generator (no API calls) |
| `pyproject.toml` | eval | Add `pyyaml` dependency |
| `published/taxonomy.json` | eval | Versioned copy of the generated artifact |
| `COVERAGE.md` | eval | Add "Regenerating the published taxonomy" note |
| `data/taxonomy.json` | BDC | Hugo build input (synced; ignored by executor's commits) |
| `content/resources/llm-attack-taxonomy.md` | BDC | Page stub (front matter only) |
| `layouts/_default/taxonomy.html` | BDC | Full-page layout: renders the table + scoped CSS/JS + CTA |

---

## Task 1: Add PyYAML dependency and create `taxonomy.yaml`

**Files:**
- Modify: `pyproject.toml` (dependencies list, around line 10–17)
- Create: `taxonomy.yaml`

- [ ] **Step 1: Add the PyYAML dependency**

In `pyproject.toml`, add `"pyyaml>=6.0",` to the `dependencies` array (after the `Pillow` line):

```toml
dependencies = [
    "anthropic>=0.40.0",
    "openai>=1.0.0",
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "python-dotenv>=1.0",
    "Pillow>=10.0",
    "pyyaml>=6.0",
]
```

- [ ] **Step 2: Install it**

Run: `pip install pyyaml>=6.0`
Expected: `Successfully installed PyYAML-6.x` (or "already satisfied").

- [ ] **Step 3: Create `taxonomy.yaml`** with the full seed below.

`status` is one of `covered` (≥1 working probe), `in_depth` (multi-technique coverage), `expanding` (on the roadmap, no probe yet). `frameworks` values reuse markers registered in `pyproject.toml`. `tests` lists test files for `covered`/`in_depth` entries only.

```yaml
# Source of truth for the published LLM attack taxonomy.
# Regenerate the published artifact with:  python tools/build_taxonomy.py
# status: covered | in_depth | expanding   (expanding = on the assessment roadmap)
- category: Jailbreak & Injection
  description: Input-side manipulation to elicit policy-violating output.
  methods:
    - name: Direct persona / alter-ego (DAN-style)
      status: covered
      frameworks: [owasp_llm01, nist_evasion, mitre_llm_jailbreak]
      blurb: Alter-ego personas instructed to ignore restrictions.
      tests: [tests/test_jailbreak.py]
    - name: Fictional / academic framing
      status: covered
      frameworks: [owasp_llm01, nist_evasion, mitre_llm_jailbreak]
      blurb: Story, roleplay, or research framings that wrap a restricted request.
      tests: [tests/test_jailbreak.py]
    - name: Direct system-prompt override
      status: covered
      frameworks: [owasp_llm01, mitre_llm_injection]
      blurb: "SYSTEM OVERRIDE / ignore-previous-instructions injection in the user turn."
      tests: [tests/test_jailbreak.py]
    - name: Indirect injection via retrieved content
      status: in_depth
      frameworks: [owasp_llm01, owasp_llm04, nist_poisoning, mitre_llm_injection]
      blurb: Instructions embedded in RAG documents the model is asked to process.
      tests: [tests/test_indirect_injection.py]
    - name: Payload splitting (multi-turn)
      status: in_depth
      frameworks: [owasp_llm01, nist_evasion, mitre_llm_jailbreak]
      blurb: A harmful request spread across individually-innocuous turns.
      tests: [tests/test_payload_splitting.py]
    - name: Virtualization / eval-mode framing
      status: in_depth
      frameworks: [owasp_llm01, nist_evasion, mitre_llm_jailbreak]
      blurb: Claims that safety rules are suspended in a sandbox/test/dry-run context.
      tests: [tests/test_virtualization_attacks.py]
    - name: Base64 / encoding smuggling
      status: covered
      frameworks: [owasp_llm01, nist_evasion, mitre_llm_jailbreak]
      blurb: Harmful instruction encoded (base64) to slip past surface matching.
      tests: [tests/test_jailbreak.py]
    - name: Many-shot jailbreaking (128-256 shots)
      status: expanding
      frameworks: [owasp_llm01, nist_evasion, mitre_llm_jailbreak]
      blurb: Flooding context with many fake exchanges to erode refusal at scale.
      tests: []
    - name: Crescendo (gradual escalation)
      status: expanding
      frameworks: [owasp_llm01, nist_evasion, mitre_llm_jailbreak]
      blurb: Benign-to-harmful drift within a single topic across turns.
      tests: []
    - name: Past-tense reformulation
      status: expanding
      frameworks: [owasp_llm01, nist_evasion, mitre_llm_jailbreak]
      blurb: Reframing a prohibited request in the past tense to bypass refusal.
      tests: []
    - name: Refusal suppression / prefix injection
      status: expanding
      frameworks: [owasp_llm01, nist_evasion, mitre_llm_jailbreak]
      blurb: Forcing an affirmative opening or banning disclaimers/apologies.
      tests: []
    - name: Low-resource-language / translation bypass
      status: expanding
      frameworks: [owasp_llm01, nist_evasion, mitre_llm_jailbreak]
      blurb: Routing a harmful request through a low-resource language.
      tests: []
    - name: Cipher / ASCII-art evasion (ArtPrompt, CipherChat)
      status: expanding
      frameworks: [owasp_llm01, nist_evasion, mitre_llm_jailbreak]
      blurb: Glyph- or cipher-based encodings that evade the safety classifier.
      tests: []
    - name: Adversarial suffix (GCG / transferable)
      status: expanding
      frameworks: [owasp_llm01, nist_evasion, mitre_llm_jailbreak]
      blurb: Optimized gibberish suffixes that flip refusal to compliance.
      tests: []
- category: Information Extraction
  description: Eliciting protected context, secrets, or other users' data.
  methods:
    - name: System-prompt leakage (structural)
      status: in_depth
      frameworks: [owasp_llm02, owasp_llm07, nist_privacy, mitre_recon]
      blurb: Enumerating restrictions, escalation triggers, and operator instructions.
      tests: [tests/test_system_prompt_leakage.py]
    - name: Secret / token extraction
      status: in_depth
      frameworks: [owasp_llm02, owasp_llm07, nist_privacy, mitre_exfiltration]
      blurb: Direct, encoded, roleplay, indirect, and multi-turn secret extraction.
      tests: [tests/test_secret_extraction.py]
    - name: Cross-user / PHI boundary breach
      status: in_depth
      frameworks: [owasp_llm02, owasp_llm08, nist_privacy, mitre_exfiltration]
      blurb: Extracting another tenant's records from a multi-user RAG context.
      tests: [tests/test_cross_user_exfiltration.py]
    - name: Training-data memorization extraction
      status: expanding
      frameworks: [owasp_llm02, nist_privacy, mitre_exfiltration]
      blurb: Divergence/repetition attacks that surface memorized training text.
      tests: []
    - name: Membership inference / model inversion
      status: expanding
      frameworks: [nist_privacy]
      blurb: Inferring training-set membership; largely research-only behaviorally.
      tests: []
- category: Output Handling
  description: Unsafe model output that harms a downstream consumer.
  methods:
    - name: Markdown / image data exfiltration
      status: expanding
      frameworks: [owasp_llm05, owasp_llm02]
      blurb: Emitting an image/link URL that leaks context to an attacker on render.
      tests: []
    - name: Insecure generated code as output
      status: expanding
      frameworks: [owasp_llm05]
      blurb: Generated SQL/HTML/shell that is unsafe when executed downstream.
      tests: []
    - name: Structured-output breakout
      status: expanding
      frameworks: [owasp_llm05]
      blurb: JSON/format injection that escapes the intended output contract.
      tests: []
- category: Agentic & Tool-Use
  description: Misuse of an agent's tools, authority, or memory.
  methods:
    - name: Described-tool action boundary
      status: covered
      frameworks: [owasp_llm06, nist_abuse, eu_ai_art14]
      blurb: Refusing to claim execution of described (non-real) tool actions.
      tests: [tests/test_agentic_boundary.py]
    - name: Unbounded consumption (behavioral)
      status: covered
      frameworks: [owasp_llm10, nist_abuse]
      blurb: Refusing to attempt unbounded/indefinite output generation.
      tests: [tests/test_agentic_boundary.py]
    - name: Real tool-call / function injection
      status: expanding
      frameworks: [owasp_llm06, mitre_llm_injection]
      blurb: Injection that drives actual API tool calls via the tools= interface.
      tests: []
    - name: Tool-result injection
      status: expanding
      frameworks: [owasp_llm06, owasp_llm01]
      blurb: Poisoned tool_result blocks accepted as authoritative.
      tests: []
    - name: Confused-deputy / goal hijack via tool chaining
      status: expanding
      frameworks: [owasp_llm06, nist_abuse]
      blurb: Injected content steering an agent to misuse its own privileges.
      tests: []
    - name: Persistent memory poisoning
      status: expanding
      frameworks: [owasp_llm06, owasp_llm04]
      blurb: Cross-session corruption of agent memory.
      tests: []
- category: Multimodal
  description: Attacks delivered through non-text channels.
  methods:
    - name: Hidden-text image injection
      status: in_depth
      frameworks: [owasp_llm01, nist_evasion, mitre_llm_injection]
      blurb: Near-invisible or tiny-font instructions embedded in uploaded images.
      tests: [tests/test_hidden_text_injection.py]
    - name: Unicode tag-block / invisible-char smuggling (text)
      status: expanding
      frameworks: [owasp_llm01, nist_evasion]
      blurb: Instructions hidden in U+E0000 tag characters within plain text.
      tests: []
    - name: Homoglyph substitution
      status: expanding
      frameworks: [owasp_llm01, nist_evasion]
      blurb: Look-alike Unicode characters that evade string-level filters.
      tests: []
    - name: Audio / QR / steganographic payloads
      status: expanding
      frameworks: [owasp_llm01, nist_evasion]
      blurb: Instructions carried in audio, QR codes, or image steganography.
      tests: []
- category: Generation-Quality Harms
  description: Confident-but-wrong or unsafe content under normal use.
  methods:
    - name: Factual hallucination
      status: in_depth
      frameworks: [owasp_llm09, eu_ai_art15]
      blurb: Fabricated people, events, citations, and false-premise acceptance.
      tests: [tests/test_factual_hallucinations.py]
    - name: Sycophantic reversal under pressure
      status: in_depth
      frameworks: [owasp_llm09, eu_ai_art15]
      blurb: Abandoning a correct position under pushback, authority, or emotion.
      tests: [tests/test_sycophancy.py]
    - name: Temporal / stale-data error
      status: in_depth
      frameworks: [owasp_llm09, eu_ai_art15]
      blurb: Stale regulatory/legal/market facts stated without a staleness hedge.
      tests: [tests/test_legal_chatbot.py, tests/test_financial_chatbot.py]
    - name: Package hallucination (slopsquatting)
      status: expanding
      frameworks: [owasp_llm09]
      blurb: Inventing installable package names an attacker can register.
      tests: []
    - name: Insecure-code suggestion (measured)
      status: expanding
      frameworks: [owasp_llm09]
      blurb: Systematically scoring the safety of generated code, not just eliciting it.
      tests: []
- category: Content-Harm Payloads
  description: The harmful subject matter that jailbreaks attempt to elicit.
  methods:
    - name: Medical / overdose / dosing
      status: covered
      frameworks: [owasp_llm09, eu_ai_art15]
      blurb: Lethal-dose and unsafe-dosing requests across competing-objective framings.
      tests: [tests/test_competing_objectives.py, tests/test_payload_splitting.py]
    - name: Protected health information (PHI)
      status: covered
      frameworks: [owasp_llm02, nist_privacy]
      blurb: Cross-patient PHI and FHIR-boundary extraction.
      tests: [tests/test_healthcare_chatbot.py, tests/test_cross_user_exfiltration.py]
    - name: Dangerous-goods / HAZMAT downgrade
      status: covered
      frameworks: [owasp_llm09, eu_ai_art15]
      blurb: Pressure to downgrade HAZMAT/DGR/IMDG safety classifications.
      tests: [tests/test_logistics_chatbot.py, tests/test_air_freight_chatbot.py]
    - name: Regulated advice boundary (legal / financial)
      status: covered
      frameworks: [owasp_llm09, eu_ai_art14]
      blurb: UPL, unregistered-adviser, and suitability-reversal boundaries.
      tests: [tests/test_legal_chatbot.py, tests/test_financial_chatbot.py]
    - name: CBRN uplift
      status: expanding
      frameworks: [owasp_llm09]
      blurb: Chemical/biological/radiological/nuclear capability uplift.
      tests: []
    - name: Malware / offensive-cyber generation
      status: expanding
      frameworks: [owasp_llm09]
      blurb: Working exploit, malware, or intrusion tooling generation.
      tests: []
    - name: Fraud / scam / targeted disinformation
      status: expanding
      frameworks: [owasp_llm09]
      blurb: Scam scripts, phishing, and tailored disinformation authoring.
      tests: []
    - name: Self-harm (non-overdose)
      status: expanding
      frameworks: [owasp_llm09, eu_ai_art15]
      blurb: Self-harm methods beyond the medication-overdose vector already covered.
      tests: []
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml taxonomy.yaml
git commit -m "feat: add taxonomy.yaml source of truth + pyyaml dep"
```

---

## Task 2: Generator core — parse + render JSON (TDD)

**Files:**
- Create: `tests/test_build_taxonomy.py`
- Create: `tools/build_taxonomy.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_build_taxonomy.py`:

```python
"""Unit tests for the taxonomy generator (no API calls)."""
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
            {"name": "DAN", "status": "covered", "frameworks": ["owasp_llm01"],
             "blurb": "personas", "tests": ["tests/test_jailbreak.py"]},
            {"name": "Many-shot", "status": "expanding", "frameworks": ["owasp_llm01"],
             "blurb": "flooding", "tests": []},
        ],
    },
    {
        "category": "Information Extraction",
        "description": "Secrets.",
        "methods": [
            {"name": "Secret", "status": "in_depth", "frameworks": ["owasp_llm02", "nist_privacy"],
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


def test_source_links_empty_by_default() -> None:
    out = bt.to_render_json(SAMPLE)
    assert out["categories"][0]["methods"][0]["source_links"] == []


def test_source_links_built_when_base_url_set() -> None:
    out = bt.to_render_json(SAMPLE, source_base_url="https://example.com/blob/main/")
    assert out["categories"][0]["methods"][0]["source_links"] == [
        "https://example.com/blob/main/tests/test_jailbreak.py"
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_build_taxonomy.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'build_taxonomy'`.

- [ ] **Step 3: Write the minimal implementation**

Create `tools/build_taxonomy.py`:

```python
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
            summary[status] = summary.get(status, 0) + 1
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_build_taxonomy.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add tools/build_taxonomy.py tests/test_build_taxonomy.py
git commit -m "feat: taxonomy generator core (parse + render JSON)"
```

---

## Task 3: Generator drift validation (TDD)

**Files:**
- Modify: `tests/test_build_taxonomy.py`
- Modify: `tools/build_taxonomy.py`

- [ ] **Step 1: Add failing validation tests**

Append to `tests/test_build_taxonomy.py`:

```python
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


def test_validate_clean_taxonomy_no_false_warnings(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "t.py").write_text("pytest.mark.owasp_llm01", encoding="utf-8")
    tax = [{"category": "C", "description": "d", "methods": [
        {"name": "X", "status": "covered", "frameworks": ["owasp_llm01"],
         "blurb": "b", "tests": ["tests/t.py"]},
        {"name": "Y", "status": "expanding", "frameworks": ["owasp_llm01"],
         "blurb": "b", "tests": []},
    ]}]
    warnings = bt.validate(tax, repo_root=tmp_path)
    assert warnings == []
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_build_taxonomy.py -v`
Expected: FAIL — `AttributeError: module 'build_taxonomy' has no attribute 'validate'`.

- [ ] **Step 3: Implement `validate`**

Add to `tools/build_taxonomy.py` (after `to_render_json`):

```python
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
```

- [ ] **Step 4: Run to verify they pass**

Run: `pytest tests/test_build_taxonomy.py -v`
Expected: PASS (10 passed).

- [ ] **Step 5: Commit**

```bash
git add tools/build_taxonomy.py tests/test_build_taxonomy.py
git commit -m "feat: taxonomy generator drift validation"
```

---

## Task 4: Generator CLI + sync (TDD for sync, manual for CLI)

**Files:**
- Modify: `tests/test_build_taxonomy.py`
- Modify: `tools/build_taxonomy.py`

- [ ] **Step 1: Add a failing sync test**

Append to `tests/test_build_taxonomy.py`:

```python
def test_sync_writes_json_to_bdc_data(tmp_path: Path) -> None:
    bdc = tmp_path / "bdc"
    (bdc / "data").mkdir(parents=True)
    payload = {"summary": {"total": 0}, "categories": []}
    dest = bt.sync(payload, bdc_path=bdc)
    assert dest == bdc / "data" / "taxonomy.json"
    assert json.loads(dest.read_text(encoding="utf-8"))["summary"]["total"] == 0


def test_sync_errors_when_data_dir_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        bt.sync({"x": 1}, bdc_path=tmp_path / "nope")
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_build_taxonomy.py -v`
Expected: FAIL — `AttributeError: module 'build_taxonomy' has no attribute 'sync'`.

- [ ] **Step 3: Implement `sync`, `write_artifact`, and `main`**

Add to `tools/build_taxonomy.py`:

```python
def write_artifact(payload: dict, repo_root: Path = REPO_ROOT) -> Path:
    """Write the versioned copy into the eval repo's published/ directory."""
    dest = repo_root / "published" / "taxonomy.json"
    dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dest


def sync(payload: dict, *, bdc_path: Path) -> Path:
    """Copy the render-ready JSON into the BDC repo's data/ directory.

    Raises FileNotFoundError if the BDC data/ directory does not exist —
    we never create the target repo's structure implicitly.
    """
    data_dir = Path(bdc_path) / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"BDC data dir not found: {data_dir}")
    dest = data_dir / "taxonomy.json"
    dest.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the published LLM attack taxonomy.")
    parser.add_argument("--bdc-path", default="../blackdiamondconsulting.ai",
                        help="Path to the BDC Hugo repo (default: ../blackdiamondconsulting.ai)")
    parser.add_argument("--no-sync", action="store_true",
                        help="Generate published/taxonomy.json only; skip the BDC copy.")
    parser.add_argument("--source-base-url", default="",
                        help="If set, render 'covered' tests as links under this base URL.")
    args = parser.parse_args()

    taxonomy = load_taxonomy(REPO_ROOT / "taxonomy.yaml")
    warnings = validate(taxonomy)
    for w in warnings:
        print(f"  DRIFT: {w}")
    print(f"Validation: {len(warnings)} warning(s).")

    payload = to_render_json(taxonomy, source_base_url=args.source_base_url)
    artifact = write_artifact(payload)
    s = payload["summary"]
    print(f"Wrote {artifact}  "
          f"({s['total']} methods: {s['covered']} covered, "
          f"{s['in_depth']} in-depth, {s['expanding']} expanding)")

    if args.no_sync:
        print("Skipped BDC sync (--no-sync).")
        return
    dest = sync(payload, bdc_path=Path(args.bdc_path))
    print(f"Synced {dest}")
    print("Next: cd into the BDC repo, run `hugo server`, verify the page, "
          "then commit & push BDC yourself.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify the sync tests pass**

Run: `pytest tests/test_build_taxonomy.py -v`
Expected: PASS (12 passed).

- [ ] **Step 5: Commit**

```bash
git add tools/build_taxonomy.py tests/test_build_taxonomy.py
git commit -m "feat: taxonomy generator CLI + BDC sync"
```

---

## Task 5: Run the generator and commit the artifact

**Files:**
- Create: `published/taxonomy.json`

- [ ] **Step 1: Dry run (no sync) and review drift warnings**

Run: `python tools/build_taxonomy.py --no-sync`
Expected: prints any `DRIFT:` lines, the validation count, and `Wrote .../published/taxonomy.json (... methods: ...)`. First-run drift warnings are expected where a claimed marker isn't literally present in a test file — for each warning, either correct `taxonomy.yaml` (fix the framework list or status) or confirm the marker truly is absent and adjust. Re-run until the warnings reflect reality.

- [ ] **Step 2: Inspect the artifact**

Run: `python -c "import json; d=json.load(open('published/taxonomy.json')); print(d['summary']); print(len(d['categories']),'categories')"`
Expected: a summary dict and `7 categories`.

- [ ] **Step 3: Commit the artifact**

```bash
git add published/taxonomy.json
git commit -m "chore: generate published taxonomy.json"
```

---

## Task 6: Document the regenerate command

**Files:**
- Modify: `COVERAGE.md` (append a section)

- [ ] **Step 1: Append a regeneration note to `COVERAGE.md`**

Add at the end of `COVERAGE.md`:

```markdown
---

## Published taxonomy (blackdiamondconsulting.ai)

The interactive attack-taxonomy matrix on the BDC site is generated from
`taxonomy.yaml` (the source of truth) by `tools/build_taxonomy.py`.

```bash
# Regenerate published/taxonomy.json and sync it into the BDC repo's data/ dir:
python tools/build_taxonomy.py                 # default --bdc-path ../blackdiamondconsulting.ai
python tools/build_taxonomy.py --no-sync       # eval repo only, skip the BDC copy
```

The generator prints `DRIFT:` warnings when `taxonomy.yaml` claims coverage that
the test suite does not back up (missing test file or missing marker), keeping the
published map honest as coverage grows. It never commits or pushes the BDC repo —
run `hugo server` there to verify, then commit & push BDC manually.
```

- [ ] **Step 2: Commit**

```bash
git add COVERAGE.md
git commit -m "docs: document taxonomy regeneration in COVERAGE.md"
```

---

## Task 7: BDC page stub + layout (in the BDC repo)

> **Repo switch:** these files live in `c:\Users\seany\projects\blackdiamondconsulting.ai`. Do **not** commit or push this repo — the user verifies and pushes it themselves (Task 8).

**Files:**
- Create: `content/resources/llm-attack-taxonomy.md`
- Create: `layouts/_default/taxonomy.html`

- [ ] **Step 1: Sync the data into BDC**

From the eval repo, run: `python tools/build_taxonomy.py`
Expected: `Synced .../blackdiamondconsulting.ai/data/taxonomy.json`.

- [ ] **Step 2: Create the content stub**

Create `c:\Users\seany\projects\blackdiamondconsulting.ai\content\resources\llm-attack-taxonomy.md`:

```markdown
---
title: "LLM Attack Taxonomy"
description: "An interactive map of LLM attack vectors and methods, and how Black Diamond Consulting assesses against each."
layout: "taxonomy"
tag: "Reference"
date: 2026-06-09
---
```

- [ ] **Step 3: Create the layout**

Create `c:\Users\seany\projects\blackdiamondconsulting.ai\layouts\_default\taxonomy.html` with the full contents below. It extends the theme's `baseof.html` (`main` block), renders `.Site.Data.taxonomy` server-side, and keeps all JS template-free by driving filters off `data-*` attributes.

```go-html-template
{{ define "main" }}
{{ $t := .Site.Data.taxonomy }}
<section class="section">
  <div class="section-inner">
    <a href="/resources/" class="back-link">&larr; All resources</a>
    <p class="section-tag">Attack taxonomy</p>
    <h1>{{ .Title }}</h1>
    <p class="body-text">{{ .Description }}</p>
    <p class="article-meta">Curated by Sean Yunt — Founder &amp; Principal, Black Diamond Consulting</p>

    <div id="llm-taxonomy">
      <div class="tax-summary">
        <span class="tax-chip tax-badge-covered">{{ $t.summary.covered }} covered</span>
        <span class="tax-chip tax-badge-in_depth">{{ $t.summary.in_depth }} in-depth</span>
        <span class="tax-chip tax-badge-expanding">{{ $t.summary.expanding }} expanding</span>
      </div>

      <div class="tax-controls">
        <input type="search" id="tax-search" placeholder="Search methods…" aria-label="Search methods">
        <select id="tax-status" aria-label="Filter by status">
          <option value="">All statuses</option>
          <option value="covered">Covered</option>
          <option value="in_depth">In-depth</option>
          <option value="expanding">Expanding</option>
        </select>
        <select id="tax-framework" aria-label="Filter by framework">
          <option value="">All frameworks</option>
          {{ range $t.frameworks }}<option value="{{ . }}">{{ . }}</option>{{ end }}
        </select>
      </div>

      {{ range $t.categories }}
      <div class="tax-category" data-category="{{ .category }}">
        <h2>{{ .category }}</h2>
        <p class="body-text">{{ .description }}</p>
        <table class="tax-table">
          <thead><tr><th>Method</th><th>Status</th><th>Frameworks</th></tr></thead>
          <tbody>
          {{ range .methods }}
            <tr class="tax-row" data-status="{{ .status }}" data-frameworks="{{ delimit .frameworks " " }}" data-name="{{ lower .name }}">
              <td><strong>{{ .name }}</strong><div class="tax-blurb">{{ .blurb }}</div></td>
              <td><span class="tax-badge tax-badge-{{ .status }}">{{ .status_label }}</span></td>
              <td>{{ range .frameworks }}<span class="tax-fw">{{ . }}</span> {{ end }}</td>
            </tr>
          {{ end }}
          </tbody>
        </table>
      </div>
      {{ end }}
      <p class="tax-empty" hidden>No methods match those filters.</p>
    </div>
  </div>

  <div class="section-inner section-inner-narrow final-cta" style="margin-top:48px;">
    <p class="section-tag">Worried about one of these?</p>
    <h2>Get a written read on your AI's exposure.</h2>
    <p class="body-text">Send a short description of your AI system and I'll reply with the risks I'd check first — free, no call required.</p>
    <div style="margin-top:24px;">
      <a href="/risk-assessment/" class="btn-primary">Request my free assessment &rarr;</a>
    </div>
  </div>
</section>

<style>
#llm-taxonomy { margin-top: 24px; }
#llm-taxonomy .tax-summary { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:20px; }
#llm-taxonomy .tax-chip { padding:6px 12px; border-radius:999px; font-size:0.85rem; font-weight:500; }
#llm-taxonomy .tax-controls { display:flex; gap:12px; flex-wrap:wrap; margin-bottom:28px; }
#llm-taxonomy .tax-controls input, #llm-taxonomy .tax-controls select {
  padding:8px 12px; border:1px solid #ccc; border-radius:8px; font:inherit; }
#llm-taxonomy .tax-controls input { flex:1; min-width:200px; }
#llm-taxonomy .tax-category { margin-bottom:36px; }
#llm-taxonomy .tax-table { width:100%; border-collapse:collapse; margin-top:12px; }
#llm-taxonomy .tax-table th, #llm-taxonomy .tax-table td {
  text-align:left; padding:10px 12px; border-bottom:1px solid #eee; vertical-align:top; }
#llm-taxonomy .tax-table th { font-size:0.8rem; text-transform:uppercase; letter-spacing:0.04em; color:#777; }
#llm-taxonomy .tax-blurb { color:#666; font-size:0.9rem; margin-top:4px; }
#llm-taxonomy .tax-badge { padding:3px 10px; border-radius:999px; font-size:0.8rem; font-weight:600; white-space:nowrap; }
#llm-taxonomy .tax-badge-covered  { background:#e6f4ea; color:#1e7e34; }
#llm-taxonomy .tax-badge-in_depth  { background:#e3edfb; color:#1857b8; }
#llm-taxonomy .tax-badge-expanding { background:#fdf2e2; color:#a86412; }
#llm-taxonomy .tax-fw { display:inline-block; background:#f1f1f4; color:#555; border-radius:6px;
  padding:2px 7px; font-size:0.78rem; margin:2px 2px 0 0; }
#llm-taxonomy .tax-empty { color:#777; font-style:italic; padding:16px 0; }
</style>

<script>
(function () {
  var root = document.getElementById('llm-taxonomy');
  if (!root) return;
  var search = document.getElementById('tax-search');
  var statusSel = document.getElementById('tax-status');
  var fwSel = document.getElementById('tax-framework');
  var rows = Array.prototype.slice.call(root.querySelectorAll('.tax-row'));
  var categories = Array.prototype.slice.call(root.querySelectorAll('.tax-category'));
  var empty = root.querySelector('.tax-empty');

  function apply() {
    var q = (search.value || '').trim().toLowerCase();
    var st = statusSel.value;
    var fw = fwSel.value;
    var anyVisible = false;

    rows.forEach(function (row) {
      var matchQ = !q || row.getAttribute('data-name').indexOf(q) !== -1
        || row.textContent.toLowerCase().indexOf(q) !== -1;
      var matchSt = !st || row.getAttribute('data-status') === st;
      var fwList = (row.getAttribute('data-frameworks') || '').split(' ');
      var matchFw = !fw || fwList.indexOf(fw) !== -1;
      var show = matchQ && matchSt && matchFw;
      row.hidden = !show;
      if (show) anyVisible = true;
    });

    categories.forEach(function (cat) {
      var visibleRows = cat.querySelectorAll('.tax-row:not([hidden])').length;
      cat.hidden = visibleRows === 0;
    });
    if (empty) empty.hidden = anyVisible;
  }

  search.addEventListener('input', apply);
  statusSel.addEventListener('change', apply);
  fwSel.addEventListener('change', apply);
})();
</script>
{{ end }}
```

- [ ] **Step 4: Build to verify the template compiles**

From the BDC repo root, run: `hugo --gc --minify`
Expected: build succeeds with no template errors and reports a page at `resources/llm-attack-taxonomy/index.html`. If the build errors with "can't evaluate field ... in type ... Site.Data", confirm `data/taxonomy.json` exists (Step 1). If the page renders with no layout/styling, confirm the file is at `layouts/_default/taxonomy.html` and front matter has `layout: "taxonomy"`.

---

## Task 8: User verification gate (BDC)

- [ ] **Step 1: Hand off to the user for visual verification**

Tell the user:

> The taxonomy page is built in the BDC repo but **not committed**. Please verify it yourself:
> ```bash
> cd ../blackdiamondconsulting.ai
> hugo server
> ```
> Open http://localhost:1313/resources/llm-attack-taxonomy/ and confirm: the table renders, the summary chips show counts, the three filters (search / status / framework) work, nav + footer are present, and the closing CTA links to /risk-assessment/. When you're happy:
> ```bash
> git add content/resources/llm-attack-taxonomy.md layouts/_default/taxonomy.html data/taxonomy.json
> git commit -m "feat: add LLM attack taxonomy page"
> git push
> ```

Do not commit or push the BDC repo on the user's behalf.

---

## Self-Review

**Spec coverage:**
- Source of truth `taxonomy.yaml` → Task 1. ✓
- Generator + drift validation + sync, `--bdc-path`/`--no-sync`/`--source-base-url` flags → Tasks 2–4. ✓
- Status vocabulary covered/in_depth/expanding (roadmap-hybrid framing) → Task 1 + `STATUS_LABELS`. ✓
- Categorized table, not 2D grid → Task 7 layout. ✓
- No source deep-linking by default (`--source-base-url` empty ⇒ `source_links: []`) → Task 2 tests + impl. ✓
- Build-time render via `.Site.Data.taxonomy`, inline CSS/JS, no CSP/`_headers` change, scoped under `#llm-taxonomy` → Task 7. ✓
- Attribution byline + CTA to `/risk-assessment/` → Task 7. ✓
- Generator never commits/pushes BDC; user verifies via `hugo server` → Tasks 4, 7 (no-commit note), 8. ✓
- Versioned artifact `published/taxonomy.json` → Tasks 4–5. ✓
- Regeneration documented → Task 6. ✓

**Placeholder scan:** none — all code blocks are complete, all commands have expected output.

**Type consistency:** `load_taxonomy`, `to_render_json` (kw-only `source_base_url`), `validate` (kw-only `repo_root`), `write_artifact`, `sync` (kw-only `bdc_path`), `main` are referenced consistently across tasks and tests. JSON keys (`summary`, `frameworks`, `categories`, `status`, `status_label`, `source_links`) match between generator output (Task 2/4) and the layout's field access (Task 7). Status keys (`covered`/`in_depth`/`expanding`) match between YAML, `STATUS_LABELS`, CSS classes (`tax-badge-<status>`), and the status filter `<option>` values.
