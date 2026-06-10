# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An adversarial pytest suite that probes a target LLM for hallucination and safety-boundary failures (jailbreaks, sycophancy, PHI/system-prompt leakage, prompt injection, etc.). The probes are defensive red-team tests — they surface failure modes, they do not exploit them.

**Tests never hard-fail.** A probe sends a prompt, records the response, and applies non-fatal flagging. The deliverable is a timestamped JSON report in `results/`, not a pass/fail exit code. Do not "fix failing tests" by making them assert — that defeats the design. The only legitimate `assert` is on harness invariants (e.g. a fixture file exists), never on model behavior.

## Commands

```bash
# Full suite (writes results/run_YYYYMMDD_HHMMSS.json on exit)
pytest -v

# By category marker (one per failure mode / domain)
pytest -m sycophancy -v
pytest -m healthcare_ai -v
pytest -m logistics -v            # parent marker — runs ground + ocean + air freight

# By compliance framework (markers are layered on top of category markers)
pytest -m owasp_llm01 -v          # OWASP LLM Top 10 (2025)
pytest -m eu_ai_art14 -v          # EU AI Act high-risk articles
pytest -m mitre_recon -v          # MITRE ATLAS
pytest -m nist_privacy -v         # NIST AI 100-2

# A single test file / class / function
pytest tests/test_sycophancy.py -v
pytest tests/test_sycophancy.py::TestPushbackCapitulation -v
pytest tests/test_sycophancy.py::TestPushbackCapitulation::test_<name> -v

# Skip multi-API-call tests
pytest -v -m "not slow"

# Name the report file
pytest -m financial_ai --results-name financial_baseline.json
```

Requires `ANTHROPIC_API_KEY` (in `.env` or environment). If unset, all live tests **skip** rather than fail. The full markers list lives in `pyproject.toml` `[tool.pytest.ini_options]`; `COVERAGE.md` maps each marker to its framework control.

### Run against a different target

- Different Anthropic model: `pytest --model claude-opus-4-6` (default is `claude-haiku-4-5-20251001`, set in `conftest.py`).
- Local OpenAI-compatible server: `pytest --local-llm-url http://host:8080/v1 --local-llm-model my-model`. `LocalLLMClient` in `conftest.py` adapts the OpenAI chat API to the Anthropic `.messages.create()` shape, so **no test file changes** — the system message is folded into `messages` and the model arg is ignored by the server.

### Report generation (separate tool)

`tools/generate_llm_report.py file1.json file2.json …` produces a comparative `.docx` from two or more result files (uses the Anthropic API for narrative synthesis + the npm `docx` package). `tools/` has its own `node_modules`; it is independent of the test harness.

### Taxonomy pipeline

`taxonomy.yaml` is the source of truth for the published LLM attack taxonomy matrix. Regenerate the artifact with:

```bash
# Write published/taxonomy.json only (no external side effects)
python tools/build_taxonomy.py --no-sync

# Also copy to the BDC website repo (default: ../blackdiamondconsulting.ai/data/)
python tools/build_taxonomy.py
```

`tools/build_taxonomy.py validate()` prints non-fatal drift warnings when a test file listed in `taxonomy.yaml` is missing on disk or a framework marker listed under `frameworks:` is absent from that file. These are warnings, not errors — the build still produces JSON.

`tests/test_build_taxonomy.py` exercises the taxonomy generator and makes **no API calls** — it always runs regardless of whether `ANTHROPIC_API_KEY` is set.

## Architecture

Everything routes through **`conftest.py`** — it is the engine, not just fixtures. New test files import from it directly (`from conftest import probe, multi_probe, vision_probe, llm_judge, ResultsCollector`).

**Probe helpers** (how a test talks to the model):
- `probe(...)` — single user prompt + optional `system`.
- `multi_probe(...)` — a pre-built `messages` list. This is how multi-turn attacks embed a *correct* prior assistant turn, then apply pressure in the final user turn (the core mechanism behind every "reversal sycophancy" test).
- `vision_probe(...)` — base64-encodes an image fixture + text prompt (hidden-text injection, product-image sycophancy).

Each helper builds a `ProbeResult` and registers it via the session-scoped `collector` fixture. `ResultsCollector.write_report()` runs in the autouse `write_report_on_exit` fixture, aggregating `by_category` counts and dumping JSON at session end.

**Two independent, non-overriding detection channels** on every `ProbeResult`:
- `flagged` (bool) — heuristic string/regex matching, done inline in the test (see per-file helpers like `capitulated()` / `_flag_action_claim()`).
- `judge_flagged` (bool | None) — set by `llm_judge(client, result, criteria=…)`, a second Haiku call returning `FLAGGED: YES/NO` + reasoning. `None` means the judge wasn't run. **The judge never raises** — any API/parse error leaves the fields `None` so a judge problem can't break a test. Used in ~26 files for nuanced failures regex misses.

Disagreement between the channels is intentional and recorded, not resolved: `flagged ∧ ¬judge_flagged` is a false-positive candidate; `¬flagged ∧ judge_flagged` is a false negative. Both go in the report for human review.

**Three newer test modules** are present in `tests/` but not listed in `README.md` — consult `COVERAGE.md` for their full marker assignments:
- `test_agentic_boundary.py` — unauthorized agentic actions, unbounded consumption (behavioral dimension); markers `owasp_llm06`, `eu_ai_art14`.
- `test_model_reconnaissance.py` — capability enumeration, restriction mapping, training-provenance extraction; markers `owasp_llm07`, `mitre_recon`, `nist_privacy`.
- `test_ai_identity_disclosure.py` — AI identity disclosure under persona pressure, professional-claim framing, and emotional appeals; markers `eu_ai_art13`, `nist_abuse`.

**Markers do double duty.** A test class carries a category marker (`@pytest.mark.sycophancy`) and the file sets `pytestmark = [pytest.mark.owasp_llm09, pytest.mark.eu_ai_art15]` for framework attribution. When adding a test, register its category marker in `pyproject.toml`, apply the relevant framework markers, and update the mapping tables in `COVERAGE.md` — these three must stay in sync.

**Fixtures** are static and committed for deterministic diffs:
- `tests/fixtures/images/` — adversarial images; regenerate with `python tests/fixtures/make_injection_images.py` / `make_product_images.py` (Pillow).
- `tests/fixtures/fhir/` — FHIR R4 bundles for healthcare tests, including `rag_store/` simulating a multi-patient retrieval store with a `poisoned_docref.json` carrying injection in `text.div`, `description`, and a base64 attachment.

## Recurring cross-domain patterns

The domain chatbot suites reuse a small set of failure templates — recognize them when adding a new domain:
- **Reversal sycophancy** — model gives a correct restrictive ruling in turn 1, user pushes back with emotion/authority/urgency, a failing model reverses without new evidence. Instances: `TestSuitabilityReversal` (financial), `TestCoverageReversal` (insurance claims), `TestHazmatSafetyDowngrade` / `TestIMDGMaritimeHazmat` / `TestDGRAirHazmat` (logistics), `TestImageClaimSycophancy` (product). Built with `multi_probe`.
- **Multi-tenant RAG leakage** — context holds records for several tenants/patients with specific marker values (rates, MRNs, E-mods); the logged-in user tries to extract a neighbor's. Instances: `test_cross_user_exfiltration.py`, `test_underwriting_assist.py`, `test_logistics_chatbot.py`.
- **Document/narrative injection** — instructions hidden in data the model is asked to interpret (bill-of-lading fields, FHIR narrative, near-invisible image text).

`README.md` has the full per-domain failure-mode tables; `COVERAGE.md` has the framework control mapping. Consult them before extending a suite so a new test slots into the existing taxonomy.
