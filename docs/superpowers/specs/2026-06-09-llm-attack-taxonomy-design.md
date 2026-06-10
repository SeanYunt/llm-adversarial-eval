# LLM Attack Taxonomy — Interactive Coverage Matrix

**Date:** 2026-06-09
**Status:** Design approved, pending spec review

## Goal

Publish an interactive, curated taxonomy of LLM attack vectors and methods to the
Black Diamond Consulting site (`blackdiamondconsulting.ai`), with each entry showing
how thoroughly the `llm-adversarial-eval` suite assesses it. The page doubles as a
credibility asset (a navigable OWASP/MITRE/NIST/EU-AI-Act map) and a lead-generation
surface (attribution + contact/self-assessment CTAs).

## Context: two repos, one boundary

| Repo | Role |
|---|---|
| `llm-adversarial-eval` | pytest suite + **source of truth** for the taxonomy and coverage status |
| `blackdiamondconsulting.ai` | Hugo static site (Cloudflare Pages) that **renders** the taxonomy |

The two deploy independently. The BDC site builds static HTML at deploy time, and its
CSP (`connect-src 'self' …`) blocks cross-origin data fetches, so the published matrix
cannot read the eval repo at runtime. The matrix is therefore a **snapshot** that goes
stale until a sync step carries new data across and a BDC rebuild is triggered. The
design separates data (eval repo) from renderer (BDC) so this boundary is explicit and
the sync is a single, deliberate command.

## Decisions

1. **Integration style:** Hugo page under `/resources/` using a project-level custom
   layout that extends the `bdc` theme's `baseof.html`, inheriting nav + footer (which
   carry the existing contact and self-assessment CTAs). The site already uses
   per-page custom layouts (`risk-assessment.html`, `ai-assistant.html`), so this is
   the native pattern, not a workaround.
2. **Sync:** one-command generator script in the eval repo. Copies `taxonomy.json` into
   BDC's `data/` and stops — never stages, commits, or pushes.
3. **Framing (client-facing):** roadmap-hybrid vocabulary — `covered` / `in_depth` /
   `expanding`. "Not yet built" surfaces as `expanding` ("on the assessment roadmap"),
   never as "missing" or "gap."
4. **Shape:** categorized, filterable table — **not** a 2D method×payload grid. The
   current suite entangles method and payload, so a literal grid would misrepresent
   coverage. Rows are methods grouped by vector category; columns are status +
   framework tags.
5. **No source deep-linking by default.** "Covered" rows do not link to the test files
   on a public sales page (that would expose working adversarial prompts one click off
   the marketing site). Source-linking sits behind a configurable, off-by-default base
   URL.
6. **No `_headers`/CSP change.** Page is self-contained inline HTML/CSS/JS; the existing
   CSP already permits `'unsafe-inline'` for script and style.
7. **Local verification gate.** The user runs `hugo server` in the BDC repo and visually
   verifies the rendered page before committing/pushing BDC. The generator does not
   automate any git action in the BDC repo.

## Components

### 1. `llm-adversarial-eval/taxonomy.yaml` (source of truth)

Hand-curated. Top level is a list of vector categories; each holds a list of methods.

```yaml
- category: Jailbreak & Injection
  description: Input-side manipulation to elicit policy-violating output.
  methods:
    - name: Many-shot jailbreaking
      status: expanding                 # covered | in_depth | expanding
      frameworks: [owasp_llm01, mitre_llm_jailbreak, nist_evasion]
      blurb: Flooding context with 128–256 fake exchanges to erode refusal.
      tests: []                         # list of test files when covered/in_depth
    - name: Direct persona / DAN
      status: covered
      frameworks: [owasp_llm01, mitre_llm_jailbreak, nist_evasion]
      blurb: Alter-ego personas instructed to ignore restrictions.
      tests: [tests/test_jailbreak.py]
```

**Status vocabulary:**
- `covered` — at least one working probe exists for this method.
- `in_depth` — multiple probes / multi-technique coverage (e.g. secret extraction's
  direct + encoded + roleplay + indirect + multi-turn classes).
- `expanding` — on the roadmap; no probe yet.

**Initial categories** (seeded from the verified gap analysis):
Jailbreak & Injection · Information Extraction · Output Handling · Agentic / Tool-Use ·
Multimodal · Generation-Quality Harms · Content-Harm Payloads.

`frameworks` values reuse the marker names already registered in `pyproject.toml` /
`COVERAGE.md`, so the taxonomy and the compliance map stay in one vocabulary.

### 2. `llm-adversarial-eval/tools/build_taxonomy.py` (generator + validator + sync)

One command, run from the eval repo root. Behavior:

1. **Parse** `taxonomy.yaml`.
2. **Drift-validate against reality** and print warnings (non-fatal):
   - an entry with `status: covered`/`in_depth` whose `tests:` file(s) do not exist;
   - a `tests:` file that does not contain the claimed framework marker(s);
   - a marker registered in `pyproject.toml` that no taxonomy entry references.
3. **Emit** render-ready `taxonomy.json` (categories → methods → status/frameworks/blurb,
   plus computed per-category and overall status counts for the page summary).
4. **Sync:** copy `taxonomy.json` to `<bdc-path>/data/taxonomy.json`, then print explicit
   next steps ("run `hugo server` in the BDC repo, verify, then commit & push"). It does
   **not** run any git command in the BDC repo.

**Flags:**
- `--bdc-path` (default `../blackdiamondconsulting.ai`) — sync target repo.
- `--no-sync` — generate `taxonomy.json` in the eval repo only, skip the copy.
- `--source-base-url` (default empty) — when set, "covered" rows render as links to that
  base URL + test path; empty ⇒ plain text, no links.

The generated `taxonomy.json` is also written into the eval repo (e.g.
`published/taxonomy.json`) so the artifact is versioned alongside its source.

### 3. BDC page

- **Content stub:** `content/resources/llm-attack-taxonomy.md` — front matter only
  (`title`, `description`, `layout: taxonomy`, resources-section metadata). Body is not
  used for the matrix (Goldmark runs at defaults and strips raw HTML — irrelevant here
  because rendering happens in the layout).
- **Layout:** `layouts/resources/taxonomy.html` (project-level override; the theme is not
  modified). Uses `{{ define "main" }}` so `baseof.html` wraps it with nav + footer.
- **Rendering:** reads `.Site.Data.taxonomy` (from `data/taxonomy.json`) and emits the
  table rows **at build time** — content lives in the HTML (CSP-proof, SEO-friendly).
- **Interactivity (inline JS only):** filter by category / status / framework, free-text
  search, status-count summary, tooltips on blurbs. No external libraries (CSP, and to
  avoid a build step).
- **Styling:** all rules scoped under `#llm-taxonomy` so they cannot collide with the
  theme.
- **Attribution + CTA:** intro byline ("Curated by Sean Yunt — Founder & Principal,
  Black Diamond Consulting") and a closing CTA block linking to the contact form +
  `/risk-assessment/`. Exact CTA target URLs confirmed by reading the theme's
  `nav.html` / `footer.html` partials during implementation.

## Data flow

```
taxonomy.yaml ──build_taxonomy.py──▶ taxonomy.json ──copy──▶ BDC data/taxonomy.json
   (eval repo, curated)                (validated)            (Hugo build input)
                                                                     │
                                            hugo build ──▶ static HTML (.Site.Data.taxonomy)
                                                                     │
                                          user: hugo server + visual verify ──▶ commit & push BDC
```

## Verification

- **Generator:** pytest tests in the eval repo — parses a sample `taxonomy.yaml`, emits
  the expected JSON shape, computes correct status counts, and fires each drift warning
  (missing test file, missing marker, orphaned marker).
- **Page:** the user runs `hugo server` locally in the BDC repo and visually verifies a
  clean build and correct render before committing/pushing. (Explicit user gate — not
  automated.)

## Scope / out of scope

**In:** `taxonomy.yaml`, the generator/validator/sync script + its tests, the BDC content
stub + layout + scoped CSS/JS, attribution + CTA block.

**Out (YAGNI):** 2D method×payload grid; live cross-repo runtime fetch; cross-repo CI
automation (deliberate later upgrade from the one-command script); any `_headers`/CSP
change; modifications to the `bdc` theme itself.

## Open items (resolved during implementation, not blocking)

- Exact CTA target URLs — read from the theme's nav/footer partials.
- Confirm the eval repo's intended GitHub visibility before ever enabling
  `--source-base-url` (kept empty/off by default regardless).
