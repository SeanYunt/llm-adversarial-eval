# Framework Coverage

This test suite maps to four industry standards: **OWASP LLM Top 10 (2025)**, **NIST AI 100-2**, **MITRE ATLAS**, and **EU AI Act** (high-risk obligations effective August 2026).

Run tests by standard:

| Standard | Command |
|---|---|
| All OWASP LLM Top 10 | `pytest -m "owasp_llm01 or owasp_llm02 or owasp_llm04 or owasp_llm06 or owasp_llm07 or owasp_llm08 or owasp_llm09 or owasp_llm10"` |
| All NIST AI 100-2 | `pytest -m "nist_evasion or nist_poisoning or nist_privacy or nist_abuse"` |
| All MITRE ATLAS | `pytest -m "mitre_recon or mitre_llm_injection or mitre_llm_jailbreak or mitre_exfiltration or mitre_data_poisoning"` |
| All EU AI Act | `pytest -m "eu_ai_art9 or eu_ai_art13 or eu_ai_art14 or eu_ai_art15"` |

Individual marks (examples):

```bash
pytest -m owasp_llm01          # Prompt Injection
pytest -m eu_ai_art14          # Human oversight / escalation
pytest -m eu_ai_art9           # Bias and discriminatory outputs
pytest -m mitre_recon          # MITRE ATLAS Reconnaissance
pytest -m nist_privacy         # NIST AI 100-2 Privacy/extraction
```

---

## Framework Coverage Summary

| Framework | Status | Coverage |
|---|---|---|
| OWASP LLM Top 10 (2025) | ✅ Covered | 7/10 risks (LLM03 out of scope; LLM05 pending; LLM10 behavioral only — see below) |
| NIST AI 100-2 | ✅ Covered | 4/4 attack classes |
| MITRE ATLAS | ✅ Covered | 5 techniques across 2 tactics |
| EU AI Act (High-Risk) | ✅ Covered | Art. 9, 13, 14, 15 |

---

## OWASP LLM Top 10 Mapping

| Risk | Mark | Test Files | Notes |
|---|---|---|---|
| LLM01 Prompt Injection | `owasp_llm01` | test_jailbreak.py, test_indirect_injection.py, test_hidden_text_injection.py, test_payload_splitting.py, test_virtualization_attacks.py, test_healthcare_chatbot.py (FHIR), test_logistics_chatbot.py | Direct, indirect, RAG, hidden-text, and domain-specific injection |
| LLM02 Sensitive Info Disclosure | `owasp_llm02` | test_secret_extraction.py, test_system_prompt_leakage.py, test_cross_user_exfiltration.py, test_healthcare_chatbot.py, test_underwriting_assist.py | System prompt leakage, PHI exfiltration, cross-user leakage, IP/trade-secret leakage across employees in a patent-management RAG context |
| LLM03 Supply Chain | _omitted_ | — | Not testable at inference time; addressed in deployment/procurement audit |
| LLM04 Data/Model Poisoning | `owasp_llm04` | test_indirect_injection.py | Adversarial content injected via retrieved documents |
| LLM05 Improper Output Handling | `owasp_llm05` | _(reserved for future output-parsing tests)_ | No current behavioral test; mark registered |
| LLM06 Excessive Agency | `owasp_llm06` | test_specification_gaming.py, test_competing_objectives.py, test_agentic_boundary.py | Reward hacking, misaligned objectives, unauthorized agentic actions |
| LLM07 System Prompt Leakage | `owasp_llm07` | test_system_prompt_leakage.py, test_secret_extraction.py, test_model_reconnaissance.py | Structural and content leakage |
| LLM08 Vector/Embedding Weaknesses | `owasp_llm08` | test_indirect_injection.py, test_cross_user_exfiltration.py, test_underwriting_assist.py | RAG/retrieval attack surface |
| LLM09 Misinformation | `owasp_llm09` | test_factual_hallucinations.py, test_sycophancy.py, test_airline_chatbot.py, test_product_support_chatbot.py, test_legal_chatbot.py, test_financial_chatbot.py, test_policy_qna_chatbot.py, test_ip_confidentiality_disclosure.py, test_privacy_policy_accuracy.py, test_ip_risk_warnings.py | Factual fabrication, sycophantic capitulation, domain hallucinations, false privacy assurances, hallucinated data-handling policy details, omitted IP/patent disclosure warnings |
| LLM10 Unbounded Consumption | `owasp_llm10` | test_agentic_boundary.py (TestUnboundedConsumption) | Behavioral dimension only — see Out of Scope note |

---

## NIST AI 100-2 Mapping

| Attack Class | Mark | Test Files | Notes |
|---|---|---|---|
| Evasion (adversarial inputs at inference) | `nist_evasion` | test_jailbreak.py, test_hidden_text_injection.py, test_payload_splitting.py, test_virtualization_attacks.py | Persona attacks, hidden text, payload fragmentation, eval-mode framing |
| Poisoning (corrupting data the model processes) | `nist_poisoning` | test_indirect_injection.py | Adversarial RAG document injection |
| Privacy/Extraction | `nist_privacy` | test_secret_extraction.py, test_system_prompt_leakage.py, test_cross_user_exfiltration.py, test_model_reconnaissance.py, test_ip_confidentiality_disclosure.py, test_privacy_policy_accuracy.py | System prompt, user data, training provenance extraction, false privacy assurances, hallucinated data-handling policy details, IP/trade-secret cross-user leakage |
| Abuse (misusing the model as a tool) | `nist_abuse` | test_specification_gaming.py, test_competing_objectives.py, test_agentic_boundary.py, test_ai_identity_disclosure.py | Reward hacking, misaligned objectives, unauthorized actions, false identity claims |

---

## MITRE ATLAS Mapping

| Tactic / Technique | Mark | Test Files | Notes |
|---|---|---|---|
| AML.TA0002 Reconnaissance | `mitre_recon` | test_system_prompt_leakage.py, test_model_reconnaissance.py, test_underwriting_assist.py | Mapping restrictions, enumerating capabilities, extracting guidelines |
| AML.T0054 LLM Prompt Injection | `mitre_llm_injection` | test_jailbreak.py (TestPromptInjection), test_indirect_injection.py, test_hidden_text_injection.py | Direct, indirect, and hidden injection |
| AML.T0051 LLM Jailbreak | `mitre_llm_jailbreak` | test_jailbreak.py, test_payload_splitting.py, test_virtualization_attacks.py | Persona attacks, payload splitting, eval-mode framing |
| AML.T0025 Exfiltration via ML Inference API | `mitre_exfiltration` | test_secret_extraction.py, test_cross_user_exfiltration.py | System prompt secrets, cross-user PHI and IP/trade-secret data |
| AML.T0043.003 Training Data Poisoning | `mitre_data_poisoning` | test_indirect_injection.py | Adversarial document injection into RAG context |

---

## EU AI Act Mapping

| Article | Obligation | Mark | Test Files | Notes |
|---|---|---|---|---|
| Art. 9 | Risk management — bias and discriminatory outputs | `eu_ai_art9` | test_underwriting_assist.py (TestDiscriminatoryFactors), test_financial_chatbot.py (TestBiasFairness), test_healthcare_chatbot.py (TestBiasFairness), test_legal_chatbot.py (TestBiasFairness) | Pairwise demographic-signal differential tests |
| Art. 13 | Transparency — AI identity disclosure | `eu_ai_art13` | test_ai_identity_disclosure.py, test_ip_confidentiality_disclosure.py, test_ip_risk_warnings.py, test_legal_chatbot.py (TestIPPatentDisclosureRisk) | Direct, persona-pressure, emotional, and professional-claim scenarios; false data-handling transparency; omitted IP/patent disclosure warnings; legal AI failing to flag that sharing invention details in-conversation may constitute prior art or compromise trade-secret status |
| Art. 14 | Human oversight — escalation to human | `eu_ai_art14` | test_airline_chatbot.py, test_healthcare_chatbot.py, test_legal_chatbot.py, test_financial_chatbot.py, test_auto_claims_chatbot.py, test_home_claims_chatbot.py, test_commercial_quoting_chatbot.py, test_underwriting_assist.py, test_logistics_chatbot.py, test_policy_qna_chatbot.py, test_agentic_boundary.py | Must-escalate, no-false-escalation, and escalation-under-suppression per domain |
| Art. 15 | Accuracy, robustness, cybersecurity | `eu_ai_art15` | All domain chatbot files (test_airline_chatbot.py, test_healthcare_chatbot.py, test_logistics_chatbot.py, test_container_shipping_chatbot.py, test_air_freight_chatbot.py, test_legal_chatbot.py, test_financial_chatbot.py, test_auto_claims_chatbot.py, test_home_claims_chatbot.py, test_commercial_quoting_chatbot.py, test_underwriting_assist.py, test_policy_qna_chatbot.py, test_product_support_chatbot.py), test_factual_hallucinations.py, test_sycophancy.py, test_refusal_calibration.py, test_specification_gaming.py, test_competing_objectives.py | Broad accuracy and robustness coverage |

---

## Out of Scope

**OWASP LLM03 (Supply Chain Vulnerabilities):** Not testable at inference time. Compromised model weights, poisoned training data, and vulnerable dependencies are deployment and procurement audit concerns outside the scope of behavioral prompt testing.

**OWASP LLM10 (Unbounded Consumption) — infrastructure dimension:** Token exhaustion at the API gateway level, prompt flooding, and rate-limit bypass are infrastructure concerns. `TestUnboundedConsumption` in `test_agentic_boundary.py` covers the behavioral dimension only (model refusing to attempt unbounded output generation).

**OWASP LLM05 (Improper Output Handling) — behavioral tests pending:** Downstream injection via structured output (JSON/code generation exploits) and insufficient sanitization are testable at inference time but require a dedicated output-parsing harness. The mark `owasp_llm05` is registered; tests will be added in a future iteration. `pytest -m owasp_llm05` currently collects zero tests.

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
the test suite does not back up (a missing test file, or a framework marker the
cited file does not carry), keeping the published map honest as coverage grows.
It never commits or pushes the BDC repo — run `hugo server` there to verify, then
commit & push BDC manually.
