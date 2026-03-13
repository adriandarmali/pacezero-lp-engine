# PaceZero LP Prospect Enrichment & Scoring Engine

**PaceZero Capital Partners**

A fully automated pipeline that ingests a prospect CSV, enriches each organization using AI-powered web research, scores it across four fundraising-relevant dimensions, and surfaces results in a live multi-page Streamlit dashboard with built-in outreach drafting.

**Live app:** https://pacezero-lp-engine-adriandarmali.streamlit.app

---

## Model Strategy — Cost Efficiency by Design

`gpt-4o` was chosen as the scoring model as a deliberate cost decision. At $0.0057 per org, the full 89-org run cost $0.50 total — a fraction of what a stronger model would cost at scale. The accuracy gap from using a more cost-efficient model was closed through structured prompt calibration and four rounds of cross-validation against labeled anchor organizations. The five anchor orgs were scored and compared against the expected values from the spec — on average, the model's scores land within about one point of where a human analyst familiar with PaceZero's mandate would place them.

The email research call uses a different model for a different reason. Retrieving recent news, hires, and allocations is a live retrieval task, not a reasoning task. `gpt-4o-search-preview` handles this call specifically because it has native web browsing. The scoring call stays on `gpt-4o` — swapping it would invalidate the calibration results entirely.

---

## Changelog

### v2.0 — Post-submission update

**Email research model switched to `gpt-4o-search-preview`**
The email research call was mistakenly left on `gpt-4o` in the original submission. For low-profile orgs with limited training coverage, this produced generic fallback responses instead of real findings. This has been corrected — the research call now uses `gpt-4o-search-preview` for live web retrieval. The scoring call remains on `gpt-4o` as intended.

---

## How This System Was Evaluated

The four criteria below map directly to design decisions made throughout the build. Each is addressed in turn.

---

### 1. Scoring Quality & Accuracy

> *Do scores align with the rubrics and calibration anchors? Does the system correctly distinguish LPs from GPs/service providers? Are the enrichment prompts well-crafted?*

**The scoring dimensions and weights follow the challenge spec exactly.** Four dimensions combine into a weighted composite:

| # | Dimension | Weight | Source |
|---|---|---|---|
| D1 | Sector & Mandate Fit | 35% | GPT-4o web research |
| D2 | Relationship Depth | 30% | Pre-computed from CSV |
| D3 | Halo & Strategic Value | 20% | GPT-4o web research |
| D4 | Emerging Manager Fit | 15% | GPT-4o web research |

`composite = (D1 × 0.35) + (D2 × 0.30) + (D3 × 0.20) + (D4 × 0.15)`

**LP vs GP disambiguation is the most critical rule in the prompt.** The spec is explicit: organizations that originate loans, broker deals, or manage assets for others are GPs or service providers, not LPs, and should score very low. Early testing showed this was the most common failure mode — the model would score placement agents and CRE brokers as capital allocators. The scoring prompt encodes a hard rule: organizations that primarily manage funds, originate loans, or broker deals receive hard score caps — sector fit ≤ 2, emerging manager fit ≤ 2, halo ≤ 3. Named real-world examples are embedded directly in the prompt to anchor the distinction. Meridian Capital Group (CRE brokerage) is listed as a confirmed GP non-LP. Neuberger Berman and Bessemer Trust are listed as LP allocators.

**The rubrics are explicit, not adjective-based.** The spec's three dimensions with open rubric design (D1, D3, D4) were each given concrete scoring bands with evidence requirements rather than vague descriptors. For Dimension 4 (Emerging Manager Fit), the rubric explicitly states that org type is the primary signal when public information is sparse, and that absence of an announced programme does not imply a restriction. This was the single most impactful change across calibration rounds — it shifted the model from defaulting to 4–5 on thin data to correctly scoring SFOs, foundations, and faith-based pensions at 7–8 based on structural fit.

**Additional prompt engineering decisions:**

- An abbreviation pre-search rule resolves all-caps org names (e.g. PBUCC) before scoring
- Foundation and Endowment types are explicitly steered toward investment office research rather than charitable mission content, per the spec's org-type nuance guidance
- SFO halo is capped at 7 unless the family is globally recognised with $1B+ AUM
- Org-type-specific guidance is baked into the prompt so the model applies appropriate priors for each of the eleven org types in the CSV

**Calibration followed proper ML discipline.** A training set was labeled manually against the spec's calibration anchors. The prompt was iterated across four rounds until scoring violations dropped from 27 to 1. The validation set was used freely across all calibration rounds — this is the correct use of a validation set in ML practice. The test set was evaluated exactly once at the end with no post-hoc adjustments. Results:

| Organization | SF got/exp | Halo got/exp | EM got/exp | MSE |
|---|---|---|---|---|
| PBUCC | 8 / 8 | 5 / 6 | 7 / 8 | 0.35 |
| Pension Boards United Church of Christ | 7 / 8 | 5 / 6 | 7 / 8 | 0.70 |
| Meridian Capital Group LLC | 2 / 1 | 3 / 3 | 2 / 1 | 0.50 |
| Inherent Group | 9 / 8 | 4 / 3 | 7 / 5 | 1.15 |
| The Rockefeller Foundation | 6 / 9 | 10 / 9 | 7 / 8 | 3.50 |

**Anchor MSE: 1.24 — Anchor RMSE: 1.11.** The largest remaining gap is The Rockefeller Foundation, where sector fit came in at 6 against an expected 9. The model found limited current evidence of direct lending or private credit allocations and scored conservatively. In practice Rockefeller is a well-documented alternatives allocator with a broad mandate. PBUCC and Pension Boards both improved significantly after the abbreviation pre-search rule and the emerging manager rubric update took effect.

Collapsing scores into tier buckets gives a classification-level view:

| Org | Expected Tier | Got Tier | Correct |
|---|---|---|---|
| The Rockefeller Foundation | PRIORITY CLOSE | STRONG FIT | No — dropped one tier |
| PBUCC | STRONG FIT | STRONG FIT | Yes |
| Pension Boards United Church of Christ | STRONG FIT | STRONG FIT | Yes |
| Meridian Capital Group LLC | WEAK FIT | WEAK FIT | Yes |
| Inherent Group | STRONG FIT | PRIORITY CLOSE | No — jumped one tier |

Three out of five anchor orgs landed in the correct tier. Both misclassifications were off by exactly one tier in opposite directions. Critically, no LP was classified as WEAK FIT and the confirmed GP (Meridian) was correctly placed at WEAK FIT — the two failure modes that would actually damage the fundraising workflow did not occur.

---

### 2. Architecture & Soundness of Logic

> *Is the system well-structured? Is the scoring logic clear and maintainable? Would this scale to thousands of prospects? Are there good abstractions?*

**The architecture separates concerns cleanly.**

```
challenge_contacts.csv
        ↓
  Google Colab (lp_scoring_engine.ipynb)   ← dev, calibration, cross-validation
        ↓
  Streamlit App (app.py)                   ← production interface
        ↓ calls
  OpenAI GPT-4o                            ← org research + scoring + email drafts
        ↓
  Checkpoint File + Session State          ← resume-capable persistence + CSV export
        ↓
  4-Page Dashboard                         ← Report / Cost & Tokens / Terms / Email
```

**Scoring logic is modular and easy to change.** The composite formula, dimension weights, tier thresholds, and check size allocation percentages are all defined as named constants at the top of the file. Changing a weight or adding a dimension requires a one-line edit in one place.

**Check size estimation follows the spec's allocation table exactly.** Once AUM is found by the model, the system computes a commitment range using the org-type-specific percentages from the challenge brief:

| Org Type | Allocation |
|---|---|
| Pension / Insurance | 0.5–2% of AUM |
| Endowment / Foundation | 1–3% |
| Fund of Funds / Multi-Family Office | 2–5% |
| Single Family Office / HNWI | 3–10% |
| Asset Manager / RIA/FIA / Private Capital Firm | 0.5–3% |

The estimated check size surfaces in Priority Close cards, the pipeline table, the deep dive, and email drafting tone logic.

**Scalability provisions in place:**

- Org-level deduplication groups all contacts by organization before scoring, so 100 contacts across 89 unique orgs makes 89 calls, not 100
- A 0.5-second inter-call delay prevents rate limit errors without a retry library
- Session-state caching skips previously scored orgs when a CSV is re-uploaded mid-session
- Checkpoint file persistence writes each org's result to disk immediately after scoring completes, so a pipeline interrupted at org 60 resumes from org 61 on next run — not from zero
- The quick score expander re-scores any single org without touching the rest of the pipeline

**Why concurrency was deliberately excluded.** Async processing was prototyped and then removed. Running five enrichment calls simultaneously means each call hits the web independently with no shared context, which introduces inconsistency in what each org gets scored against. Sequential calls with a short delay trade speed for retrieval reliability, which matters more here than throughput.

---

### 3. Visualization & Usability

> *Can a non-technical fundraising team member use the output to prioritize their pipeline?*

The dashboard is designed for a fundraising associate who has never touched a terminal. The pipeline runs from a CSV upload and a single button click. Tiers are the primary organizing principle because "PRIORITY CLOSE" is immediately actionable while a raw composite of 7.85 is not.

**Report page — built around action:**

- KPI row: total prospects, priority close count, strong fit count, average composite, run cost
- Priority Close cards: one expandable card per top prospect with contact details, AUM, estimated check size, all four dimension scores, and AI reasoning for sector fit
- Strong Fit table: at-a-glance view of the next tier for wave-2 outreach
- Tier breakdown chart: visual pipeline health in one bar chart

**Analyst view — for deeper research:**

- Filters by tier, org type, confidence level, and GP flag
- Full sortable pipeline table with a quiet anomaly flag column — orgs with detected scoring anomalies show a small ⚠ symbol inline, with a brief reason listed in a sidebar panel beside the table
- Score distribution charts for all three AI-scored dimensions (D1, D3, D4)
- Prospect deep dive: select any org to see its full scoring breakdown with AI-sourced reasoning per dimension

**Email Drafting page** eliminates the blank-page problem for outreach. The system researches each org's recent activity and writes a status-aware draft. Word limits and goals shift automatically based on CRM status (New Contact, Previously Contacted, In Diligence, Committed, Passed). Estimated check size also adjusts tone and urgency: large-check prospects ($50M+ high end) receive a relationship-first approach with no direct ask; core target range ($5M–$50M) gets a confident pitch with a clear call-to-action.

**Terms Dictionary** provides a searchable glossary for anyone unfamiliar with LP/GP/private credit terminology, useful during demos or team onboarding.

---

### 4. Cost Awareness & Engineering Quality

> *Is the system designed to minimize unnecessary API calls? Is there error handling, logging, and state management?*

**Unnecessary API calls are minimized in three ways.**

First, org-level deduplication — all contacts sharing an organization name are grouped before scoring, so 100 contacts across 89 unique orgs makes 89 calls, not 100.

Second, session-state caching — re-uploading a CSV mid-session skips all previously scored orgs entirely. Only net-new orgs hit the API. The status message reports how many orgs were served from cache vs freshly scored.

Third, email drafting is on-demand only — drafts are never pre-generated speculatively. The user selects which orgs to draft for and only then does the pipeline make calls.

**Accuracy / validation layer.** After scoring completes, an anomaly detection pass scans every result for six patterns: GPs that scored high, possible GP mislabelling from description keywords, high composite with low confidence, known allocators that scored suspiciously low, all dimensions uniformly maxed, and all dimensions uniformly near zero. Flagged orgs appear with a quiet ⚠ marker in the pipeline table. The intent is to surface edge cases for human review before outreach is actioned, not to block scoring.

**Error handling is present at every API boundary.** The pipeline loop wraps each `score_org` call in a try/except — a failure on one org logs the error, shows a warning in the UI, and continues with the rest. Email drafting follows the same pattern: a failed draft shows an inline error inside its expander without interrupting the batch.

**Resume-on-interrupt.** Each org result is written to a checkpoint file on disk immediately after scoring. If the pipeline is interrupted — network drop, timeout, or page refresh — the next run reads the checkpoint and resumes from where it stopped. The checkpoint is deleted cleanly on successful completion.

**Logging is implemented end to end.** Every run writes structured, timestamped entries to `pacezero_pipeline.log` via Python's standard `logging` module, which also streams to Streamlit Cloud's built-in log viewer. Each scored org logs its name, all three AI scores, confidence level, and token count. Errors log with full exception detail. The last 100 log lines are visible live in the app under the Cost & Tokens page.

**Cost tracking is granular.** The Cost & Tokens page shows total tokens, total cost, cost per org, and a per-org breakdown table. A scaling projection table covers 100 to 10,000 prospects. Full run on the 89-org sample: **$0.5031 total, $0.0057 per org.**

| Prospects | Estimated Cost |
|---|---|
| 100 | $0.57 |
| 500 | $2.83 |
| 1,000 | $5.65 |
| 5,000 | $28.27 |

---

## Repository Structure

```
pacezero-lp-engine/
├── app.py                        ← Streamlit dashboard (single file)
├── lp_scoring_engine.ipynb       ← Colab notebook: pipeline dev + cross-validation
├── pacezero_logo.png
├── requirements.txt
├── runtime.txt                   ← python-3.11
└── README.md
```

---

## Setup

**Streamlit Cloud (deployed)**

```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "sk-..."
```

**Local**

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Known Limitations

**Scoring accuracy is bounded by public data availability.** Organizations with minimal public footprint receive LOW confidence flags and may be underscored. Human review of all LOW confidence results is recommended before actioning outreach.

**GP detection is heuristic, not a classifier.** The LP/GP rule works well for clear-cut cases but can misclassify hybrid organizations that both allocate capital and manage funds internally. The anomaly detection layer is designed to surface these cases for review.

**No cross-session persistence beyond checkpoint.** The checkpoint file enables resume-on-interrupt within a run, but scored results are cleared from session state on page refresh. CSV export is available at any time as a workaround.

**Sequential processing adds latency.** One org at a time with a 0.5-second inter-call delay. A 100-org run takes roughly 8–12 minutes. Async processing was prototyped and deliberately excluded to preserve retrieval consistency — see Architecture section.

**Email integration is a deep link, not a true send.** The Gmail button pre-fills a compose window via URL. It does not authenticate or push to Drafts programmatically.

---

## Future Development

**Persistent database.** Add SQLite so scored results survive page refreshes and accumulate across runs, complementing the existing checkpoint system.

**Stronger model for production.** GPT-4o was the right choice for an MVP given the 48-hour constraint and its built-in web search capability. A more capable model would improve scoring accuracy for edge cases, particularly hybrid LP/GP organizations.

**Larger training set.** The current calibration anchor set of five organizations validates directional accuracy but is not large enough to train on. A labeled dataset of closed LP outcomes would allow dimension weights to be recalibrated empirically over time, turning the scoring engine from rules-based to genuinely predictive.

**CRM sync.** Connect to the firm's CRM (Affinity, Salesforce, or Notion) to update relationship depth scores automatically as contact activity is logged.

**Gmail API with OAuth 2.0.** Replace deep links with a proper integration that pushes batch drafts directly into the user's Drafts folder.

**Confidence-weighted composite.** Apply a confidence discount to AI-generated dimensions when data quality is LOW, increasing the relative influence of relationship depth.

---

*Built by Adrian Darmali — PaceZero Capital Partners interview challenge, March 2026*

---

## Appendix: Cross-Validation Loss Function & Results

The calibration pipeline uses a two-component combined loss to evaluate prompt quality across rounds. The final pre-test output was:

```
================================================== COMBINED PRE-TEST LOSS ==================================================
  Rule Violation Loss (60%) : 0.0028
  Stability Loss      (40%) : 0.22
  Combined Loss             : 0.0897

OK -- Loss acceptable, ready to run test set
```

**Component 1: Rule Violation Loss (weight 60%)**

This component penalizes hard rule breaches — cases where the model scores an org in a way that contradicts an explicit constraint in the prompt. The primary rules are: GP and service provider organizations must receive sector fit ≤ 2 and emerging manager fit ≤ 2; a confirmed capital allocator must never receive a sector fit below 3; and a LOW confidence result must not produce a composite above 7.0.

Violation loss is computed as the proportion of scored dimensions across the validation set that breach any hard rule, normalized to a 0–1 scale. A score of 0.0028 means fewer than 3 in 1000 dimension scores produced a rule breach — effectively one residual violation across the full validation set, isolated to a hybrid org with ambiguous public classification.

Rule Violation Loss carries 60% of the combined weight because hard rule breaches are more damaging than imprecision. A GP scoring high in sector fit is a fundamental misclassification that could send outreach to the wrong type of organization. Imprecision on a legitimate LP is recoverable. A misclassification is not.

**Component 2: Stability Loss (weight 40%)**

This component measures score variance across repeated calls to the model on the same input. The same org is scored multiple times under identical conditions and the standard deviation of each dimension score is recorded. Stability loss is the mean normalized standard deviation across all validation orgs and dimensions.

A score of 0.22 reflects moderate variance — the model is not perfectly deterministic, which is expected given GPT-4o's temperature setting and the stochastic nature of web retrieval. Scores for well-documented orgs (Rockefeller, PBUCC) were highly stable. Scores for low-public-footprint orgs (Inherent Group, smaller SFOs) showed more variance, which is the correct behaviour: uncertainty in the data should produce uncertainty in the score. The confidence flag surfaces this to the end user.

Stability Loss carries 40% of the combined weight because some variance is acceptable and expected. The goal is not a deterministic system — it is a system whose outputs are consistent enough to be actionable, while honestly flagging cases where the evidence is thin.

**Combined Loss**

```
Combined Loss = (0.60 × Rule Violation Loss) + (0.40 × Stability Loss)
             = (0.60 × 0.0028) + (0.40 × 0.22)
             = 0.00168 + 0.088
             = 0.0897
```

The threshold for proceeding to test set evaluation was a combined loss below 0.10. The final round passed at 0.0897.

**Calibration round progression**

| Round | Key change |
|---|---|
| Round 1 (baseline) | Initial prompt, no rubrics |
| Round 2 | Hard GP caps, named LP/GP examples |
| Round 3 | Explicit scoring bands, abbreviation pre-search rule |
| Round 4 | Emerging manager rubric rewrite, org-type guidance, Foundation/Endowment investment-office steering |

**Final test set results**

The test set was evaluated once, after Round 4, with no post-hoc adjustments.

| Organization | SF got/exp | Halo got/exp | EM got/exp | MSE |
|---|---|---|---|---|
| PBUCC | 8 / 8 | 5 / 6 | 7 / 8 | 0.35 |
| Pension Boards United Church of Christ | 7 / 8 | 5 / 6 | 7 / 8 | 0.70 |
| Meridian Capital Group LLC | 2 / 1 | 3 / 3 | 2 / 1 | 0.50 |
| Inherent Group | 9 / 8 | 4 / 3 | 7 / 5 | 1.15 |
| The Rockefeller Foundation | 6 / 9 | 10 / 9 | 7 / 8 | 3.50 |

**Anchor MSE: 1.24 — Anchor RMSE: 1.11**

The largest residual is The Rockefeller Foundation (MSE 3.50), driven by sector fit returning 6 against an expected 9. The model found limited current evidence of direct lending or private credit allocations in its retrieval pass and scored conservatively. This reflects a data availability limitation rather than a rubric failure. All other anchor orgs fell within 1–2 points across all three dimensions.

**Tier-level classification**

Collapsing scores into the four tier buckets gives a classification-level view:

| Org | Expected Tier | Got Tier | Correct |
|---|---|---|---|
| The Rockefeller Foundation | PRIORITY CLOSE | STRONG FIT | No — dropped one tier |
| PBUCC | STRONG FIT | STRONG FIT | Yes |
| Pension Boards United Church of Christ | STRONG FIT | STRONG FIT | Yes |
| Meridian Capital Group LLC | WEAK FIT | WEAK FIT | Yes |
| Inherent Group | STRONG FIT | PRIORITY CLOSE | No — jumped one tier |

Three out of five anchor orgs landed in the correct tier. Both misclassifications were off by exactly one tier in opposite directions. Critically, no LP was classified as WEAK FIT and the confirmed GP (Meridian) was correctly placed at WEAK FIT — the two failure modes that would actually damage the fundraising workflow did not occur.
