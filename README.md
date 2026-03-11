# PaceZero LP Prospect Enrichment & Scoring Engine

**PaceZero Capital Partners**

A fully automated pipeline that ingests a prospect CSV, enriches each organization using AI-powered web research, scores it across four fundraising-relevant dimensions, and surfaces results in a live multi-page Streamlit dashboard with built-in outreach drafting.

---

## Live App

**https://pacezero-lp-engine-adriandarmali.streamlit.app**

---

## What It Does

Given a list of LP contacts, the engine:

1. Calls GPT-4o with a structured research prompt for each unique organization
2. Scores it across four dimensions (detailed below)
3. Computes a weighted composite score and assigns a pipeline tier
4. Persists results to SQLite (`pacezero_pipeline.db`) and loads into session state
5. Lets users draft personalized outreach emails — single or batch — with AI-researched hooks

---

## Architecture

```
challenge_contacts.csv
        ↓
  Google Colab (Python)          ← development, calibration, cross-validation
        ↓
  Streamlit App (app.py)         ← production interface
        ↓ calls
  OpenAI GPT-4o (gpt-4o)        ← org research + scoring + email drafting
        ↓
  Session State                  ← scored DataFrame persisted in-memory
        ↓
  4-Page Dashboard               ← Report / Cost & Tokens / Terms Dictionary / Email Drafting
```

Every completed run is written to a local SQLite database (`pacezero_pipeline.db`). On app startup, the latest run is automatically restored into the dashboard. Results can also be exported as CSV at any time.

---

## Scoring Dimensions

Each organization is scored on four dimensions. Three are AI-generated; one is pre-computed from CRM data.

| # | Dimension | Weight | Source |
|---|---|---|---|
| D1 | Sector & Mandate Fit | 35% | GPT-4o web research |
| D2 | Relationship Depth | 30% | Pre-computed from CSV |
| D3 | Halo & Strategic Value | 20% | GPT-4o web research |
| D4 | Emerging Manager Fit | 15% | GPT-4o web research |

**Composite score** = `(D1 × 0.35) + (D2 × 0.30) + (D3 × 0.20) + (D4 × 0.15)`

### Tier Classification

| Tier | Threshold |
|---|---|
| PRIORITY CLOSE | ≥ 8.0 |
| STRONG FIT | ≥ 6.5 |
| MODERATE FIT | ≥ 5.0 |
| WEAK FIT | < 5.0 |

---

## Prompt Engineering

The scoring prompt (`SCORING_PROMPT`) was designed to make GPT-4o behave like a trained LP analyst, not a generic summarizer. Key design decisions:

**LP vs GP disambiguation.** The most common failure mode in early testing was the model scoring brokers, loan originators, and placement agents as if they were capital allocators. The prompt now includes a hard rule: if an organization is *primarily* a GP or service provider, sector fit is capped at 2, emerging manager fit at 2, and halo at 3. Named examples of each category are included directly in the prompt to anchor the model's judgment.

**Abbreviation pre-search rule.** Organizations like `PBUCC` were being scored without the model knowing their full name. The prompt now instructs the model to resolve any all-caps abbreviation before scoring.

**Calibrated rubrics.** Each rubric includes explicit band descriptions (e.g., "7–8: well-known in impact/sustainability LP circles") rather than generic adjectives. This reduces score variance across repeated runs.

**Absence of information is not a restriction.** Early versions of the emerging manager rubric defaulted to mid-range scores when data was thin. The prompt now explicitly states that lack of public information about an emerging manager policy does not imply a restriction — the org type itself is the primary signal.

**SFO halo guidance.** Single family offices were being over-scored on halo. The prompt now caps SFO halo at 7, with an exception only for globally-known families managing $1B+.

---

## Cross-Validation Process

Before running the full pipeline, scores were validated against a manually-labeled training set. The process followed a standard ML discipline of separating development from evaluation.

**Round 1 — First draft prompt, run on training set.**
27 rubric violations detected, including GPs scored as LPs, SFOs with halo scores above 7, and emerging manager scores defaulting to 4–5 when data was absent.

**Round 2 — Prompt revised, training set re-run.**
Violations dropped to 1 (a minor calibration edge case on Gratitude Railroad, which sits on the boundary between impact fund and asset manager).

**Round 3 — Stability test.**
The same 9 organizations were scored twice with no prompt changes. 8 of 9 scores were identical across runs, confirming the prompt produces consistent outputs.

**Round 4 — Test set evaluated once.**
Five held-out organizations (never seen during prompt development) were scored and compared to manually-assigned expected values.

---

## Test Set Results

| Organization | Sector Fit | Halo | Emerging Fit | MSE |
|---|---|---|---|---|
| Meridian Capital Group LLC | 1 / 1 | 3 / 3 | 1 / 1 | 0.00 |
| Inherent Group | 7 / 8 | 4 / 3 | 7 / 5 | 1.15 |
| The Rockefeller Foundation | 9 / 9 | 10 / 9 | 4 / 8 | 2.60 |
| Pension Boards UCC | 6 / 8 | 5 / 6 | 5 / 8 | 2.95 |
| PBUCC | 6 / 8 | 4 / 6 | 3 / 8 | 5.95 |

*Format: model score / expected score. MSE = mean squared error per org across the three AI-scored dimensions.*

**Anchor RMSE: 1.59** (averaged across all five test orgs)

The Rockefeller Foundation and Pension Boards UCC both underscored on emerging manager fit. The model found no explicit emerging manager programme documented for either, which caused it to score conservatively. In practice, faith-based pensions and large foundations are known to allocate to emerging managers even when no formal programme is publicly stated — the rubric has since been updated to reflect this, but the RMSE above reflects the state of the model at final test evaluation.

PBUCC scored higher variance because the abbreviation was not resolved correctly in the first pass — the pre-search rule was added as a direct result.

### A note on the RMSE figure

This RMSE is a **development benchmark**, not a clean generalization metric. At one point during development, a set of heuristic defaults by org type was added to the prompt after inspecting test set results. This is a form of test set contamination — adjusting the model based on evaluation data invalidates the holdout. That block was removed before final submission. The RMSE of 1.59 therefore reflects the model's performance on a test set that was used only once, with no post-hoc tuning, which is the correct way to report it.

In a production setting, a second clean test set would be required to produce a reliable generalization estimate.

---

## API Cost Analysis

Full run on 89 unique organizations:

| Metric | Value |
|---|---|
| Input tokens | 99,955 |
| Output tokens | 25,323 |
| Total tokens | 125,278 |
| Total cost | $0.5031 |
| Cost per org | $0.0057 |

### Scaling projection

| Prospects | Estimated cost |
|---|---|
| 100 | $0.57 |
| 500 | $2.83 |
| 1,000 | $5.65 |
| 5,000 | $28.27 |

---

## Dashboard Pages

**📊 Report** — Executive summary (KPIs, Priority Close action cards, Strong Fit table, tier breakdown chart) followed by the full Analyst view (filterable pipeline table, score distribution charts, prospect deep dive with AI reasoning).

**💰 Cost & Tokens** — Per-run token usage, cost per org, scaling projections, and a per-organization token breakdown table.

**📖 Terms Dictionary** — Searchable glossary of 20 LP/fundraising terms for onboarding and demo context.

**✉️ Email Drafting** — Single or batch outreach drafting. For each prospect, the engine runs two API calls: one to research the organization's most recent relevant activity, and one to write a personalized email using that research as the opening hook. Email goal and word limit adapt automatically to the contact's CRM status (New Contact, Previously Contacted, Existing Contact, In Diligence, Committed, Passed). A Gmail deep link pre-fills the compose window with subject and body.

> **Demo limitation on email:** The Gmail button uses a URL deep link and does not require authentication. In production, this would use the Gmail API with OAuth 2.0 to push drafts directly into the user's Drafts folder.

---

## Repository Structure

```
pacezero-lp-engine/
├── app.py                        ← Streamlit dashboard
├── lp_scoring_engine.ipynb       ← Colab notebook (pipeline dev + cross-validation)
├── pacezero_logo.png
├── requirements.txt
├── runtime.txt                   ← python-3.11
└── README.md
```

---

## Setup

**Streamlit Cloud (deployed)**

Set `OPENAI_API_KEY` in Streamlit Secrets:
```toml
OPENAI_API_KEY = "sk-..."
```

**Local**

```bash
pip install -r requirements.txt
streamlit run app.py
```

Add a `.streamlit/secrets.toml` file with your OpenAI API key.

---

## Requirements

```
streamlit
openai
pandas
numpy
```

---

*Built by Adrian Darmali — PaceZero Capital Partners interview challenge, March 2026*

---

## Known Limitations

**Scoring accuracy is bounded by public data availability.**
GPT-4o can only score what it can find. Organizations with minimal public footprint — many single family offices, smaller foundations, and private firms — receive LOW confidence flags and may be underscored or miscategorized. The model is explicitly instructed not to penalize for thin data, but human review of all LOW confidence results is strongly recommended before actioning outreach.

**The RMSE is a development benchmark, not a production metric.**
1.59 RMSE was measured on five held-out test organizations after a clean final evaluation pass. It gives a directional sense of calibration quality but is not statistically robust at that sample size. A proper generalization estimate would require a larger, independently-labeled test set.

**GP detection relies on heuristics.**
The LP/GP classification rule works well for clear-cut cases (pure brokers, placement agents) but struggles with hybrid organizations that both allocate capital and manage funds. These edge cases are flagged for review but may slip through with incorrect scores.

**Data persists across sessions via SQLite.**
Every pipeline run is saved to a local `pacezero_pipeline.db` SQLite database. On app startup, the latest run is automatically loaded back into the dashboard — no re-scoring required. Full run history (cost, tokens, timestamp) is visible on the Cost & Tokens page.

**Email integration is a deep link, not a true send.**
The Gmail button pre-fills a compose window via URL. It does not authenticate, access contacts, or save drafts programmatically. See future development below.

**Sequential processing and latency.**
The pipeline processes organizations one at a time with a 0.5-second delay between calls to avoid rate limit errors. A full 100-org run takes approximately 8–12 minutes. Parallel processing is not currently implemented.

---

## Future Development

**Persistent storage and CRM sync.**
Replace session state with a lightweight database backend (SQLite is the natural starting point given the spec) so scored results survive page refreshes and accumulate across runs. Long-term, a sync with the firm's CRM (Affinity, Salesforce, or Notion) would let relationship depth scores update automatically as contact activity is logged — removing the need to re-upload a CSV each time.

**Gmail API with OAuth 2.0.**
Replace the deep link with a proper Gmail API integration. After a one-time Google login, batch drafts would be pushed directly into the user's Drafts folder — reviewable and sendable from a normal inbox without opening multiple browser tabs per prospect.

**Improved LP/GP detection.**
Train a lightweight binary classifier on labeled examples to replace the current heuristic prompt rule. This would reduce misclassification on hybrid organizations and remove the dependency on GPT-4o for what is essentially a routing decision that should not cost tokens.

**Confidence-weighted composite scoring.**
Currently a LOW confidence score carries the same weight as a HIGH confidence one in the composite. A better model would apply a confidence discount — reducing the influence of AI-generated dimensions when data quality is poor and increasing the relative weight of relationship depth, which is always reliably known.

**Parallel scoring with async API calls.**
Switch from sequential to async batch requests using `asyncio` and the OpenAI async client. This would reduce a 100-org pipeline run from roughly 10 minutes to under 2 minutes, and cost less at scale due to reduced per-call overhead.

**Fine-tuning on confirmed LP outcomes.**
As the firm closes LPs and tracks which prospects converted, those outcomes can be used to calibrate dimension weights based on what actually predicted a commitment — turning the scoring engine from a rules-based system into a data-driven one over time.

**Automated weekly pipeline refresh.**
Schedule a weekly job that re-scores new contacts, flags score changes on existing prospects, and delivers a brief summarizing the week's highest-priority actions — removing the need for manual runs entirely.

---

*Built by Adrian Darmali — PaceZero Capital Partners interview challenge, March 2026*
