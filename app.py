import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import re
from openai import OpenAI

# ── Page config ──
st.set_page_config(
    page_title="PaceZero LP Scoring Engine",
    page_icon="🏦",
    layout="wide"
)

# ── API setup ──
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
MODEL  = "gpt-4o"

# ── Scoring constants ──
WEIGHTS = {
    'sector_fit':         0.35,
    'relationship_depth': 0.30,
    'halo_value':         0.20,
    'emerging_fit':       0.15,
}

TIERS = [
    (8.0, 'PRIORITY CLOSE'),
    (6.5, 'STRONG FIT'),
    (5.0, 'MODERATE FIT'),
    (0.0, 'WEAK FIT'),
]

TIER_COLORS = {
    'PRIORITY CLOSE': '🟢',
    'STRONG FIT':     '🔵',
    'MODERATE FIT':   '🟡',
    'WEAK FIT':       '🔴',
}

TEST_ORGS = [
    "PBUCC",
    "Pension Boards United Church of Christ",
    "Meridian Capital Group LLC",
    "Inherent Group",
    "The Rockefeller Foundation",
]

SCORING_PROMPT = """
You are an LP analyst for PaceZero Capital Partners — a Toronto-based
sustainability-focused private credit firm (Fund II, emerging manager).

Research the organization below using web search, then score it across
3 dimensions using the rubrics provided.

ORGANIZATION:
  Name   : {org_name}
  Type   : {org_type}
  Contact: {role}

CRITICAL RULE — LP vs GP DISTINCTION:
  LP = allocates capital INTO externally managed funds
  GP = PRIMARILY manages funds for others / originates loans / brokers deals
  Only mark as GP if PURELY a service provider with no external fund allocations.

  Examples of LPs (do NOT flag as GP):
    - Neuberger Berman, Lincoln Financial, Bessemer Trust, BBH
    - Ludwig Institute for Cancer Research, Safra Group

  Examples of true GPs:
    - Meridian Capital Group (pure CRE brokerage)
    - Gratitude Railroad (impact asset manager)
    - Placement agents, pure loan originators

  GP hard caps: sector_fit <= 2, emerging_fit <= 2, halo <= 3

RUBRIC 1 — SECTOR & MANDATE FIT (1-10):
  9-10 : Private credit allocation + sustainability/ESG mandate (BOTH confirmed)
  7-8  : One confirmed strongly, other strongly implied
  5-6  : LP with alternatives exposure but weak ESG, OR strong ESG but no credit signal
  3-4  : LP confirmed but no alignment signals found
  1-2  : GP / broker / lender / service provider

RUBRIC 2 — HALO & STRATEGIC VALUE (1-10):
  9-10 : Globally recognized institution
  7-8  : Well-known in impact/sustainability LP circles
  5-6  : Respected regionally or within a niche
  3-4  : Limited public presence
  1-2  : Unknown or reputation-neutral
  SFOs: max 7 unless globally famous billionaire family with $1B+ AUM

RUBRIC 3 — EMERGING MANAGER FIT (1-10):
  9-10 : Documented emerging manager programme; backed Fund I/II before
  7-8  : Flexible smaller institution; SFOs and small MFOs with open mandate
  5-6  : No explicit programme but org type typically flexible
  3-4  : Large institution; likely prefers established managers
  1-2  : Known policy against emerging managers OR confirmed GP

  Do NOT default to 4-5. SFOs and small MFOs should score 6-7 unless restricted.
  Foundations and faith-based pensions with impact mandates score 7-8 on EM.

Respond ONLY with valid JSON. No preamble, no markdown, no extra text.

{{
  "org_name": "{org_name}",
  "org_description": "<2 sentence factual summary>",
  "is_lp": <true|false>,
  "is_gp_or_service_provider": <true|false>,
  "gp_evidence": "<why flagged as GP or null if LP>",
  "aum_estimate": "<e.g. $2.5B or Unknown>",
  "aum_figure_millions": <number or null>,
  "sector_fit_score": <1-10>,
  "sector_fit_reasoning": "<2 sentences citing evidence>",
  "halo_score": <1-10>,
  "halo_reasoning": "<2 sentences citing evidence>",
  "emerging_fit_score": <1-10>,
  "emerging_fit_reasoning": "<2 sentences citing evidence>",
  "confidence": "<HIGH|MEDIUM|LOW>",
  "data_quality_notes": "<what was or wasnt available>"
}}
"""

def compute_composite(sf, rel, halo, em):
    return round(
        sf   * WEIGHTS['sector_fit']         +
        rel  * WEIGHTS['relationship_depth']  +
        halo * WEIGHTS['halo_value']          +
        em   * WEIGHTS['emerging_fit'],
        2
    )

def classify_tier(composite):
    for threshold, label in TIERS:
        if composite >= threshold:
            return label
    return 'WEAK FIT'

def score_org(org_name, org_type, role):
    prompt = SCORING_PROMPT.format(
        org_name=org_name,
        org_type=org_type,
        role=role,
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        max_tokens=1000,
    )
    raw_text      = response.choices[0].message.content.strip()
    input_tokens  = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    data = json.loads(raw_text)
    data['_input_tokens']  = input_tokens
    data['_output_tokens'] = output_tokens
    data['_tokens']        = input_tokens + output_tokens
    return data


# ════════════════════════════════════════
# SIDEBAR — Role selector
# ════════════════════════════════════════
st.sidebar.image("https://via.placeholder.com/200x60?text=PaceZero", width=200)
st.sidebar.markdown("---")
role_view = st.sidebar.radio(
    "Dashboard View",
    ["Executive", "Analyst"],
    index=0
)
st.sidebar.markdown("---")
st.sidebar.markdown("**PaceZero Capital Partners**")
st.sidebar.markdown("LP Prospect Scoring Engine v1.0")


# ════════════════════════════════════════
# FILE UPLOAD
# ════════════════════════════════════════
st.title("🏦 LP Prospect Scoring Engine")
st.markdown("Upload your prospect CSV to enrich and score your LP pipeline.")

uploaded_file = st.file_uploader("Upload challenge_contacts.csv", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['Contact Name', 'Organization'])
    df['Organization']      = df['Organization'].str.strip()
    df['Relationship Depth'] = pd.to_numeric(df['Relationship Depth'], errors='coerce').fillna(5)

    # Split train/test
    test_mask = df['Organization'].isin(TEST_ORGS)
    train_df  = df[~test_mask].copy()

    st.success(f"Loaded {len(df)} contacts — {train_df['Organization'].nunique()} unique orgs to score")

    if st.button("🚀 Run Scoring Pipeline", type="primary"):

        results     = []
        scored_rows = []

        unique_orgs = train_df.groupby('Organization').first().reset_index()
        progress    = st.progress(0)
        status      = st.empty()
        total       = len(unique_orgs)

        total_input_tokens  = 0
        total_output_tokens = 0

        for i, (_, row) in enumerate(unique_orgs.iterrows()):
            org_name = row['Organization']
            status.text(f"Scoring {i+1}/{total}: {org_name}")
            try:
                data = score_org(org_name, row['Org Type'], row['Role'])
                data['org_name'] = org_name
                data['org_type'] = row['Org Type']
                results.append(data)
                total_input_tokens  += data.get('_input_tokens', 0)
                total_output_tokens += data.get('_output_tokens', 0)
            except Exception as e:
                st.warning(f"Failed: {org_name} — {e}")
            progress.progress((i + 1) / total)
            time.sleep(0.5)

        # Build scored_df
        for _, contact in train_df.iterrows():
            org_name = contact['Organization']
            org_data = next((r for r in results if r['org_name'] == org_name), None)
            if not org_data:
                continue
            sf  = org_data.get('sector_fit_score', 5)
            ha  = org_data.get('halo_score', 5)
            em  = org_data.get('emerging_fit_score', 5)
            rel = float(contact['Relationship Depth'])
            composite = compute_composite(sf, rel, ha, em)
            tier      = classify_tier(composite)
            scored_rows.append({
                'Contact Name':    contact['Contact Name'],
                'Organization':    org_name,
                'Type':            contact['Org Type'],
                'Region':          contact['Region'],
                'Status':          contact['Contact Status'],
                'Sector Fit':      sf,
                'Rel Depth':       rel,
                'Halo':            ha,
                'Emerging Fit':    em,
                'Composite':       composite,
                'Tier':            tier,
                'AUM':             org_data.get('aum_estimate', 'Unknown'),
                'Confidence':      org_data.get('confidence', 'LOW'),
                'Is GP':           org_data.get('is_gp_or_service_provider', False),
                'Why':             org_data.get('sector_fit_reasoning', ''),
                'Halo Reasoning':  org_data.get('halo_reasoning', ''),
                'EM Reasoning':    org_data.get('emerging_fit_reasoning', ''),
                'Org Summary':     org_data.get('org_description', ''),
            })

        scored_df = pd.DataFrame(scored_rows).sort_values('Composite', ascending=False).reset_index(drop=True)

        # Cost
        cost = ((total_input_tokens / 1000 * 0.0025) + (total_output_tokens / 1000 * 0.0100))
        status.text("Pipeline complete")
        progress.progress(1.0)

        st.session_state['scored_df'] = scored_df
        st.session_state['results']   = results
        st.session_state['cost']      = cost
        st.session_state['tokens']    = total_input_tokens + total_output_tokens


# ════════════════════════════════════════
# DASHBOARD — show after scoring
# ════════════════════════════════════════
if 'scored_df' in st.session_state:
    scored_df = st.session_state['scored_df']
    cost      = st.session_state['cost']
    tokens    = st.session_state['tokens']

    tier_counts = scored_df['Tier'].value_counts()

    # ── EXECUTIVE VIEW ──
    if role_view == "Executive":
        st.markdown("## Executive Pipeline Summary")

        # KPI cards
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Total Prospects", len(scored_df))
        col2.metric("Priority Close",  tier_counts.get('PRIORITY CLOSE', 0))
        col3.metric("Strong Fit",       tier_counts.get('STRONG FIT', 0))
        col4.metric("Avg Composite",    round(scored_df['Composite'].mean(), 2))
        col5.metric("Run Cost",         f"${cost:.4f}")

        st.markdown("---")

        # Priority Close prospects only
        st.markdown("### 🎯 Priority Close — Action Required This Week")
        priority = scored_df[scored_df['Tier'] == 'PRIORITY CLOSE']
        if len(priority):
            for _, row in priority.iterrows():
                with st.expander(f"{row['Organization']}  |  Score: {row['Composite']}  |  {row['AUM']}"):
                    c1, c2 = st.columns(2)
                    c1.markdown(f"**Contact:** {row['Contact Name']}")
                    c1.markdown(f"**Type:** {row['Type']}")
                    c1.markdown(f"**Status:** {row['Status']}")
                    c1.markdown(f"**Confidence:** {row['Confidence']}")
                    c2.markdown(f"**Sector Fit:** {row['Sector Fit']} / 10")
                    c2.markdown(f"**Halo:** {row['Halo']} / 10")
                    c2.markdown(f"**Emerging Fit:** {row['Emerging Fit']} / 10")
                    st.markdown(f"**Why:** {row['Why']}")
        else:
            st.info("No Priority Close prospects yet.")

        st.markdown("---")

        # Strong Fit summary
        st.markdown("### 💪 Strong Fit Pipeline")
        strong = scored_df[scored_df['Tier'] == 'STRONG FIT'][
            ['Contact Name', 'Organization', 'Type', 'Composite', 'AUM', 'Confidence']
        ]
        st.dataframe(strong, use_container_width=True, hide_index=True)

        # Tier breakdown chart
        st.markdown("---")
        st.markdown("### Pipeline Breakdown")
        tier_df = pd.DataFrame({
            'Tier':  list(tier_counts.index),
            'Count': list(tier_counts.values)
        })
        st.bar_chart(tier_df.set_index('Tier'))


    # ── ANALYST VIEW ──
    elif role_view == "Analyst":
        st.markdown("## Analyst Pipeline Dashboard")

        # Filters
        st.markdown("### Filters")
        fc1, fc2, fc3, fc4 = st.columns(4)
        tier_filter = fc1.multiselect(
            "Tier",
            options=['PRIORITY CLOSE', 'STRONG FIT', 'MODERATE FIT', 'WEAK FIT'],
            default=['PRIORITY CLOSE', 'STRONG FIT', 'MODERATE FIT', 'WEAK FIT']
        )
        type_filter = fc2.multiselect(
            "Org Type",
            options=sorted(scored_df['Type'].unique()),
            default=list(scored_df['Type'].unique())
        )
        conf_filter = fc3.multiselect(
            "Confidence",
            options=['HIGH', 'MEDIUM', 'LOW'],
            default=['HIGH', 'MEDIUM', 'LOW']
        )
        hide_gp = fc4.checkbox("Hide GPs", value=True)

        # Apply filters
        filtered_df = scored_df[
            scored_df['Tier'].isin(tier_filter) &
            scored_df['Type'].isin(type_filter) &
            scored_df['Confidence'].isin(conf_filter)
        ]
        if hide_gp:
            filtered_df = filtered_df[filtered_df['Is GP'] == False]

        st.markdown(f"Showing **{len(filtered_df)}** prospects")
        st.markdown("---")

        # KPI row
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total",         len(filtered_df))
        c2.metric("Priority Close", len(filtered_df[filtered_df['Tier'] == 'PRIORITY CLOSE']))
        c3.metric("Strong Fit",     len(filtered_df[filtered_df['Tier'] == 'STRONG FIT']))
        c4.metric("Avg Composite",  round(filtered_df['Composite'].mean(), 2))
        c5.metric("Total Tokens",   f"{tokens:,}")
        c6.metric("Run Cost",       f"${cost:.4f}")

        st.markdown("---")

        # Full pipeline table
        st.markdown("### Full Pipeline")
        display_cols = [
            'Contact Name', 'Organization', 'Type', 'Region',
            'Sector Fit', 'Rel Depth', 'Halo', 'Emerging Fit',
            'Composite', 'Tier', 'AUM', 'Confidence'
        ]
        st.dataframe(
            filtered_df[display_cols],
            use_container_width=True,
            hide_index=True
        )
       

        st.markdown("---")

        # Score distribution
        st.markdown("### Score Distribution")
        dc1, dc2, dc3 = st.columns(3)
        dc1.bar_chart(pd.DataFrame({'Sector Fit': filtered_df['Sector Fit'].value_counts().sort_index()}))
        dc2.bar_chart(pd.DataFrame({'Halo': filtered_df['Halo'].value_counts().sort_index()}))
        dc3.bar_chart(pd.DataFrame({'Emerging Fit': filtered_df['Emerging Fit'].value_counts().sort_index()}))

        st.markdown("---")

        # Prospect deep dive
        st.markdown("### Prospect Deep Dive")
        selected_org = st.selectbox(
            "Select an organization",
            options=filtered_df['Organization'].tolist()
        )
        if selected_org:
            row = filtered_df[filtered_df['Organization'] == selected_org].iloc[0]
            d1, d2 = st.columns(2)
            with d1:
                st.markdown(f"**Contact:** {row['Contact Name']}")
                st.markdown(f"**Type:** {row['Type']}")
                st.markdown(f"**Region:** {row['Region']}")
                st.markdown(f"**Status:** {row['Status']}")
                st.markdown(f"**AUM:** {row['AUM']}")
                st.markdown(f"**Confidence:** {row['Confidence']}")
                st.markdown(f"**GP Flag:** {'⚠️ Yes' if row['Is GP'] else 'No'}")
            with d2:
                st.markdown(f"**Composite Score:** {row['Composite']} / 10")
                st.markdown(f"**Tier:** {TIER_COLORS.get(row['Tier'], '')} {row['Tier']}")
                st.markdown(f"**Sector Fit (D1):** {row['Sector Fit']} / 10")
                st.markdown(f"**Rel Depth (D2):** {row['Rel Depth']} / 10")
                st.markdown(f"**Halo (D3):** {row['Halo']} / 10")
                st.markdown(f"**Emerging Fit (D4):** {row['Emerging Fit']} / 10")
            st.markdown("**Org Summary:**")
            st.info(row['Org Summary'])
            st.markdown("**Sector Fit Reasoning:**")
            st.write(row['Why'])
            st.markdown("**Halo Reasoning:**")
            st.write(row['Halo Reasoning'])
            st.markdown("**Emerging Fit Reasoning:**")
            st.write(row['EM Reasoning'])

        st.markdown("---")

        # Cost tracker
        st.markdown("### API Cost Tracker")
        cost_per_org = cost / max(len(st.session_state['results']), 1)
        cost_df = pd.DataFrame({
            'Prospects':  [100, 500, 1000, 5000],
            'Est. Cost':  [f"${cost_per_org * n:.2f}" for n in [100, 500, 1000, 5000]]
        })
        st.table(cost_df)

        # Download button
        st.markdown("---")
        st.download_button(
            label="Download Scored Results as CSV",
            data=scored_df.to_csv(index=False),
            file_name="pacezero_scored_pipeline.csv",
            mime="text/csv"
        )
