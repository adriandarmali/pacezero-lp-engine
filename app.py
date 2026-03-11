import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from openai import OpenAI

# ── Page config ──
st.set_page_config(
    page_title="PaceZero LP Scoring Engine",
    page_icon="pacezero_logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@600;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { background-color: #f8f9fa; color: #1a1a2e; }

section[data-testid="stSidebar"] {
    background-color: #1a1a2e;
    border-right: 1px solid #2d2d44;
}
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }

header[data-testid="stHeader"] { background: #f8f9fa; border-bottom: 1px solid #e0e0e0; }

[data-testid="metric-container"] {
    background: #ffffff;
    border: 1px solid #e0e4ec;
    border-radius: 10px;
    padding: 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
[data-testid="metric-container"] label {
    color: #6b7280 !important;
    font-size: 11px !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-family: 'DM Mono', monospace !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #1a1a2e !important;
    font-size: 26px !important;
    font-weight: 700 !important;
    font-family: 'Playfair Display', serif !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid #e0e4ec;
    border-radius: 10px;
    overflow: hidden;
    background: #ffffff;
}

.stButton > button {
    background: #1a1a2e;
    color: #ffffff;
    border: none;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-weight: 500;
    padding: 10px 24px;
    transition: all 0.2s;
}
.stButton > button:hover { background: #2d2d50; transform: translateY(-1px); }

.streamlit-expanderHeader {
    background: #ffffff !important;
    border: 1px solid #e0e4ec !important;
    border-radius: 8px !important;
    color: #1a1a2e !important;
    font-weight: 500 !important;
}
.streamlit-expanderContent {
    background: #ffffff !important;
    border: 1px solid #e0e4ec !important;
    border-top: none !important;
}

.stProgress > div > div { background-color: #1a1a2e; }

hr { border-color: #e0e4ec; margin: 28px 0; }

.stAlert { background: #ffffff; border: 1px solid #e0e4ec; border-radius: 8px; }

.badge-priority { background:#d1fae5; color:#065f46; padding:3px 10px; border-radius:12px; font-size:11px; font-weight:600; font-family:'DM Mono',monospace; letter-spacing:0.05em; }
.badge-strong   { background:#dbeafe; color:#1e40af; padding:3px 10px; border-radius:12px; font-size:11px; font-weight:600; font-family:'DM Mono',monospace; letter-spacing:0.05em; }
.badge-moderate { background:#fef3c7; color:#92400e; padding:3px 10px; border-radius:12px; font-size:11px; font-weight:600; font-family:'DM Mono',monospace; letter-spacing:0.05em; }
.badge-weak     { background:#fee2e2; color:#991b1b; padding:3px 10px; border-radius:12px; font-size:11px; font-weight:600; font-family:'DM Mono',monospace; letter-spacing:0.05em; }
.badge-high     { background:#d1fae5; color:#065f46; padding:2px 8px; border-radius:10px; font-size:10px; font-weight:600; font-family:'DM Mono',monospace; }
.badge-medium   { background:#fef3c7; color:#92400e; padding:2px 8px; border-radius:10px; font-size:10px; font-weight:600; font-family:'DM Mono',monospace; }
.badge-low      { background:#fee2e2; color:#991b1b; padding:2px 8px; border-radius:10px; font-size:10px; font-weight:600; font-family:'DM Mono',monospace; }

.section-header { font-family:'Playfair Display',serif; font-size:24px; font-weight:700; color:#1a1a2e; margin-bottom:4px; }
.section-sub    { font-family:'DM Mono',monospace; font-size:11px; color:#6b7280; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:20px; }

.term-card { background:#ffffff; border:1px solid #e0e4ec; border-left:3px solid #1a1a2e; border-radius:0 8px 8px 0; padding:14px 18px; margin-bottom:10px; box-shadow:0 1px 3px rgba(0,0,0,0.04); }
.term-name { font-family:'DM Mono',monospace; font-size:13px; font-weight:500; color:#1a1a2e; margin-bottom:4px; }
.term-def  { font-size:13px; color:#6b7280; line-height:1.6; }
</style>
""", unsafe_allow_html=True)

# ── API Setup ──
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
MODEL  = "gpt-4o"

WEIGHTS = {'sector_fit':0.35,'relationship_depth':0.30,'halo_value':0.20,'emerging_fit':0.15}
TIERS   = [(8.0,'PRIORITY CLOSE'),(6.5,'STRONG FIT'),(5.0,'MODERATE FIT'),(0.0,'WEAK FIT')]

TEST_ORGS = ["PBUCC","Pension Boards United Church of Christ","Meridian Capital Group LLC","Inherent Group","The Rockefeller Foundation"]
ORG_TYPES = ["Single Family Office","Multi-Family Office","Fund of Funds","Foundation","Endowment","Pension","Insurance","Asset Manager","RIA/FIA","HNWI","Private Capital Firm"]
CONTACT_STATUSES = ["New Contact","Previously Contacted","Existing Contact","In Diligence","Committed","Passed"]

TERMS = {
    "LP (Limited Partner)": "A capital allocator that invests money INTO funds managed by external GPs. Examples include foundations, endowments, pensions, and family offices. LPs are the fundraising targets for PaceZero.",
    "GP (General Partner)": "An entity that manages funds on behalf of LPs. PaceZero is a GP. Brokers, loan originators, and placement agents are also GPs or service providers. GPs are NOT valid LP prospects.",
    "Private Credit": "A form of lending where funds directly loan money to companies, bypassing traditional banks. PaceZero's core strategy — also called direct lending or private debt.",
    "Emerging Manager": "A fund manager raising their first (Fund I) or second (Fund II) fund. PaceZero is an emerging manager. Some LPs have dedicated programmes to back emerging managers.",
    "AUM (Assets Under Management)": "The total value of investments an organization manages. Used to estimate how large a commitment they might make to a single fund.",
    "ESG": "Environmental, Social, and Governance — a framework for sustainable investing. LPs with ESG mandates are more likely to invest in sustainability-focused funds like PaceZero.",
    "Fund of Funds (FoF)": "An investment vehicle that allocates to other funds rather than directly to companies. FoFs are LPs by definition and are key prospects.",
    "Single Family Office (SFO)": "An organization that manages wealth for one ultra-high-net-worth family. SFOs are flexible investors with less bureaucracy than institutions.",
    "Multi-Family Office (MFO)": "Like an SFO but serves multiple wealthy families. Typically allocates to external managers on behalf of clients.",
    "Halo Value": "The strategic signal value of winning a specific LP. A well-known LP on your cap table tells other prospects that someone credible already validated the fund.",
    "Composite Score": "The weighted average of all 4 dimensions: Sector Fit (35%) + Relationship Depth (30%) + Halo Value (20%) + Emerging Manager Fit (15%).",
    "PRIORITY CLOSE": "Composite score >= 8.0. Highest priority prospects — personalize outreach immediately.",
    "STRONG FIT": "Composite score >= 6.5. High priority — research contact angle before outreach.",
    "MODERATE FIT": "Composite score >= 5.0. Include in wave 2 outreach.",
    "WEAK FIT": "Composite score < 5.0. Low priority or confirmed non-LP.",
    "Relationship Depth": "Pre-computed CRM score (1-10) reflecting how warm the existing relationship is. Based on call history, email recency, meeting notes, and deal associations.",
    "Confidence Flag": "HIGH / MEDIUM / LOW — indicates how much public data the AI found. LOW confidence scores should be manually verified before actioning.",
    "ICCR": "Interfaith Center on Corporate Responsibility — a coalition of faith-based institutional investors known for responsible investing mandates.",
    "UNPRI": "United Nations Principles for Responsible Investment — a global network of investors committed to ESG integration. Signatories are strong sustainability mandate signals.",
    "Direct Lending": "A private credit strategy where a fund lends directly to mid-market companies. PaceZero's primary investment approach.",
    "Check Size": "The estimated dollar amount an LP might commit to a single fund, calculated as a percentage of their AUM based on org type.",
}

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

  Examples of LPs: Neuberger Berman, Lincoln Financial, Bessemer Trust, BBH, Ludwig Institute
  Examples of GPs: Meridian Capital Group (CRE brokerage), Gratitude Railroad (asset manager)

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

Respond ONLY with valid JSON:
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
    return round(sf*WEIGHTS['sector_fit'] + rel*WEIGHTS['relationship_depth'] + halo*WEIGHTS['halo_value'] + em*WEIGHTS['emerging_fit'], 2)

def classify_tier(composite):
    for threshold, label in TIERS:
        if composite >= threshold:
            return label
    return 'WEAK FIT'

def tier_badge(tier):
    cls = {'PRIORITY CLOSE':'priority','STRONG FIT':'strong','MODERATE FIT':'moderate','WEAK FIT':'weak'}.get(tier,'weak')
    return f'<span class="badge-{cls}">{tier}</span>'

def conf_badge(conf):
    return f'<span class="badge-{(conf or "LOW").lower()}">{conf}</span>'

def score_org(org_name, org_type, role):
    prompt   = SCORING_PROMPT.format(org_name=org_name, org_type=org_type, role=role)
    response = client.chat.completions.create(model=MODEL, messages=[{"role":"user","content":prompt}], response_format={"type":"json_object"}, max_tokens=1000)
    data = json.loads(response.choices[0].message.content.strip())
    data['_input_tokens']  = response.usage.prompt_tokens
    data['_output_tokens'] = response.usage.completion_tokens
    data['_tokens']        = response.usage.total_tokens
    return data

def build_scored_df(df, results):
    rows = []
    for _, contact in df.iterrows():
        org_name = contact['Organization']
        org_data = next((r for r in results if r['org_name'] == org_name), None)
        if not org_data:
            continue
        sf, ha, em = org_data.get('sector_fit_score',5), org_data.get('halo_score',5), org_data.get('emerging_fit_score',5)
        rel = float(contact.get('Relationship Depth', 5))
        composite = compute_composite(sf, rel, ha, em)
        rows.append({
            'Contact Name':contact.get('Contact Name',''), 'Organization':org_name,
            'Type':contact.get('Org Type',''), 'Region':contact.get('Region',''),
            'Status':contact.get('Contact Status',''), 'Sector Fit':sf,
            'Rel Depth':rel, 'Halo':ha, 'Emerging Fit':em, 'Composite':composite,
            'Tier':classify_tier(composite), 'AUM':org_data.get('aum_estimate','Unknown'),
            'Confidence':org_data.get('confidence','LOW'), 'Is GP':org_data.get('is_gp_or_service_provider',False),
            'Why':org_data.get('sector_fit_reasoning',''), 'Halo Reasoning':org_data.get('halo_reasoning',''),
            'EM Reasoning':org_data.get('emerging_fit_reasoning',''), 'Org Summary':org_data.get('org_description',''),
        })
    return pd.DataFrame(rows).sort_values('Composite', ascending=False).reset_index(drop=True)


# ════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════
with st.sidebar:
    try:
        st.image("pacezero_logo.png", width=140)
    except:
        st.markdown("**PaceZero**")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-family:DM Mono,monospace;font-size:10px;color:#8b949e;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:16px;'>LP Scoring Engine v1.0</div>", unsafe_allow_html=True)
    st.markdown("---")
    role_view = st.radio("DASHBOARD VIEW", ["Executive", "Analyst", "Terms Dictionary"], index=0)
    st.markdown("---")

    st.markdown("<div style='font-family:DM Mono,monospace;font-size:10px;color:#8b949e;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:10px;'>Scoring Weights</div>", unsafe_allow_html=True)
    for dim, w in [("Sector Fit","35%"),("Rel. Depth","30%"),("Halo Value","20%"),("Emerging Fit","15%")]:
        st.markdown(f"<div style='display:flex;justify-content:space-between;margin-bottom:6px;'><span style='font-size:12px;color:#8b949e;'>{dim}</span><span style='font-family:DM Mono,monospace;font-size:12px;color:#58a6ff;'>{w}</span></div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div style='font-family:DM Mono,monospace;font-size:10px;color:#8b949e;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:8px;'>Tier Thresholds</div>", unsafe_allow_html=True)
    for label, thr, color in [("PRIORITY CLOSE","≥ 8.0","#3fb950"),("STRONG FIT","≥ 6.5","#58a6ff"),("MODERATE FIT","≥ 5.0","#e3b341"),("WEAK FIT","< 5.0","#f85149")]:
        st.markdown(f"<div style='display:flex;justify-content:space-between;margin-bottom:6px;'><span style='font-size:11px;color:{color};font-family:DM Mono,monospace;'>{label}</span><span style='font-family:DM Mono,monospace;font-size:11px;color:#8b949e;'>{thr}</span></div>", unsafe_allow_html=True)


# ════════════════════════════════════════
# HEADER
# ════════════════════════════════════════
st.markdown("""
<div style='padding:28px 0 8px 0;'>
    <div style='font-family:DM Mono,monospace;font-size:11px;color:#6b7280;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:8px;'>PaceZero Capital Partners — Fund II</div>
    <div style='font-family:Playfair Display,serif;font-size:38px;font-weight:700;color:#1a1a2e;line-height:1.2;'>LP Prospect Scoring Engine</div>
    <div style='font-size:15px;color:#6b7280;margin-top:8px;'>AI-powered enrichment and scoring for your fundraising pipeline</div>
</div>
""", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)


# ════════════════════════════════════════
# TERMS DICTIONARY
# ════════════════════════════════════════
if role_view == "Terms Dictionary":
    st.markdown('<div class="section-header">Terms Dictionary</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Jargon guide for the LP prospect scoring system</div>', unsafe_allow_html=True)
    search   = st.text_input("Search terms", placeholder="e.g. LP, AUM, composite...")
    filtered = {k:v for k,v in TERMS.items() if not search or search.lower() in k.lower() or search.lower() in v.lower()}
    if not filtered:
        st.info("No terms found.")
    else:
        for term, definition in filtered.items():
            st.markdown(f'<div class="term-card"><div class="term-name">{term}</div><div class="term-def">{definition}</div></div>', unsafe_allow_html=True)
    st.stop()


# ════════════════════════════════════════
# QUICK SCORE
# ════════════════════════════════════════
st.markdown('<div class="section-header">Quick Score</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Manually score a single prospect without uploading a CSV</div>', unsafe_allow_html=True)

with st.expander("Open Quick Score", expanded=False):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**Organization**")
        manual_org      = st.text_input("Organization name", placeholder="e.g. Carnegie Corporation", label_visibility="collapsed", key="manual_org")
        manual_org_type = st.selectbox("Org Type", ORG_TYPES, key="manual_org_type")
    with col2:
        st.markdown("**Contact Person**")
        manual_contact = st.text_input("Contact name", placeholder="e.g. Jane Smith", label_visibility="collapsed", key="manual_contact")
        manual_role    = st.text_input("Role / Title", placeholder="e.g. Director of Investments", key="manual_role")
        manual_status  = st.selectbox("Contact Status", CONTACT_STATUSES, key="manual_status")
    with col3:
        st.markdown("**Relationship**")
        manual_rel = st.slider("Relationship Depth (D2)", 1, 10, 5, key="manual_rel", help="Your CRM score — how warm is this relationship?")
        st.caption("Sector Fit, Halo, and Emerging Fit are fully AI-generated.")

    run_manual = st.button("Score This Prospect", key="run_manual", type="primary")

    if run_manual:
        if not manual_org:
            st.warning("Please enter an organization name.")
        else:
            with st.spinner(f"Researching {manual_org}..."):
                try:
                    data      = score_org(manual_org, manual_org_type, manual_role or "Unknown")
                    sf        = data.get('sector_fit_score', 5)
                    ha        = data.get('halo_score', 5)
                    em        = data.get('emerging_fit_score', 5)
                    composite = compute_composite(sf, manual_rel, ha, em)
                    tier      = classify_tier(composite)

                    st.markdown("<hr>", unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style='margin-bottom:12px;'>
                        <span style='font-family:Playfair Display,serif;font-size:22px;font-weight:700;color:#1a1a2e;'>{manual_org}</span>
                        &nbsp;&nbsp;{tier_badge(tier)}&nbsp;&nbsp;{conf_badge(data.get('confidence','LOW'))}
                    </div>
                    <div style='font-size:13px;color:#6b7280;margin-bottom:20px;'>{data.get('org_description','')}</div>
                    """, unsafe_allow_html=True)

                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Composite",    composite)
                    m2.metric("Sector Fit",   sf)
                    m3.metric("Rel Depth",    manual_rel, help="Your manual input")
                    m4.metric("Halo",         ha)
                    m5.metric("Emerging Fit", em)

                    with st.expander("View full AI reasoning"):
                        st.markdown(f"**AUM:** {data.get('aum_estimate','Unknown')}")
                        st.markdown(f"**GP Flag:** {'Yes — not an LP' if data.get('is_gp_or_service_provider') else 'No'}")
                        st.info(f"**Sector Fit:** {data.get('sector_fit_reasoning','')}")
                        st.info(f"**Halo:** {data.get('halo_reasoning','')}")
                        st.info(f"**Emerging Fit:** {data.get('emerging_fit_reasoning','')}")
                        cost_q = (data.get('_input_tokens',0)/1000*0.0025)+(data.get('_output_tokens',0)/1000*0.0100)
                        st.caption(f"Tokens: {data.get('_tokens',0):,}  |  Cost: ${cost_q:.4f}")
                except Exception as e:
                    st.error(f"Scoring failed: {e}")

st.markdown("<hr>", unsafe_allow_html=True)


# ════════════════════════════════════════
# FULL PIPELINE
# ════════════════════════════════════════
st.markdown('<div class="section-header">Full Pipeline</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Upload your prospect CSV to enrich and score the full pipeline</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload challenge_contacts.csv", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=['Contact Name','Organization'])
    df['Organization']       = df['Organization'].str.strip()
    df['Relationship Depth'] = pd.to_numeric(df['Relationship Depth'], errors='coerce').fillna(5)
    test_mask = df['Organization'].isin(TEST_ORGS)
    train_df  = df[~test_mask].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Contacts",  len(df))
    c2.metric("Unique Orgs",     train_df['Organization'].nunique())
    c3.metric("Held Out (Test)", int(test_mask.sum()))

    if st.button("Run Full Pipeline", type="primary"):
        results  = []
        unique_orgs = train_df.groupby('Organization').first().reset_index()
        total       = len(unique_orgs)
        progress    = st.progress(0)
        status      = st.empty()
        total_in = total_out = 0

        for i, (_, row) in enumerate(unique_orgs.iterrows()):
            org_name = row['Organization']
            status.caption(f"Scoring {i+1}/{total} — {org_name}")
            try:
                data = score_org(org_name, row['Org Type'], row['Role'])
                data['org_name'] = org_name
                data['org_type'] = row['Org Type']
                results.append(data)
                total_in  += data.get('_input_tokens', 0)
                total_out += data.get('_output_tokens', 0)
            except Exception as e:
                st.warning(f"Failed: {org_name} — {e}")
            progress.progress((i+1)/total)
            time.sleep(0.5)

        scored_df = build_scored_df(train_df, results)
        cost      = (total_in/1000*0.0025) + (total_out/1000*0.0100)
        status.caption("Pipeline complete")

        st.session_state['scored_df'] = scored_df
        st.session_state['results']   = results
        st.session_state['cost']      = cost
        st.session_state['tokens']    = total_in + total_out


# ════════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════════
if 'scored_df' in st.session_state:
    scored_df   = st.session_state['scored_df']
    cost        = st.session_state['cost']
    tokens      = st.session_state['tokens']
    tier_counts = scored_df['Tier'].value_counts()

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── EXECUTIVE ──
    if role_view == "Executive":
        st.markdown('<div class="section-header">Executive Summary</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">High-level pipeline overview for GP and partner review</div>', unsafe_allow_html=True)

        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Prospects", len(scored_df))
        c2.metric("Priority Close",  tier_counts.get('PRIORITY CLOSE', 0))
        c3.metric("Strong Fit",      tier_counts.get('STRONG FIT', 0))
        c4.metric("Avg Composite",   round(scored_df['Composite'].mean(), 2))
        c5.metric("Run Cost",        f"${cost:.4f}")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:DM Mono,monospace;font-size:11px;color:#065f46;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:16px;'>Priority Close — Action Required This Week</div>", unsafe_allow_html=True)

        priority = scored_df[scored_df['Tier'] == 'PRIORITY CLOSE']
        if len(priority):
            for _, row in priority.iterrows():
                with st.expander(f"{row['Organization']}   ·   Score {row['Composite']}   ·   {row['AUM']}"):
                    d1, d2 = st.columns(2)
                    with d1:
                        st.markdown(f"**Contact:** {row['Contact Name']}")
                        st.markdown(f"**Type:** {row['Type']}")
                        st.markdown(f"**Status:** {row['Status']}")
                        st.markdown(f"**Confidence:** {row['Confidence']}")
                    with d2:
                        s1, s2 = st.columns(2)
                        s1.metric("Composite",  row['Composite'])
                        s2.metric("Sector Fit", row['Sector Fit'])
                        s3, s4 = st.columns(2)
                        s3.metric("Halo",         row['Halo'])
                        s4.metric("Emerging Fit", row['Emerging Fit'])
                    st.info(row['Why'])
        else:
            st.info("No Priority Close prospects in this run.")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:DM Mono,monospace;font-size:11px;color:#1e40af;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:12px;'>Strong Fit Pipeline</div>", unsafe_allow_html=True)
        strong = scored_df[scored_df['Tier'] == 'STRONG FIT'][['Contact Name','Organization','Type','Composite','AUM','Confidence']]
        st.dataframe(strong, use_container_width=True, hide_index=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:DM Mono,monospace;font-size:11px;color:#6b7280;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:12px;'>Pipeline Tier Breakdown</div>", unsafe_allow_html=True)
        tier_order = ['PRIORITY CLOSE','STRONG FIT','MODERATE FIT','WEAK FIT']
        st.bar_chart(pd.DataFrame({'Tier':tier_order,'Count':[tier_counts.get(t,0) for t in tier_order]}).set_index('Tier'))

    # ── ANALYST ──
    elif role_view == "Analyst":
        st.markdown('<div class="section-header">Analyst Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Full pipeline with filters, scoring breakdown, and prospect deep dive</div>', unsafe_allow_html=True)

        fc1, fc2, fc3, fc4 = st.columns(4)
        tier_filter = fc1.multiselect("Tier", ['PRIORITY CLOSE','STRONG FIT','MODERATE FIT','WEAK FIT'], default=['PRIORITY CLOSE','STRONG FIT','MODERATE FIT','WEAK FIT'])
        type_filter = fc2.multiselect("Org Type", sorted(scored_df['Type'].unique()), default=list(scored_df['Type'].unique()))
        conf_filter = fc3.multiselect("Confidence", ['HIGH','MEDIUM','LOW'], default=['HIGH','MEDIUM','LOW'])
        hide_gp     = fc4.checkbox("Hide GPs", value=True)

        filtered_df = scored_df[scored_df['Tier'].isin(tier_filter) & scored_df['Type'].isin(type_filter) & scored_df['Confidence'].isin(conf_filter)]
        if hide_gp:
            filtered_df = filtered_df[filtered_df['Is GP'] == False]

        st.caption(f"{len(filtered_df)} prospects shown")

        k1, k2, k3, k4, k5, k6 = st.columns(6)
        k1.metric("Showing",        len(filtered_df))
        k2.metric("Priority Close", len(filtered_df[filtered_df['Tier']=='PRIORITY CLOSE']))
        k3.metric("Strong Fit",     len(filtered_df[filtered_df['Tier']=='STRONG FIT']))
        k4.metric("Avg Composite",  round(filtered_df['Composite'].mean(),2) if len(filtered_df) else 0)
        k5.metric("Total Tokens",   f"{tokens:,}")
        k6.metric("Run Cost",       f"${cost:.4f}")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:DM Mono,monospace;font-size:11px;color:#6b7280;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:12px;'>Full Pipeline</div>", unsafe_allow_html=True)
        display_cols = ['Contact Name','Organization','Type','Region','Sector Fit','Rel Depth','Halo','Emerging Fit','Composite','Tier','AUM','Confidence']
        st.dataframe(filtered_df[display_cols], use_container_width=True, hide_index=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:DM Mono,monospace;font-size:11px;color:#6b7280;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:12px;'>Score Distribution</div>", unsafe_allow_html=True)

        full_index = pd.RangeIndex(1, 11)
        dc1, dc2, dc3 = st.columns(3)

        sf_counts = filtered_df['Sector Fit'].value_counts().reindex(full_index, fill_value=0).reset_index()
        sf_counts.columns = ['Score','Count']
        dc1.markdown("**D1 — Sector & Mandate Fit**")
        dc1.bar_chart(sf_counts.set_index('Score'))

        halo_counts = filtered_df['Halo'].value_counts().reindex(full_index, fill_value=0).reset_index()
        halo_counts.columns = ['Score','Count']
        dc2.markdown("**D3 — Halo & Strategic Value**")
        dc2.bar_chart(halo_counts.set_index('Score'))

        em_counts = filtered_df['Emerging Fit'].value_counts().reindex(full_index, fill_value=0).reset_index()
        em_counts.columns = ['Score','Count']
        dc3.markdown("**D4 — Emerging Manager Fit**")
        dc3.bar_chart(em_counts.set_index('Score'))

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:DM Mono,monospace;font-size:11px;color:#6b7280;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:12px;'>Prospect Deep Dive</div>", unsafe_allow_html=True)

        selected_org = st.selectbox("Select an organization", options=filtered_df['Organization'].tolist())
        if selected_org:
            row = filtered_df[filtered_df['Organization'] == selected_org].iloc[0]
            st.markdown(f"""
            <div style='margin-bottom:12px;'>
                <span style='font-family:Playfair Display,serif;font-size:22px;font-weight:700;color:#1a1a2e;'>{row['Organization']}</span>
                &nbsp;&nbsp;{tier_badge(row['Tier'])}&nbsp;&nbsp;{conf_badge(row['Confidence'])}
                {'&nbsp;&nbsp;<span style="background:#fee2e2;color:#991b1b;padding:2px 8px;border-radius:10px;font-size:10px;font-family:DM Mono,monospace;">GP FLAG</span>' if row['Is GP'] else ''}
            </div>
            <div style='font-size:13px;color:#6b7280;margin-bottom:20px;'>{row['Org Summary']}</div>
            """, unsafe_allow_html=True)

            d1, d2 = st.columns(2)
            with d1:
                st.markdown(f"**Contact:** {row['Contact Name']}")
                st.markdown(f"**Type:** {row['Type']}")
                st.markdown(f"**Region:** {row['Region']}")
                st.markdown(f"**Status:** {row['Status']}")
                st.markdown(f"**AUM:** {row['AUM']}")
            with d2:
                st.metric("Composite Score", row['Composite'])
                s1, s2 = st.columns(2)
                s1.metric("Sector Fit (D1)",   row['Sector Fit'])
                s2.metric("Rel Depth (D2)",    row['Rel Depth'])
                s3, s4 = st.columns(2)
                s3.metric("Halo (D3)",         row['Halo'])
                s4.metric("Emerging Fit (D4)", row['Emerging Fit'])

            st.markdown("<br>", unsafe_allow_html=True)
            st.info(f"**Sector Fit:** {row['Why']}")
            st.info(f"**Halo:** {row['Halo Reasoning']}")
            st.info(f"**Emerging Fit:** {row['EM Reasoning']}")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:DM Mono,monospace;font-size:11px;color:#6b7280;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:12px;'>API Cost Tracker</div>", unsafe_allow_html=True)
        cost_per_org = cost / max(len(st.session_state['results']), 1)
        st.table(pd.DataFrame({'Prospects':[100,500,1000,5000],'Est. Cost':[f"${cost_per_org*n:.2f}" for n in [100,500,1000,5000]]}))

        st.markdown("<hr>", unsafe_allow_html=True)
        st.download_button(label="Download Scored Results as CSV", data=scored_df.to_csv(index=False), file_name="pacezero_scored_pipeline.csv", mime="text/csv")
