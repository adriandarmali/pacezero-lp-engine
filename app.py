import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import urllib.parse
from openai import OpenAI

st.set_page_config(
    page_title="PaceZero LP Scoring Engine",
    page_icon="pacezero_logo.png",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=DM+Mono:wght@400;500&family=Playfair+Display:wght@600;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background-color: #f8f9fa; color: #1a1a2e; }
section[data-testid="stSidebar"] { background-color: #1a1a2e; border-right: 1px solid #2d2d44; }
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }
header[data-testid="stHeader"] { background: #f8f9fa; border-bottom: 1px solid #e0e0e0; }
[data-testid="metric-container"] { background: #ffffff; border: 1px solid #e0e4ec; border-radius: 10px; padding: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
[data-testid="metric-container"] label { color: #6b7280 !important; font-size: 11px !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; font-family: 'DM Mono', monospace !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #1a1a2e !important; font-size: 26px !important; font-weight: 700 !important; font-family: 'Playfair Display', serif !important; }
[data-testid="stDataFrame"] { border: 1px solid #e0e4ec; border-radius: 10px; overflow: hidden; background: #ffffff; }
.stButton > button { background: #1a1a2e; color: #ffffff; border: none; border-radius: 8px; font-family: 'DM Sans', sans-serif; font-weight: 500; padding: 10px 24px; transition: all 0.2s; }
.stButton > button:hover { background: #2d2d50; transform: translateY(-1px); }
.streamlit-expanderHeader { background: #ffffff !important; border: 1px solid #e0e4ec !important; border-radius: 8px !important; color: #1a1a2e !important; font-weight: 500 !important; }
.streamlit-expanderContent { background: #ffffff !important; border: 1px solid #e0e4ec !important; border-top: none !important; }
.stProgress > div > div { background-color: #1a1a2e; }
hr { border-color: #e0e4ec; margin: 28px 0; }
.stAlert { background: #ffffff; border: 1px solid #e0e4ec; border-radius: 8px; }
.badge-priority { background:#d1fae5; color:#065f46; padding:3px 10px; border-radius:12px; font-size:11px; font-weight:600; font-family:'DM Mono',monospace; }
.badge-strong   { background:#dbeafe; color:#1e40af; padding:3px 10px; border-radius:12px; font-size:11px; font-weight:600; font-family:'DM Mono',monospace; }
.badge-moderate { background:#fef3c7; color:#92400e; padding:3px 10px; border-radius:12px; font-size:11px; font-weight:600; font-family:'DM Mono',monospace; }
.badge-weak     { background:#fee2e2; color:#991b1b; padding:3px 10px; border-radius:12px; font-size:11px; font-weight:600; font-family:'DM Mono',monospace; }
.badge-high     { background:#d1fae5; color:#065f46; padding:2px 8px; border-radius:10px; font-size:10px; font-weight:600; font-family:'DM Mono',monospace; }
.badge-medium   { background:#fef3c7; color:#92400e; padding:2px 8px; border-radius:10px; font-size:10px; font-weight:600; font-family:'DM Mono',monospace; }
.badge-low      { background:#fee2e2; color:#991b1b; padding:2px 8px; border-radius:10px; font-size:10px; font-weight:600; font-family:'DM Mono',monospace; }
.section-header { font-family:'Playfair Display',serif; font-size:24px; font-weight:700; color:#1a1a2e; margin-bottom:4px; }
.section-sub    { font-family:'DM Mono',monospace; font-size:11px; color:#6b7280; letter-spacing:0.08em; text-transform:uppercase; margin-bottom:20px; }
.term-card { background:#ffffff; border:1px solid #e0e4ec; border-left:3px solid #1a1a2e; border-radius:0 8px 8px 0; padding:14px 18px; margin-bottom:10px; }
.term-name { font-family:'DM Mono',monospace; font-size:13px; font-weight:500; color:#1a1a2e; margin-bottom:4px; }
.term-def  { font-size:13px; color:#6b7280; line-height:1.6; }
</style>
""", unsafe_allow_html=True)

# ── API & Constants ──
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

PRE-SEARCH RULE — ABBREVIATIONS:
  If the org name looks like an abbreviation (all caps, no spaces),
  search for the full name first before scoring.

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
  GP hard cap: halo <= 3

RUBRIC 3 — EMERGING MANAGER FIT (1-10):
  9-10 : Documented emerging manager programme; backed Fund I/II before
  7-8  : Strong structural appetite: SFO, Foundation, faith-based pension,
         or smaller endowment with open mandate and no evidence of restrictions;
         OR known first-time fund backer
  5-6  : No explicit programme but org type typically flexible (MFO, FoF)
  3-4  : Large institution; likely prefers established managers
  1-2  : Known policy against emerging managers OR confirmed GP
  Do NOT default to 4-5 when data is thin.
  Absence of information is NOT evidence of restrictions.
  GP hard cap: emerging_fit <= 2

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

STATUS_GUIDANCE = {
    "New Contact": """EMAIL GOAL: First-time acquisition. They have never heard of PaceZero.
  - Lead with the recent activity hook to show you did your homework
  - Clearly explain what PaceZero does and why it is relevant to their mandate
  - Ask for a 20-minute introductory call. Under 150 words.""",
    "Previously Contacted": """EMAIL GOAL: Re-engagement. They have been contacted before but did not respond.
  - Acknowledge prior outreach briefly and naturally — do not grovel
  - Lead with something new: the recent activity hook or a fund update
  - Soft ask for a call. Under 130 words.""",
    "Existing Contact": """EMAIL GOAL: Relationship maintenance. They already know PaceZero.
  - Skip the intro — they know who you are
  - Open with the recent activity to show you follow their work
  - Frame as a check-in, not a pitch. Ask for 15 minutes. Under 120 words.""",
    "In Diligence": """EMAIL GOAL: Support and momentum. They are actively evaluating PaceZero.
  - Acknowledge the process without being pushy
  - Reference recent activity that reinforces the fit
  - Offer to answer questions or provide materials. Under 120 words.""",
    "Committed": """EMAIL GOAL: Relationship nurture. They have already committed.
  - Warm, personal tone — they are a partner now
  - Reference their recent activity as a shared interest
  - No ask needed — stay warm. Under 100 words.""",
    "Passed": """EMAIL GOAL: Keep the door open. They previously declined.
  - Do not reference the pass directly
  - Lead with the recent activity hook — show circumstances may have changed
  - Very light touch, no hard sell. Under 100 words.""",
}

# ── Helpers ──
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

def generate_draft(row, sender_name, sender_title, tone):
    research_prompt = f"""Search for the most recent publicly available activities from {row['Organization']} relevant to:
  - Private credit or alternative investment allocations
  - ESG, impact investing, or sustainability commitments
  - Emerging manager programmes or first-time fund commitments
  - New hires or leadership changes in their investment office
  - Published reports, conference appearances, or media coverage
Return 2-3 specific findings with dates. Include the date for every finding so the reader knows how recent it is.
If the most relevant information is older than 18 months, include it anyway — just make sure the date is clearly stated."""
    try:
        r1 = client.chat.completions.create(model=MODEL, messages=[{"role":"user","content":research_prompt}], max_tokens=300)
        recent_activity = r1.choices[0].message.content.strip()
        r_tok = r1.usage.total_tokens
    except Exception:
        recent_activity = "No recent activity found."
        r_tok = 0

    status   = row.get('Status', 'New Contact')
    guidance = STATUS_GUIDANCE.get(status, STATUS_GUIDANCE["New Contact"])
    draft_prompt = f"""You are a pitch person and fundraising analyst at PaceZero Capital Partners,
a Toronto-based sustainability-focused private credit firm raising Fund II.
PaceZero focus areas: Agriculture & Ecosystems, Energy Transition, Health & Education.
Deal size: $3M to $20M. Emerging manager, Fund II.

CONTACT STATUS: {status}
{guidance}
TONE: {tone}

PROSPECT:
  Organization : {row['Organization']}
  Contact Name : {row['Contact Name']}
  Org Type     : {row['Type']}
  AUM          : {row['AUM']}
  Tier         : {row['Tier']} (composite {row['Composite']} / 10)

SCORING CONTEXT (use to personalize — do not quote scores in the email):
  Sector alignment : {row['Why']}
  Halo context     : {row['Halo Reasoning']}
  Emerging fit     : {row['EM Reasoning']}
  Org summary      : {row['Org Summary']}

RECENT ACTIVITY (use the most relevant finding as your opening hook):
{recent_activity}

SENDER:
  Name  : {sender_name or '[Your Name]'}
  Title : {sender_title or '[Your Title]'}

Return ONLY the email. Subject line first, then a blank line, then the body.
No preamble, no commentary, no labels."""
    r2    = client.chat.completions.create(model=MODEL, messages=[{"role":"user","content":draft_prompt}], max_tokens=400)
    draft = r2.choices[0].message.content.strip()
    d_tok = r2.usage.total_tokens
    total_tok = r_tok + d_tok
    cost = (total_tok / 1_000_000) * 12.50
    return draft, recent_activity, total_tok, cost

def gmail_link(draft):
    lines   = draft.split("\n", 1)
    subject = lines[0].replace("Subject:", "").strip() if lines else "Introduction — PaceZero Capital Partners"
    body    = lines[1].strip() if len(lines) > 1 else draft
    return f"https://mail.google.com/mail/?view=cm&fs=1&su={urllib.parse.quote(subject)}&body={urllib.parse.quote(body)}"

DEMO_LIMIT_NOTE = """
<div style='margin-top:12px;background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:12px 16px;font-size:12px;color:#92400e;line-height:1.6;'>
    <strong>&#9733; Demo limitation:</strong> The Gmail button opens a compose window with subject and body
    pre-filled via a URL deep link. It does not authenticate with your Gmail account or save to Drafts.
    In production this would use the <strong>Gmail API with OAuth 2.0</strong> to push directly into your
    Drafts folder — requiring a one-time Google login and credentials in Streamlit Secrets.
</div>"""

BATCH_LIMIT_NOTE = """
<div style='margin-bottom:16px;background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:12px 16px;font-size:12px;color:#92400e;line-height:1.6;'>
    <strong>&#9733; Demo limitation:</strong> Each Gmail button opens an individual compose window.
    In production, batch drafts would be pushed directly into Gmail Drafts via the Gmail API with OAuth 2.0,
    letting you review and send each email from your inbox without opening multiple tabs.
</div>"""


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
    page = st.radio("NAVIGATE", ["📊 Report", "💰 Cost & Tokens", "📖 Terms Dictionary", "✉️ Email Drafting"], index=0)
    st.markdown("---")
    st.markdown("<div style='font-family:DM Mono,monospace;font-size:10px;color:#8b949e;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:10px;'>Scoring Weights</div>", unsafe_allow_html=True)
    for dim, w in [("Sector Fit","35%"),("Rel. Depth","30%"),("Halo Value","20%"),("Emerging Fit","15%")]:
        st.markdown(f"<div style='display:flex;justify-content:space-between;margin-bottom:6px;'><span style='font-size:12px;color:#8b949e;'>{dim}</span><span style='font-family:DM Mono,monospace;font-size:12px;color:#58a6ff;'>{w}</span></div>", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='font-family:DM Mono,monospace;font-size:10px;color:#8b949e;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:8px;'>Tier Thresholds</div>", unsafe_allow_html=True)
    for label, thr, color in [("PRIORITY CLOSE","≥ 8.0","#3fb950"),("STRONG FIT","≥ 6.5","#58a6ff"),("MODERATE FIT","≥ 5.0","#e3b341"),("WEAK FIT","< 5.0","#f85149")]:
        st.markdown(f"<div style='display:flex;justify-content:space-between;margin-bottom:6px;'><span style='font-size:11px;color:{color};font-family:DM Mono,monospace;'>{label}</span><span style='font-family:DM Mono,monospace;font-size:11px;color:#8b949e;'>{thr}</span></div>", unsafe_allow_html=True)


# ════════════════════════════════════════
# GLOBAL HEADER
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
# QUICK SCORE + CSV UPLOAD (always visible)
# ════════════════════════════════════════
with st.expander("Quick Score — score a single org manually"):
    q1, q2 = st.columns(2)
    with q1:
        manual_org      = st.text_input("Organization name", key="manual_org")
        manual_org_type = st.selectbox("Org Type", ORG_TYPES, key="manual_org_type")
        manual_contact  = st.text_input("Contact name", key="manual_contact")
    with q2:
        manual_role   = st.text_input("Role / Title", key="manual_role")
        manual_status = st.selectbox("Contact Status", CONTACT_STATUSES, key="manual_status")
        manual_rel    = st.slider("Relationship Depth (D2)", 1, 10, 5, key="manual_rel")

    if st.button("Score This Org", key="manual_score_btn") and manual_org:
        with st.spinner(f"Scoring {manual_org}..."):
            try:
                data      = score_org(manual_org, manual_org_type, manual_role or "Unknown")
                sf        = data.get('sector_fit_score', 5)
                ha        = data.get('halo_score', 5)
                em        = data.get('emerging_fit_score', 5)
                composite = compute_composite(sf, manual_rel, ha, em)
                tier      = classify_tier(composite)
                st.markdown(f"""
                <div style='margin-bottom:12px;margin-top:16px;'>
                    <span style='font-family:Playfair Display,serif;font-size:22px;font-weight:700;color:#1a1a2e;'>{manual_org}</span>
                    &nbsp;&nbsp;{tier_badge(tier)}&nbsp;&nbsp;{conf_badge(data.get('confidence','LOW'))}
                </div>
                <div style='font-size:13px;color:#6b7280;margin-bottom:20px;'>{data.get('org_description','')}</div>
                """, unsafe_allow_html=True)
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Composite",    composite)
                m2.metric("Sector Fit",   sf)
                m3.metric("Rel Depth",    manual_rel)
                m4.metric("Halo",         ha)
                m5.metric("Emerging Fit", em)
                with st.expander("View full AI reasoning"):
                    st.markdown(f"**AUM:** {data.get('aum_estimate','Unknown')}")
                    st.markdown(f"**GP Flag:** {'Yes' if data.get('is_gp_or_service_provider') else 'No'}")
                    st.info(f"**Sector Fit:** {data.get('sector_fit_reasoning','')}")
                    st.info(f"**Halo:** {data.get('halo_reasoning','')}")
                    st.info(f"**Emerging Fit:** {data.get('emerging_fit_reasoning','')}")
                    cost_q = (data.get('_input_tokens',0)/1000*0.0025)+(data.get('_output_tokens',0)/1000*0.0100)
                    st.caption(f"Tokens: {data.get('_tokens',0):,}  |  Cost: ${cost_q:.4f}")
            except Exception as e:
                st.error(f"Scoring failed: {e}")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown('<div class="section-sub">Upload your prospect CSV to run the full pipeline</div>', unsafe_allow_html=True)
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
        results = []
        unique_orgs = train_df.groupby('Organization').first().reset_index()
        total       = len(unique_orgs)
        progress    = st.progress(0)
        status_msg  = st.empty()
        total_in = total_out = 0

        for i, (_, row) in enumerate(unique_orgs.iterrows()):
            org_name = row['Organization']
            status_msg.caption(f"Scoring {i+1}/{total} — {org_name}")
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
        status_msg.caption("Pipeline complete")
        st.session_state['scored_df'] = scored_df
        st.session_state['results']   = results
        st.session_state['cost']      = cost
        st.session_state['tokens']    = total_in + total_out

st.markdown("<hr>", unsafe_allow_html=True)


# ════════════════════════════════════════
# PAGE: REPORT
# ════════════════════════════════════════
if page == "📊 Report":
    if 'scored_df' not in st.session_state:
        st.info("Upload and score a CSV above to see the report.")
    else:
        scored_df   = st.session_state['scored_df']
        cost        = st.session_state['cost']
        tier_counts = scored_df['Tier'].value_counts()

        # Executive Summary
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

        # Analyst Section
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Analyst View</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Full pipeline with filters, scoring breakdown, and prospect deep dive</div>', unsafe_allow_html=True)

        fc1, fc2, fc3, fc4 = st.columns(4)
        tier_filter = fc1.multiselect("Tier", ['PRIORITY CLOSE','STRONG FIT','MODERATE FIT','WEAK FIT'], default=['PRIORITY CLOSE','STRONG FIT','MODERATE FIT','WEAK FIT'])
        type_filter = fc2.multiselect("Org Type", sorted(scored_df['Type'].unique()), default=list(scored_df['Type'].unique()))
        conf_filter = fc3.multiselect("Confidence", ['HIGH','MEDIUM','LOW'], default=['HIGH','MEDIUM','LOW'])
        hide_gp     = fc4.checkbox("Hide GPs", value=True)

        filtered_df = scored_df[
            scored_df['Tier'].isin(tier_filter) &
            scored_df['Type'].isin(type_filter) &
            scored_df['Confidence'].isin(conf_filter)
        ]
        if hide_gp:
            filtered_df = filtered_df[filtered_df['Is GP'] == False]

        st.caption(f"{len(filtered_df)} prospects shown")

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Showing",        len(filtered_df))
        k2.metric("Priority Close", len(filtered_df[filtered_df['Tier']=='PRIORITY CLOSE']))
        k3.metric("Strong Fit",     len(filtered_df[filtered_df['Tier']=='STRONG FIT']))
        k4.metric("Avg Composite",  round(filtered_df['Composite'].mean(),2) if len(filtered_df) else 0)

        st.markdown("<hr>", unsafe_allow_html=True)
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
                s1.metric("Sector Fit (D1)", row['Sector Fit'])
                s2.metric("Rel Depth (D2)",  row['Rel Depth'])
                s3, s4 = st.columns(2)
                s3.metric("Halo (D3)",          row['Halo'])
                s4.metric("Emerging Fit (D4)",  row['Emerging Fit'])
            st.markdown("<br>", unsafe_allow_html=True)
            st.info(f"**Sector Fit:** {row['Why']}")
            st.info(f"**Halo:** {row['Halo Reasoning']}")
            st.info(f"**Emerging Fit:** {row['EM Reasoning']}")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.download_button("Download Scored Results as CSV", data=scored_df.to_csv(index=False), file_name="pacezero_scored_pipeline.csv", mime="text/csv")


# ════════════════════════════════════════
# PAGE: COST & TOKENS
# ════════════════════════════════════════
elif page == "💰 Cost & Tokens":
    st.markdown('<div class="section-header">Cost & Token Usage</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">API usage breakdown and scaling projections</div>', unsafe_allow_html=True)

    if 'scored_df' not in st.session_state:
        st.info("Run the pipeline above to see cost data.")
    else:
        cost    = st.session_state['cost']
        tokens  = st.session_state['tokens']
        results = st.session_state['results']
        n_orgs  = len(results)

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Total Tokens",  f"{tokens:,}")
        r2.metric("Total Cost",    f"${cost:.4f}")
        r3.metric("Orgs Scored",   n_orgs)
        r4.metric("Cost per Org",  f"${cost/max(n_orgs,1):.4f}")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:DM Mono,monospace;font-size:11px;color:#6b7280;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:12px;'>Scaling Projection</div>", unsafe_allow_html=True)
        cost_per_org = cost / max(n_orgs, 1)
        scale_df = pd.DataFrame({
            'Prospects':       [100, 500, 1000, 5000, 10000],
            'Est. Total Cost': [f"${cost_per_org*n:.2f}" for n in [100,500,1000,5000,10000]],
            'Cost per Org':    [f"${cost_per_org:.4f}"]*5,
        })
        st.table(scale_df)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div style='font-family:DM Mono,monospace;font-size:11px;color:#6b7280;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:12px;'>Token Breakdown by Org</div>", unsafe_allow_html=True)
        token_rows = [{
            'Organization':  r.get('org_name',''),
            'Input Tokens':  r.get('_input_tokens',0),
            'Output Tokens': r.get('_output_tokens',0),
            'Total Tokens':  r.get('_tokens',0),
            'Cost':          f"${(r.get('_input_tokens',0)/1000*0.0025)+(r.get('_output_tokens',0)/1000*0.0100):.4f}",
        } for r in results]
        st.dataframe(pd.DataFrame(token_rows).sort_values('Total Tokens', ascending=False), use_container_width=True, hide_index=True)


# ════════════════════════════════════════
# PAGE: TERMS DICTIONARY
# ════════════════════════════════════════
elif page == "📖 Terms Dictionary":
    st.markdown('<div class="section-header">Terms Dictionary</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Definitions for LP scoring concepts and fundraising terminology</div>', unsafe_allow_html=True)

    search = st.text_input("Search terms", placeholder="e.g. emerging manager, halo, composite...")
    filtered_terms = {k: v for k, v in TERMS.items() if not search or search.lower() in k.lower() or search.lower() in v.lower()}

    for term, definition in filtered_terms.items():
        st.markdown(f"""
        <div class="term-card">
            <div class="term-name">{term}</div>
            <div class="term-def">{definition}</div>
        </div>
        """, unsafe_allow_html=True)

    if not filtered_terms:
        st.info("No matching terms found.")


# ════════════════════════════════════════
# PAGE: EMAIL DRAFTING
# ════════════════════════════════════════
elif page == "✉️ Email Drafting":
    st.markdown('<div class="section-header">Email Drafting</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">AI-researched, personalized outreach — single or batch</div>', unsafe_allow_html=True)

    if 'scored_df' not in st.session_state:
        st.info("Run the pipeline first to enable email drafting.")
    else:
        scored_df = st.session_state['scored_df']

        # Sender details (shared across both modes)
        st.markdown("<div style='font-family:DM Mono,monospace;font-size:11px;color:#6b7280;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:12px;'>Sender Details</div>", unsafe_allow_html=True)
        e1, e2, e3 = st.columns(3)
        sender_name  = e1.text_input("Your name",  placeholder="e.g. Adrian Darmali")
        sender_title = e2.text_input("Your title", placeholder="e.g. Analyst, PaceZero Capital Partners")
        tone         = e3.selectbox("Tone", ["Professional", "Warm and conversational", "Direct and concise"])

        st.markdown("<hr>", unsafe_allow_html=True)
        mode = st.radio("Mode", ["Single org", "Batch"], horizontal=True)

        # ── SINGLE ──
        if mode == "Single org":
            selected = st.selectbox("Select a prospect", options=scored_df['Organization'].tolist())
            if selected:
                row = scored_df[scored_df['Organization'] == selected].iloc[0]
                st.markdown(f"""
                <div style='background:#f0f4ff;border:1px solid #c7d2fe;border-radius:8px;padding:12px 18px;margin-bottom:16px;font-size:13px;'>
                    <strong>{row['Organization']}</strong> &nbsp;·&nbsp; {row['Type']} &nbsp;·&nbsp;
                    Status: {row['Status']} &nbsp;·&nbsp; Composite {row['Composite']} &nbsp;·&nbsp; {row['Tier']}
                </div>
                """, unsafe_allow_html=True)

                if st.button("Generate Draft", type="primary"):
                    with st.spinner("Step 1 of 2 — Researching recent activity..."):
                        try:
                            draft, recent_activity, tok, cost_e = generate_draft(row, sender_name, sender_title, tone)

                            st.markdown("<div style='font-family:DM Mono,monospace;font-size:10px;color:#6b7280;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:8px;margin-top:16px;'>Recent Activity Found</div>", unsafe_allow_html=True)
                            st.info(recent_activity)
                            st.markdown("<div style='font-family:DM Mono,monospace;font-size:10px;color:#6b7280;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:8px;margin-top:8px;'>Generated Draft</div>", unsafe_allow_html=True)
                            st.text_area("", value=draft, height=300, key="single_draft_out")
                            st.caption(f"2 API calls  |  Tokens: {tok:,}  |  Cost: ${cost_e:.4f}")
                            st.link_button("Open in Gmail", gmail_link(draft))
                            st.markdown(DEMO_LIMIT_NOTE, unsafe_allow_html=True)

                        except Exception as e:
                            st.error(f"Draft generation failed: {e}")

        # ── BATCH ──
        else:
            st.markdown("<div style='font-family:DM Mono,monospace;font-size:11px;color:#6b7280;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:12px;'>Filter prospects</div>", unsafe_allow_html=True)
            bf1, bf2 = st.columns(2)
            batch_tier_f   = bf1.multiselect("Tier",   ['PRIORITY CLOSE','STRONG FIT','MODERATE FIT','WEAK FIT'], default=['PRIORITY CLOSE','STRONG FIT'])
            batch_status_f = bf2.multiselect("Status", CONTACT_STATUSES, default=CONTACT_STATUSES)

            batch_pool = scored_df[
                scored_df['Tier'].isin(batch_tier_f) &
                scored_df['Status'].isin(batch_status_f)
            ]

            selected_orgs = st.multiselect(
                f"Select orgs to draft for ({len(batch_pool)} match filters)",
                options=batch_pool['Organization'].tolist(),
                default=batch_pool['Organization'].tolist()[:5],
            )

            if selected_orgs:
                est_cost = len(selected_orgs) * 0.015
                st.caption(f"{len(selected_orgs)} prospects selected  |  Est. cost: ~${est_cost:.3f}  |  ~{len(selected_orgs)*2} API calls")

                if st.button(f"Generate {len(selected_orgs)} Drafts", type="primary"):
                    batch_progress  = st.progress(0)
                    batch_status_ph = st.empty()
                    batch_results   = []
                    total_tok = 0
                    total_cost_batch = 0.0

                    for i, org in enumerate(selected_orgs):
                        batch_status_ph.caption(f"Drafting {i+1}/{len(selected_orgs)} — {org}")
                        row = scored_df[scored_df['Organization'] == org].iloc[0]
                        try:
                            draft, recent_activity, tok, cost_e = generate_draft(row, sender_name, sender_title, tone)
                            total_tok        += tok
                            total_cost_batch += cost_e
                            batch_results.append({'org': org, 'row': row, 'draft': draft, 'recent': recent_activity, 'tok': tok, 'error': None})
                        except Exception as e:
                            batch_results.append({'org': org, 'row': row, 'draft': None, 'recent': None, 'tok': 0, 'error': str(e)})
                        batch_progress.progress((i+1)/len(selected_orgs))

                    batch_status_ph.caption(f"Done — {len(batch_results)} drafts generated  |  Tokens: {total_tok:,}  |  Cost: ${total_cost_batch:.4f}")
                    st.markdown(BATCH_LIMIT_NOTE, unsafe_allow_html=True)

                    for res in batch_results:
                        tier_label = res['row']['Tier']
                        status_label = res['row']['Status']
                        with st.expander(f"{res['org']}   ·   {tier_label}   ·   {status_label}"):
                            if res['error']:
                                st.error(f"Failed: {res['error']}")
                            else:
                                st.markdown("<div style='font-family:DM Mono,monospace;font-size:10px;color:#6b7280;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:6px;'>Recent Activity</div>", unsafe_allow_html=True)
                                st.info(res['recent'])
                                st.markdown("<div style='font-family:DM Mono,monospace;font-size:10px;color:#6b7280;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:6px;margin-top:12px;'>Draft</div>", unsafe_allow_html=True)
                                st.text_area("", value=res['draft'], height=260, key=f"draft_{res['org']}")
                                col_a, col_b = st.columns([1,5])
                                col_a.caption(f"Tokens: {res['tok']:,}")
                                with col_b:
                                    st.link_button("Open in Gmail", gmail_link(res['draft']))
