"""
Recto — Dashboard (v3 "Awwwards" redesign)

Design language: Vercel × Linear × Stripe
- Dark obsidian with warm amber accent
- Premium typography (Space Grotesk + IBM Plex Mono)
- Glassmorphism cards with animated hover states
- Animated hero metrics with counters
- Expandable candidate detail panels
- Natural-language reasoning prominently displayed
"""

import streamlit as st
import pandas as pd
import os
import re

st.set_page_config(
    page_title="Recto · Candidate Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

/* ── Tokens ── */
:root {
    --amber:      #e8b84b;
    --amber-dim:  #c49a30;
    --amber-glow: rgba(232, 184, 75, 0.15);
    --surface-0:  #0a0a0a;
    --surface-1:  rgba(18, 18, 18, 0.65);
    --surface-2:  rgba(28, 28, 28, 0.5);
    --surface-3:  #1a1a1a;
    --border:     rgba(255, 255, 255, 0.06);
    --border-hi:  rgba(255, 255, 255, 0.12);
    --text-0:     #f5f5f5;
    --text-1:     #a0a0a0;
    --text-2:     #555555;
    --green:      #34d399;
    --red:        #f87171;
    --blue:       #60a5fa;
    --purple:     #a78bfa;
    --mono: 'IBM Plex Mono', monospace;
    --sans: 'Space Grotesk', sans-serif;
    --radius: 16px;
    --transition: all 0.35s cubic-bezier(0.25, 0.8, 0.25, 1);
}

/* ── Base ── */
html, body, [class*="css"] { font-family: var(--sans) !important; }
.stApp {
    background: var(--surface-0) !important;
    background-image:
        radial-gradient(ellipse at 10% 20%, rgba(232, 184, 75, 0.04) 0%, transparent 50%),
        radial-gradient(ellipse at 90% 80%, rgba(96, 165, 250, 0.03) 0%, transparent 50%) !important;
    background-attachment: fixed !important;
}
#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] { display: none; }
.block-container { padding: 0 !important; max-width: 100% !important; }
* { box-sizing: border-box; }

/* ── Animated noise grain overlay ── */
.stApp::before {
    content: "";
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 200 200' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.65' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none;
    z-index: 0;
}

/* ── Top bar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 3rem;
    height: 56px;
    border-bottom: 1px solid var(--border);
    background: rgba(10, 10, 10, 0.85);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    position: sticky;
    top: 0;
    z-index: 100;
}
.topbar-logo {
    font-family: var(--mono);
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-0);
    letter-spacing: 0.08em;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.topbar-logo .accent { color: var(--amber); }
.topbar-pills {
    display: flex;
    gap: 6px;
}
.topbar-pill {
    font-family: var(--mono);
    font-size: 0.55rem;
    color: var(--text-2);
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 3px 10px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    transition: var(--transition);
}
.topbar-pill:hover {
    border-color: var(--amber);
    color: var(--amber);
}

/* ── Hero Section ── */
.hero {
    padding: 3rem 3rem 2rem;
    border-bottom: 1px solid var(--border);
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: var(--text-0);
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 0.5rem;
}
.hero-title .highlight {
    background: linear-gradient(135deg, var(--amber), #f0c674);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-subtitle {
    font-size: 0.85rem;
    color: var(--text-1);
    max-width: 600px;
    line-height: 1.5;
}
.hero-metrics {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1px;
    margin-top: 2rem;
    background: var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    border: 1px solid var(--border);
}
.hero-metric {
    background: var(--surface-0);
    padding: 1.25rem 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}
.hero-metric:first-child { border-radius: var(--radius) 0 0 var(--radius); }
.hero-metric:last-child { border-radius: 0 var(--radius) var(--radius) 0; }
.hm-value {
    font-family: var(--mono);
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--text-0);
    line-height: 1;
}
.hm-value.amber { color: var(--amber); }
.hm-label {
    font-size: 0.65rem;
    color: var(--text-2);
    letter-spacing: 0.04em;
}

/* ── Controls bar ── */
.controls-bar {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1rem 3rem;
    border-bottom: 1px solid var(--border);
    background: rgba(10, 10, 10, 0.5);
    backdrop-filter: blur(12px);
    position: sticky;
    top: 56px;
    z-index: 99;
}

/* ── Candidate cards ── */
.candidates-grid {
    padding: 1.5rem 3rem 3rem;
    display: flex;
    flex-direction: column;
    gap: 12px;
}
.c-card {
    display: grid;
    grid-template-columns: 60px 1fr 240px 80px;
    align-items: start;
    padding: 1.25rem 1.5rem;
    background: var(--surface-1);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    cursor: default;
    transition: var(--transition);
    gap: 1.25rem;
}
.c-card:hover {
    border-color: var(--border-hi);
    transform: translateY(-2px);
    box-shadow: 0 16px 48px -12px rgba(0, 0, 0, 0.6);
}
.c-card.gold-border {
    border-color: rgba(232, 184, 75, 0.25);
    background: linear-gradient(135deg, rgba(232, 184, 75, 0.04), transparent 60%);
}
.c-card.gold-border:hover {
    border-color: rgba(232, 184, 75, 0.5);
    box-shadow: 0 16px 48px -12px rgba(232, 184, 75, 0.15);
}

/* ── Rank column ── */
.c-rank {
    font-family: var(--mono);
    font-size: 1.4rem;
    font-weight: 700;
    color: var(--text-2);
    line-height: 1;
    text-align: center;
    padding-top: 4px;
}
.c-rank.gold { color: var(--amber); text-shadow: 0 0 20px rgba(232, 184, 75, 0.3); }
.c-rank.silver { color: #b0b0b0; }
.c-rank.bronze { color: #cd7c2f; }

/* ── Main body column ── */
.c-name {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-0);
    letter-spacing: 0.01em;
    margin-bottom: 2px;
}
.c-meta {
    font-family: var(--mono);
    font-size: 0.6rem;
    color: var(--amber-dim);
    letter-spacing: 0.04em;
    margin-bottom: 0.65rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.c-reasoning {
    font-size: 0.8rem;
    color: var(--text-1);
    line-height: 1.6;
    margin-bottom: 0.6rem;
}
.c-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
}
.tag {
    font-family: var(--mono);
    font-size: 0.58rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    padding: 3px 8px;
    border-radius: 100px;
    border: 1px solid;
    transition: var(--transition);
}
.tag:hover { transform: scale(1.05); }
.tag-green  { color: var(--green);  border-color: rgba(52, 211, 153, 0.25); background: rgba(52, 211, 153, 0.08); }
.tag-amber  { color: var(--amber);  border-color: rgba(232, 184, 75, 0.25); background: rgba(232, 184, 75, 0.08); }
.tag-blue   { color: var(--blue);   border-color: rgba(96, 165, 250, 0.25); background: rgba(96, 165, 250, 0.08); }
.tag-purple { color: var(--purple); border-color: rgba(167, 139, 250, 0.25); background: rgba(167, 139, 250, 0.08); }
.tag-muted  { color: var(--text-2); border-color: var(--border); background: transparent; }

/* ── Key strengths column ── */
.c-strengths {
    display: flex;
    flex-direction: column;
    gap: 4px;
    padding-top: 2px;
}
.strength-item {
    font-size: 0.68rem;
    color: var(--text-1);
    display: flex;
    align-items: center;
    gap: 6px;
}
.strength-dot {
    width: 4px;
    height: 4px;
    border-radius: 50%;
    background: var(--amber);
    flex-shrink: 0;
}

/* ── Score ring column ── */
.c-score-col {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-width: 70px;
}
.circular-chart {
    display: block;
    margin: 0 auto;
    max-width: 56px;
    max-height: 56px;
}
.circle-bg {
    fill: none;
    stroke: var(--surface-3);
    stroke-width: 2.4;
}
.circle {
    fill: none;
    stroke-width: 2.4;
    stroke-linecap: round;
    animation: progress 1.2s ease-out forwards;
    transform-origin: center;
}
@keyframes progress {
    0% { stroke-dasharray: 0 100; }
}
.percentage {
    fill: var(--text-0);
    font-family: var(--mono);
    font-size: 0.42rem;
    font-weight: 700;
    text-anchor: middle;
}
.match-label {
    font-size: 0.5rem;
    font-family: var(--mono);
    color: var(--text-2);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 5px;
    text-align: center;
}

/* ── Footer ── */
.app-footer {
    border-top: 1px solid var(--border);
    padding: 2rem 3rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.footer-left {
    font-family: var(--mono);
    font-size: 0.6rem;
    color: var(--text-2);
    letter-spacing: 0.04em;
}
.footer-right {
    font-family: var(--mono);
    font-size: 0.6rem;
    color: var(--text-2);
}
.footer-right a {
    color: var(--amber);
    text-decoration: none;
}

/* ── Sidebar / Score Distribution ── */
.sidebar-section {
    padding: 1.5rem;
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--surface-1);
    backdrop-filter: blur(12px);
    margin-bottom: 1rem;
}
.sidebar-label {
    font-family: var(--mono);
    font-size: 0.58rem;
    letter-spacing: 0.12em;
    color: var(--text-2);
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.dist-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}
.dist-label {
    font-family: var(--mono);
    font-size: 0.6rem;
    color: var(--text-2);
    width: 48px;
    text-align: right;
    flex-shrink: 0;
}
.dist-bar-wrap {
    flex: 1;
    height: 4px;
    background: var(--surface-3);
    border-radius: 2px;
    overflow: hidden;
}
.dist-bar {
    height: 100%;
    border-radius: 2px;
    background: linear-gradient(90deg, var(--amber-dim), var(--amber));
    transition: width 0.6s ease-out;
}
.dist-count {
    font-family: var(--mono);
    font-size: 0.58rem;
    color: var(--text-2);
    width: 22px;
    flex-shrink: 0;
}

/* ── Pipeline viz ── */
.pipeline-flow {
    display: flex;
    flex-direction: column;
    gap: 0;
}
.pipe-step {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    padding: 0.5rem 0;
    position: relative;
}
.pipe-step:not(:last-child)::after {
    content: '';
    position: absolute;
    left: 5px;
    top: 24px;
    bottom: -4px;
    width: 1px;
    background: var(--border-hi);
}
.pipe-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    border: 2px solid var(--amber);
    background: var(--surface-0);
    flex-shrink: 0;
    margin-top: 2px;
    position: relative;
    z-index: 1;
}
.pipe-dot.active {
    background: var(--amber);
    box-shadow: 0 0 8px rgba(232, 184, 75, 0.4);
}
.pipe-text { }
.pipe-title {
    font-size: 0.72rem;
    color: var(--text-0);
    font-weight: 500;
}
.pipe-desc {
    font-size: 0.6rem;
    color: var(--text-2);
    margin-top: 1px;
}

/* ── Streamlit overrides ── */
[data-testid="InputInstructions"] { display: none !important; }
.stTextInput input {
    background: var(--surface-2) url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="%23a0a0a0" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>') no-repeat 14px center !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: 100px !important;
    color: var(--text-0) !important;
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    padding: 0.5rem 1rem 0.5rem 2.4rem !important;
    transition: var(--transition) !important;
}
.stTextInput input:focus {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 3px rgba(232, 184, 75, 0.1) !important;
}
[data-testid="stDownloadButton"] button {
    background: transparent !important;
    border: 1px solid var(--border-hi) !important;
    color: var(--text-1) !important;
    font-family: var(--mono) !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.06em !important;
    border-radius: 100px !important;
    padding: 6px 16px !important;
    transition: var(--transition) !important;
}
[data-testid="stDownloadButton"] button:hover {
    border-color: var(--amber) !important;
    color: var(--amber) !important;
    box-shadow: 0 0 16px rgba(232, 184, 75, 0.1) !important;
}
[data-testid="stSelectbox"] > div > div {
    background: var(--surface-2) !important;
    border-color: var(--border-hi) !important;
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    base_path = "results/final_ranking.csv"
    rich_path = "results/shortlist_top100.csv"
    if os.path.exists(base_path) and os.path.exists(rich_path):
        df_base = pd.read_csv(base_path)
        df_rich = pd.read_csv(rich_path)
        if 'candidate_name' in df_rich.columns:
            df_rich = df_rich.rename(columns={'candidate_name': 'name'})
        merged = df_base.merge(
            df_rich[['rank', 'name', 'github_stars', 'ir_roles_count', 'key_strengths']],
            on='rank', how='left'
        )
        return merged
    elif os.path.exists(base_path):
        return pd.read_csv(base_path)
    return pd.DataFrame()

def extract_signals(row):
    """Extract structured signal badges from candidate data."""
    signals = []
    reasoning = str(row.get("reasoning", ""))
    
    # Open to work
    if "open to work" in reasoning.lower() or "currently marked open" in reasoning.lower():
        signals.append(("OPEN", "tag-green"))
    
    # FAANG / top company
    if re.search(r'Google|Meta|Amazon|Netflix|Microsoft|LinkedIn|OpenAI|DeepMind|Apple', reasoning, re.I):
        signals.append(("FAANG+", "tag-amber"))
    
    # Tier-1 university
    if re.search(r'IIT |MIT|Stanford|Carnegie|Georgia Tech|BITS|Caltech|IISc|IIIT', reasoning):
        signals.append(("TIER-1", "tag-blue"))
    
    # IR roles
    ir_count = row.get('ir_roles_count', 0)
    if pd.notna(ir_count) and int(ir_count) >= 2:
        cls = "tag-green" if int(ir_count) >= 3 else "tag-amber"
        signals.append((f"{int(ir_count)} IR", cls))
    
    # Advanced IR
    if re.search(r'learning to rank|colbert|cross-encoder|rerank|dense retrieval', reasoning, re.I):
        signals.append(("ADVANCED IR", "tag-purple"))
    
    # GitHub
    gh = row.get('github_stars', -1)
    if pd.notna(gh) and float(gh) >= 50:
        signals.append((f"GH {int(gh)}", "tag-blue"))
    
    # Notice period
    m = re.search(r'(\d+)-day notice', reasoning)
    if m:
        n = int(m.group(1))
        cls = "tag-green" if n <= 30 else "tag-muted"
        signals.append((f"{n}D", cls))
    
    return signals[:6]


df = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# TOP BAR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-logo">
        <span class="accent">◈</span> RECTO
        <span style="color:var(--border);margin:0 4px;">·</span>
        <span style="color:var(--text-2);font-weight:400;font-size:0.65rem;letter-spacing:0.02em;">Candidate Intelligence</span>
    </div>
    <div class="topbar-pills">
        <div class="topbar-pill">ZERO LLM</div>
        <div class="topbar-pill">CPU-ONLY</div>
        <div class="topbar-pill">DETERMINISTIC</div>
        <div class="topbar-pill">< 3 MIN</div>
    </div>
</div>
""", unsafe_allow_html=True)

if df.empty:
    st.markdown("""
    <div style="padding:6rem;text-align:center;color:var(--text-2);font-family:var(--mono);font-size:0.8rem;">
        NO DATA — run <span style="color:var(--amber)">python main.py --candidates &lt;path&gt;</span> first
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# HERO SECTION
# ─────────────────────────────────────────────────────────────────────────────
hero_placeholder = st.empty()

# ─────────────────────────────────────────────────────────────────────────────
# CONTROLS
# ─────────────────────────────────────────────────────────────────────────────
ctrl_left, ctrl_mid, ctrl_f1, ctrl_f2, ctrl_f3, ctrl_f4, ctrl_dl = st.columns([2.0, 2.0, 1.3, 1.2, 1.4, 1.2, 0.7])
with ctrl_left:
    search = st.text_input("", placeholder="Search name, ID, or keyword…", label_visibility="collapsed")
with ctrl_mid:
    top_n = st.slider("Show Top Candidates", min_value=1, max_value=100, value=min(100, len(df)) if len(df) > 0 else 100)
with ctrl_f1:
    fast_notice = st.checkbox("Immediate (≤15d)", value=False)
with ctrl_f2:
    tier1_only = st.checkbox("Tier-1 Tech", value=False)
with ctrl_f3:
    deep_ir = st.checkbox("Deep IR (2+ roles)", value=False)
with ctrl_f4:
    high_resp = st.checkbox("High Response", value=False)
with ctrl_dl:
    st.download_button("↓ CSV", data=df.to_csv(index=False).encode("utf-8"),
                       file_name="recto_ranking.csv", mime="text/csv")

# Apply filters
filtered = df.copy()

if search:
    filtered = filtered[
        filtered['name'].str.contains(search, case=False, na=False) |
        filtered['candidate_id'].str.contains(search, case=False, na=False) |
        filtered['reasoning'].str.contains(search, case=False, na=False)
    ]

if tier1_only:
    tier1_pattern = r'(?i)\b(Google|Meta|Apple|Netflix|Microsoft|Amazon|LinkedIn|OpenAI|DeepMind|Salesforce|Uber|Airbnb|Stripe|Flipkart|Zomato|Swiggy|Razorpay)\b'
    filtered = filtered[filtered['reasoning'].str.contains(tier1_pattern, na=False)]

if fast_notice:
    filtered = filtered[filtered['reasoning'].str.contains(r'15-day notice|immediate', case=False, na=False)]

if deep_ir:
    deep_ir_pattern = r'(?i)\b([2-9] ir-relevant|[2-9] search-focused|[2-9] roles involving search|deep ir expertise)\b'
    filtered = filtered[filtered['reasoning'].str.contains(deep_ir_pattern, na=False)]

if high_resp:
    filtered = filtered[filtered['reasoning'].str.contains(r'(?i)highly responsive', na=False)]

# Apply Top N filter last
filtered = filtered.head(top_n)

# Update Hero Metrics dynamically
top_score = round(filtered['score'].max() * 100, 1) if not filtered.empty else 0.0
avg_score = round(filtered['score'].mean() * 100, 1) if not filtered.empty else 0.0
above_60 = int((filtered['score'] >= 0.6).sum())
ir_heavy = int(filtered['ir_roles_count'].fillna(0).ge(2).sum()) if 'ir_roles_count' in filtered.columns else '—'

hero_placeholder.markdown(f"""
<div class="hero">
    <div class="hero-title">Top 100 candidates, <span class="highlight">ranked.</span></div>
    <div class="hero-subtitle">
        Deterministic 5-layer pipeline scores 100K candidates in under 3 minutes.
        Zero LLMs. Zero API calls. Pure heuristic engineering.
    </div>
    <div class="hero-metrics">
        <div class="hero-metric">
            <div class="hm-value amber">{len(filtered)}</div>
            <div class="hm-label">Candidates Ranked</div>
        </div>
        <div class="hero-metric">
            <div class="hm-value">{top_score}%</div>
            <div class="hm-label">Top Match Score</div>
        </div>
        <div class="hero-metric">
            <div class="hm-value">{avg_score}%</div>
            <div class="hm-label">Average Score</div>
        </div>
        <div class="hero-metric">
            <div class="hm-value">{above_60}</div>
            <div class="hm-label">Score ≥ 60%</div>
        </div>
        <div class="hero-metric">
            <div class="hm-value">{ir_heavy}</div>
            <div class="hm-label">Deep IR (2+ roles)</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 2-COLUMN: Candidates + Sidebar
# ─────────────────────────────────────────────────────────────────────────────
col_main, col_side = st.columns([4.5, 1.5], gap="small")

# ── MAIN: Candidate Cards ─────────────────────────────────────────────────────
with col_main:
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;padding:0.75rem 0;margin-bottom:0.5rem;border-bottom:1px solid var(--border);">
        <span style="font-size:0.75rem;font-weight:600;color:var(--text-0);">Ranked Candidates</span>
        <span style="font-family:var(--mono);font-size:0.62rem;color:var(--text-2);">
            showing {len(filtered)} of {len(df)}
        </span>
    </div>
    """, unsafe_allow_html=True)

    for _, row in filtered.iterrows():
        rank = int(row["rank"])
        score_pct = round(row["score"] * 100, 1)
        signals = extract_signals(row)
        reasoning = str(row.get("reasoning", "—"))

        # Rank styling
        if rank == 1:
            rank_cls, card_cls = "gold", "gold-border"
            circle_color = "var(--amber)"
        elif rank == 2:
            rank_cls, card_cls = "silver", "gold-border"
            circle_color = "#b0b0b0"
        elif rank == 3:
            rank_cls, card_cls = "bronze", "gold-border"
            circle_color = "#cd7c2f"
        else:
            rank_cls, card_cls = "", ""
            circle_color = "var(--amber)"

        tags_html = " ".join(
            f'<span class="tag {cls}">{lbl}</span>'
            for lbl, cls in signals
        )

        c_name = row.get('name', 'Unknown')
        c_id = row['candidate_id']
        
        # Key strengths
        strengths = str(row.get('key_strengths', ''))
        strengths_items = [s.strip() for s in strengths.split(';') if s.strip() and s.strip() != 'N/A']
        strengths_html = "".join(
            f'<div class="strength-item"><div class="strength-dot"></div>{s}</div>'
            for s in strengths_items[:4]
        )

        st.markdown(f"""
        <div class="c-card {card_cls}">
            <div class="c-rank {rank_cls}">{rank:02d}</div>
            <div>
                <div class="c-name">{c_name}</div>
                <div class="c-meta">
                    <span>{c_id}</span>
                    <span style="color:var(--border);">·</span>
                    <span>Rank #{rank}</span>
                </div>
                <div class="c-reasoning">{reasoning}</div>
                <div class="c-tags">{tags_html}</div>
            </div>
            <div class="c-strengths">{strengths_html}</div>
            <div class="c-score-col">
                <svg viewBox="0 0 36 36" class="circular-chart">
                    <path class="circle-bg"
                        d="M18 2.0845
                        a 15.9155 15.9155 0 0 1 0 31.831
                        a 15.9155 15.9155 0 0 1 0 -31.831"
                    />
                    <path class="circle"
                        stroke-dasharray="{score_pct}, 100"
                        stroke="{circle_color}"
                        d="M18 2.0845
                        a 15.9155 15.9155 0 0 1 0 31.831
                        a 15.9155 15.9155 0 0 1 0 -31.831"
                    />
                    <text x="18" y="20.35" class="percentage">{score_pct}%</text>
                </svg>
                <div class="match-label">MATCH</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with col_side:
    st.markdown("<br>", unsafe_allow_html=True)

    # Score distribution
    buckets = [(0, 0.2, "0–20"), (0.2, 0.4, "20–40"), (0.4, 0.6, "40–60"),
               (0.6, 0.8, "60–80"), (0.8, 1.01, "80–100")]
    max_count = max(int(((df['score'] >= lo) & (df['score'] < hi)).sum())
                    for lo, hi, _ in buckets) or 1

    dist_html = '<div class="sidebar-section"><div class="sidebar-label">Score Distribution</div>'
    for lo, hi, lbl in reversed(buckets):
        count = int(((df['score'] >= lo) & (df['score'] < hi)).sum())
        bar_pct = int(count / max_count * 100)
        dist_html += f"""<div class="dist-row">
    <div class="dist-label">{lbl}%</div>
    <div class="dist-bar-wrap">
        <div class="dist-bar" style="width:{bar_pct}%;"></div>
    </div>
    <div class="dist-count">{count}</div>
</div>"""
    dist_html += '</div>'
    st.markdown(dist_html, unsafe_allow_html=True)

    # Load dynamic stats if available
    import json
    import os
    stats_path = "results/pipeline_stats.json"
    if os.path.exists(stats_path):
        try:
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            total = stats.get('initial_count', '100K')
            valid = stats.get('final_count', '30K')
            noise = stats.get('removed_noise', '40K')
            traps = stats.get('removed_salary_inversion', 0) + stats.get('removed_honeypot', 0)
            desc1 = f"Schema flatten, {total} → {valid} valid"
            desc2 = f"Salary traps ({traps}), noise ({noise}) dropped"
        except Exception:
            desc1, desc2 = "Schema flatten, 100K → 30K valid", "Salary traps, ghost profiles, honeypots"
    else:
        desc1, desc2 = "Schema flatten, 100K → 30K valid", "Salary traps, ghost profiles, honeypots"

    # Pipeline visualization
    pipe_html = '<div class="sidebar-section"><div class="sidebar-label">Pipeline Flow</div><div class="pipeline-flow">'
    for title, desc, active in [
        ("Ingest & Filter", desc1, True),
        ("Hard Kill", desc2, True),
        ("IR Depth Score", "Career parsing, rare skill matching", True),
        ("Semantic Boost", "TF-IDF cosine similarity", True),
        ("Deterministic Sort", "Zero inversions, tie-break by ID", True),
    ]:
        dot_cls = "pipe-dot active" if active else "pipe-dot"
        pipe_html += f"""<div class="pipe-step">
    <div class="{dot_cls}"></div>
    <div class="pipe-text">
        <div class="pipe-title">{title}</div>
        <div class="pipe-desc">{desc}</div>
    </div>
</div>"""
    pipe_html += '</div></div>'
    st.markdown(pipe_html, unsafe_allow_html=True)

    # Top 5 leaderboard
    ldr_html = '<div class="sidebar-section"><div class="sidebar-label">Top 5 Leaderboard</div>'
    for _, row in df.head(5).iterrows():
        r = int(row["rank"])
        sp = round(row["score"] * 100, 1)
        nm = str(row.get("name", row["candidate_id"]))[:20]
        accent = ' style="color:var(--amber);"' if r == 1 else ""
        ldr_html += f"""<div style="display:flex;align-items:center;gap:0.5rem;padding:0.4rem 0;border-bottom:1px solid var(--border);">
    <span style="font-family:var(--mono);font-size:0.7rem;color:var(--text-2);width:18px;">#{r}</span>
    <span style="font-size:0.7rem;color:var(--text-1);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{nm}</span>
    <span style="font-family:var(--mono);font-size:0.7rem;font-weight:600;"{accent}>{sp}%</span>
</div>"""
    ldr_html += '</div>'
    st.markdown(ldr_html, unsafe_allow_html=True)

    # Team
    st.markdown("""
    <div class="sidebar-section">
        <div class="sidebar-label">Team</div>
        <div style="font-size:0.72rem;color:var(--text-1);line-height:1.8;">
            <div>Aryan Bhargava</div>
            <div>Harshit Kudhial</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    <div class="footer-left">◈ RECTO · Deterministic Candidate Intelligence</div>
    <div class="footer-right">
        Built by Team Recto · <a href="https://github.com/harshitnub077/Recto" target="_blank">GitHub</a>
    </div>
</div>
""", unsafe_allow_html=True)
