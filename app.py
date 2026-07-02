"""
Recto — Dashboard (v2 "Editorial" redesign)

Design language: Ashby × Linear × Bloomberg Terminal
- Warm amber accent on near-black
- Editorial monospace rank numbers
- Dense, data-rich rows — not cards
- No purple, no generic AI palette
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
# DESIGN SYSTEM — Amber/Charcoal editorial theme (NOT purple, NOT generic AI)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Tokens ── */
:root {
    --amber:      #e8b84b;
    --amber-dim:  #c49a30;
    --amber-faint:#e8b84b18;
    --surface-0:  #080808;
    --surface-1:  #0f0f0f;
    --surface-2:  #161616;
    --surface-3:  #1e1e1e;
    --border:     #232323;
    --border-hi:  #2e2e2e;
    --text-0:     #f2f2f2;
    --text-1:     #a0a0a0;
    --text-2:     #555555;
    --green:      #4ade80;
    --red:        #f87171;
    --blue:       #60a5fa;
    --mono: 'IBM Plex Mono', monospace;
    --sans: 'Inter', sans-serif;
}

/* ── Base ── */
html, body, [class*="css"] { font-family: var(--sans) !important; }
.stApp { background: var(--surface-0) !important; }
#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] { display: none; }
.block-container { padding: 0 !important; max-width: 100% !important; }
* { box-sizing: border-box; }

/* ── Top bar ── */
.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 2.5rem;
    height: 52px;
    border-bottom: 1px solid var(--border);
    background: var(--surface-0);
    position: sticky;
    top: 0;
    z-index: 100;
}
.topbar-logo {
    font-family: var(--mono);
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--text-0);
    letter-spacing: 0.06em;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.topbar-logo span { color: var(--amber); }
.topbar-pill {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--text-2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 3px 8px;
    letter-spacing: 0.05em;
}

/* ── Page body ── */
.page-body {
    display: grid;
    grid-template-columns: 240px 1fr 300px;
    min-height: calc(100vh - 52px);
}

/* ── Left panel ── */
.left-panel {
    border-right: 1px solid var(--border);
    padding: 2rem 1.5rem;
    background: var(--surface-0);
}
.panel-label {
    font-family: var(--mono);
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    color: var(--text-2);
    text-transform: uppercase;
    margin-bottom: 0.75rem;
}
.stat-row {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
    margin-bottom: 2rem;
}
.stat-item {}
.stat-value {
    font-family: var(--mono);
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text-0);
    line-height: 1;
}
.stat-value.amber { color: var(--amber); }
.stat-label {
    font-size: 0.7rem;
    color: var(--text-2);
    margin-top: 0.2rem;
}
.divider { border: none; border-top: 1px solid var(--border); margin: 1.5rem 0; }
.pipeline-step {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    margin-bottom: 1rem;
    font-size: 0.72rem;
}
.step-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: var(--amber);
    margin-top: 4px;
    flex-shrink: 0;
}
.step-title { color: var(--text-0); font-weight: 500; }
.step-desc { color: var(--text-2); font-size: 0.65rem; margin-top: 1px; }

/* ── Center — candidate list ── */
.center-panel {
    border-right: 1px solid var(--border);
    overflow: hidden;
}
.list-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.25rem 2rem;
    border-bottom: 1px solid var(--border);
    background: var(--surface-0);
    position: sticky;
    top: 52px;
    z-index: 99;
}
.list-title {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-0);
    letter-spacing: 0.02em;
}
.list-meta { font-size: 0.7rem; color: var(--text-2); font-family: var(--mono); }

/* ── Candidate row ── */
.c-row {
    display: grid;
    grid-template-columns: 56px 1fr auto;
    align-items: start;
    padding: 1.1rem 2rem;
    border-bottom: 1px solid var(--border);
    cursor: default;
    transition: background 0.12s;
    gap: 1rem;
}
.c-row:hover { background: var(--surface-1); }
.c-row.top3 { background: var(--amber-faint); }
.c-row.top3:hover { background: #e8b84b22; }

.c-rank {
    font-family: var(--mono);
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text-2);
    line-height: 1;
    padding-top: 2px;
}
.c-rank.gold { color: var(--amber); }
.c-rank.silver { color: #9ca3af; }
.c-rank.bronze { color: #cd7c2f; }

.c-body {}
.c-name {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-0);
    letter-spacing: 0.01em;
    margin-bottom: 0.15rem;
}
.c-id {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--amber-dim);
    letter-spacing: 0.04em;
    margin-bottom: 0.6rem;
}
.c-snippet {
    font-size: 0.82rem;
    color: var(--text-1);
    line-height: 1.55;
    max-width: 100%;
}
.c-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 0.75rem;
}
.tag {
    font-family: var(--mono);
    font-size: 0.62rem;
    font-weight: 500;
    letter-spacing: 0.04em;
    padding: 3px 8px;
    border-radius: 4px;
    border: 1px solid;
}
.tag-green { color: var(--green); border-color: #4ade8030; background: #4ade8010; }
.tag-amber { color: var(--amber); border-color: #e8b84b30; background: #e8b84b0e; }
.tag-blue  { color: var(--blue);  border-color: #60a5fa30; background: #60a5fa0e; }
.tag-muted { color: var(--text-2); border-color: var(--border); background: transparent; }

.c-score-col {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-width: 80px;
}
.circular-chart {
    display: block;
    margin: 0 auto;
    max-width: 54px;
    max-height: 54px;
}
.circle-bg {
    fill: none;
    stroke: var(--surface-3);
    stroke-width: 2.5;
}
.circle {
    fill: none;
    stroke-width: 2.5;
    stroke-linecap: round;
    animation: progress 1s ease-out forwards;
}
@keyframes progress {
    0% { stroke-dasharray: 0 100; }
}
.percentage {
    fill: var(--text-0);
    font-family: var(--mono);
    font-size: 0.45rem;
    font-weight: 600;
    text-anchor: middle;
}
.match-label {
    font-size: 0.55rem;
    font-family: var(--mono);
    color: var(--text-2);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 6px;
    text-align: center;
}

/* ── Right panel ── */
.right-panel {
    padding: 1.5rem;
    background: var(--surface-0);
}
.rp-section { margin-bottom: 2rem; }
.rp-label {
    font-family: var(--mono);
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    color: var(--text-2);
    text-transform: uppercase;
    margin-bottom: 1rem;
}
.score-dist-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}
.score-dist-label {
    font-family: var(--mono);
    font-size: 0.62rem;
    color: var(--text-2);
    width: 50px;
    text-align: right;
    flex-shrink: 0;
}
.score-dist-bar-wrap {
    flex: 1;
    height: 5px;
    background: var(--surface-3);
    border-radius: 2px;
    overflow: hidden;
}
.score-dist-bar { height: 100%; background: var(--amber); border-radius: 2px; }
.score-dist-count {
    font-family: var(--mono);
    font-size: 0.6rem;
    color: var(--text-2);
    width: 24px;
    flex-shrink: 0;
}

/* ── Top 5 leader ── */
.leader-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.6rem 0;
    border-bottom: 1px solid var(--border);
}
.leader-rank {
    font-family: var(--mono);
    font-size: 0.75rem;
    color: var(--text-2);
    width: 20px;
    flex-shrink: 0;
}
.leader-id {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--text-1);
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.leader-score {
    font-family: var(--mono);
    font-size: 0.7rem;
    color: var(--amber);
    font-weight: 600;
}

/* ── Streamlit widget overrides ── */
.stTextInput input {
    background: var(--surface-2) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: 6px !important;
    color: var(--text-0) !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
    padding: 0.4rem 0.75rem !important;
}
.stTextInput input:focus {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 1px var(--amber) !important;
}
.stSlider [data-testid="stSlider"] { padding: 0 !important; }
[data-testid="stSlider"] div[role="slider"] {
    background: var(--amber) !important;
    border: 2px solid var(--surface-0) !important;
    width: 14px !important;
    height: 14px !important;
}
[data-testid="stSlider"] .stSlider > div > div > div:first-child {
    background: var(--amber) !important;
}
[data-testid="stDownloadButton"] button {
    background: transparent !important;
    border: 1px solid var(--border-hi) !important;
    color: var(--text-1) !important;
    font-family: var(--mono) !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.05em !important;
    border-radius: 4px !important;
    padding: 4px 12px !important;
    transition: all 0.15s !important;
}
[data-testid="stDownloadButton"] button:hover {
    border-color: var(--amber) !important;
    color: var(--amber) !important;
}
[data-testid="stSelectbox"] > div > div {
    background: var(--surface-2) !important;
    border-color: var(--border-hi) !important;
    font-family: var(--mono) !important;
    font-size: 0.75rem !important;
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
        # Merge on rank to get rich metadata
        merged = df_base.merge(df_rich[['rank', 'name', 'github_stars', 'ir_roles_count', 'key_strengths']], on='rank', how='left')
        return merged
    elif os.path.exists(base_path):
        return pd.read_csv(base_path)
    return pd.DataFrame()

def parse_signals(reasoning: str):
    """Extract structured signals from a reasoning string."""
    signals = []
    r = str(reasoning)
    if "OTW" in r:
        signals.append(("OPEN", "tag-green"))
    if re.search(r"Google|Meta|Amazon|Netflix|Microsoft|LinkedIn|OpenAI|DeepMind", r, re.I):
        signals.append(("FAANG", "tag-amber"))
    if re.search(r"IIT |MIT|Stanford|Carnegie|Georgia Tech|BITS|Caltech", r):
        signals.append(("TIER-1 UNI", "tag-blue"))
    m_roles = re.search(r"(\d+) IR roles", r)
    if m_roles:
        n = int(m_roles.group(1))
        cls = "tag-green" if n >= 3 else "tag-amber" if n >= 2 else "tag-muted"
        signals.append((f"{n} IR ROLES", cls))
    if re.search(r"Learning to Rank|LTR", r):
        signals.append(("LTR", "tag-amber"))
    if "rare IR" in r.lower():
        signals.append(("RARE IR", "tag-green"))
    m_gh = re.search(r"GitHub=(\d+)", r, re.I)
    if m_gh and int(m_gh.group(1)) > 60:
        signals.append((f"GH={m_gh.group(1)}", "tag-blue"))
    m_notice = re.search(r"notice=(\d+)d", r, re.I)
    if m_notice:
        n = int(m_notice.group(1))
        cls = "tag-green" if n <= 30 else "tag-muted"
        signals.append((f"{n}D NOTICE", cls))
    return signals[:5]

def snippet(text: str, max_len=1000) -> str:
    t = re.sub(r'\s+', ' ', str(text)).strip()
    return t[:max_len] + "…" if len(t) > max_len else t

df = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# TOP BAR
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-logo">
        <span>◈</span> RECTO
        <span style="color:var(--border);margin:0 4px;">|</span>
        <span style="color:var(--text-2);font-weight:400;font-size:0.7rem;">Candidate Intelligence</span>
    </div>
    <div style="display:flex;gap:0.5rem;">
        <div class="topbar-pill">ZERO LLM</div>
        <div class="topbar-pill">CPU-ONLY</div>
        <div class="topbar-pill">DETERMINISTIC</div>
    </div>
</div>
""", unsafe_allow_html=True)

if df.empty:
    st.markdown("""
    <div style="padding:4rem;text-align:center;color:var(--text-2);font-family:var(--mono);font-size:0.8rem;">
        NO DATA — run <span style="color:var(--amber)">python main.py --candidates &lt;path&gt;</span> first
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# CONTROLS (search + sort — minimal, above the 3-col layout)
# ─────────────────────────────────────────────────────────────────────────────
ctrl_left, ctrl_mid, ctrl_right, ctrl_dl = st.columns([3, 2, 2, 1])
with ctrl_left:
    search = st.text_input("", placeholder="Search by ID or keyword…", label_visibility="collapsed")
with ctrl_mid:
    top_n = st.select_slider("", options=[5, 10, 15, 20, 25, 50, 75, 100], value=25, label_visibility="collapsed")
with ctrl_right:
    min_score = st.select_slider("", options=[0, 20, 40, 50, 60, 70, 80], value=0, label_visibility="collapsed",
                                  format_func=lambda x: f"Score ≥ {x}%")
with ctrl_dl:
    st.download_button("↓ CSV", data=df.to_csv(index=False).encode("utf-8"),
                       file_name="recto_ranking.csv", mime="text/csv")

# Apply filters
filtered = df.copy()
filtered = filtered[filtered["score"] >= min_score / 100]
if search:
    mask = filtered.apply(lambda r: search.lower() in str(r.values).lower(), axis=1)
    filtered = filtered[mask]
filtered = filtered.head(top_n)

# ─────────────────────────────────────────────────────────────────────────────
# 3-COLUMN LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
col_left, col_center, col_right = st.columns([1, 3.2, 1.2], gap="small")

# ── LEFT PANEL ────────────────────────────────────────────────────────────────
with col_left:
    top_pct = round(df['score'].max() * 100, 1)
    avg_pct = round(df['score'].mean() * 100, 1)
    above_50 = int((df['score'] >= 0.5).sum())
    mono = "font-family:'IBM Plex Mono',monospace"
    muted = "font-size:0.65rem;color:#555;"

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f'<p style="{mono};font-size:0.6rem;letter-spacing:0.12em;color:#555;text-transform:uppercase;">OVERVIEW</p>', unsafe_allow_html=True)
    for val, label, color in [
        (len(df),    "Candidates ranked",  "#e8b84b"),
        (f"{top_pct}%", "Top hybrid score", "#f2f2f2"),
        (f"{avg_pct}%", "Average score",    "#f2f2f2"),
        (above_50,   "Score ≥ 50%",        "#f2f2f2"),
    ]:
        st.markdown(
            f'<p style="{mono};font-size:1.35rem;font-weight:600;color:{color};margin:0.25rem 0 0 0;line-height:1;">'
            f'{val}</p>'
            f'<p style="{muted};margin:2px 0 0.9rem 0;">{label}</p>',
            unsafe_allow_html=True
        )

    st.markdown('<hr style="border:none;border-top:1px solid #232323;margin:0 0 1rem 0;">', unsafe_allow_html=True)
    st.markdown(f'<p style="{mono};font-size:0.6rem;letter-spacing:0.12em;color:#555;text-transform:uppercase;">PIPELINE</p>', unsafe_allow_html=True)

    for title, desc in [
        ("Ingest",          "Schema flatten & JSON"),
        ("Hard Kill",       "Salary traps, honeypots"),
        ("Heuristic Score", "IR density, rare skills"),
        ("Semantic Boost",  "TF-IDF cosine similarity"),
        ("Sort",            "Deterministic sort"),
    ]:
        st.markdown(
            f'<p style="font-size:0.72rem;color:#f2f2f2;font-weight:500;margin:0 0 1px 0;">'
            f'<span style="color:#e8b84b;margin-right:6px;">◆</span>{title}</p>'
            f'<p style="{muted};margin:0 0 0.8rem 1rem;">{desc}</p>',
            unsafe_allow_html=True
        )

# ── CENTER PANEL ──────────────────────────────────────────────────────────────
with col_center:
    st.markdown(f"""
    <div class="list-header">
        <div class="list-title">Ranked Candidates</div>
        <div class="list-meta">showing {len(filtered)} of {len(df)}</div>
    </div>
    """, unsafe_allow_html=True)

    for _, row in filtered.iterrows():
        rank = int(row["rank"])
        score_pct = round(row["score"] * 100, 1)
        score_bar_w = int(row["score"] * 100)
        signals = parse_signals(row.get("reasoning", ""))
        snip = snippet(row.get("reasoning", "—"))

        if rank == 1:
            rank_cls, row_cls = "gold", "top3"
            circle_color = "var(--amber)"
        elif rank == 2:
            rank_cls, row_cls = "silver", "top3"
            circle_color = "#9ca3af"
        elif rank == 3:
            rank_cls, row_cls = "bronze", "top3"
            circle_color = "#cd7c2f"
        else:
            rank_cls, row_cls = "", ""
            circle_color = "var(--amber)"

        tags_html = " ".join(
            f'<span class="tag {cls}">{lbl}</span>'
            for lbl, cls in signals
        )
        
        c_name = row.get('name', 'Unknown')
        c_id = row['candidate_id']

        st.markdown(f"""
        <div class="c-row {row_cls}">
            <div class="c-rank {rank_cls}">{rank:02d}</div>
            <div class="c-body">
                <div class="c-name">{c_name}</div>
                <div class="c-id">{c_id}</div>
                <div class="c-snippet">{snip}</div>
                <div class="c-tags">{tags_html}</div>
            </div>
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

# ── RIGHT PANEL ───────────────────────────────────────────────────────────────
with col_right:
    # Score distribution (manual bucketed bars — no Plotly dependency)
    buckets = [(0, 0.2, "0–20%"), (0.2, 0.4, "20–40%"), (0.4, 0.6, "40–60%"),
               (0.6, 0.8, "60–80%"), (0.8, 1.01, "80–100%")]
    max_count = max(int(((df['score'] >= lo) & (df['score'] < hi)).sum())
                    for lo, hi, _ in buckets) or 1

    st.markdown('<div class="right-panel">', unsafe_allow_html=True)
    st.markdown('<div class="rp-section"><div class="rp-label">Score Distribution</div>', unsafe_allow_html=True)
    for lo, hi, lbl in reversed(buckets):
        count = int(((df['score'] >= lo) & (df['score'] < hi)).sum())
        bar_pct = int(count / max_count * 100)
        st.markdown(f"""
        <div class="score-dist-row">
            <div class="score-dist-label">{lbl}</div>
            <div class="score-dist-bar-wrap">
                <div class="score-dist-bar" style="width:{bar_pct}%;"></div>
            </div>
            <div class="score-dist-count">{count}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Top 5 leaderboard
    st.markdown('<div class="rp-section"><div class="rp-label">Top 5 Leaderboard</div>', unsafe_allow_html=True)
    for _, row in df.head(5).iterrows():
        rank = int(row["rank"])
        score_pct = round(row["score"] * 100, 1)
        cid = str(row["candidate_id"])
        amber_cls = ' style="color:var(--amber);"' if rank == 1 else ""
        st.markdown(f"""
        <div class="leader-row">
            <div class="leader-rank">#{rank}</div>
            <div class="leader-id" title="{cid}">{cid}</div>
            <div class="leader-score"{amber_cls}>{score_pct}%</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Team
    st.markdown("""
    <div style="font-family:var(--mono);font-size:0.62rem;color:var(--text-2);line-height:1.8;">
        <div class="rp-label">Team</div>
        <div>Aryan Bhargava</div>
        <div>Harshit Kudhial</div>
    </div>
    </div>
    """, unsafe_allow_html=True)
