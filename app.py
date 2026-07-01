"""
Recto — Premium Streamlit Dashboard

A flagship web UI for showcasing the Recto hybrid ranking pipeline output.
The sandbox demo link for the hackathon.

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import os
import re

# ──────────────────────────────────────────────────────────────────
# PAGE CONFIG (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Recto — AI Candidate Ranking",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────
# CUSTOM CSS — Premium dark glassmorphism theme
# ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --accent: #7C6FFF;
    --accent-light: #a59dff;
    --bg-card: rgba(255,255,255,0.04);
    --border: rgba(255,255,255,0.08);
    --text-muted: rgba(255,255,255,0.45);
    --text-secondary: rgba(255,255,255,0.65);
    --success: #34d399;
    --warning: #fbbf24;
    --danger: #f87171;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0d0d18 0%, #111127 50%, #0d1520 100%) !important;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; padding-bottom: 4rem !important; }

/* ── Hero Header ── */
.hero-banner {
    background: linear-gradient(135deg, rgba(124,111,255,0.15) 0%, rgba(56,189,248,0.08) 100%);
    border: 1px solid rgba(124,111,255,0.3);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -20%;
    width: 500px;
    height: 500px;
    background: radial-gradient(circle, rgba(124,111,255,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ffffff 0%, #a59dff 60%, #38bdf8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem 0;
    line-height: 1.2;
}
.hero-sub {
    color: var(--text-secondary);
    font-size: 1rem;
    font-weight: 400;
    margin: 0;
}

/* ── Stat Cards ── */
.stat-grid { display: flex; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; }
.stat-card {
    flex: 1;
    min-width: 150px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    text-align: center;
    backdrop-filter: blur(12px);
    transition: border-color 0.2s;
}
.stat-card:hover { border-color: rgba(124,111,255,0.4); }
.stat-number {
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent-light);
    display: block;
    line-height: 1;
}
.stat-label {
    font-size: 0.75rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 0.4rem;
    display: block;
}

/* ── Rank Badge ── */
.rank-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    border-radius: 10px;
    font-weight: 700;
    font-size: 0.85rem;
    font-family: 'JetBrains Mono', monospace;
    flex-shrink: 0;
}
.rank-1 { background: linear-gradient(135deg, #f59e0b, #ef4444); color: white; }
.rank-2 { background: linear-gradient(135deg, #94a3b8, #64748b); color: white; }
.rank-3 { background: linear-gradient(135deg, #d97706, #92400e); color: white; }
.rank-other { background: rgba(124,111,255,0.15); color: var(--accent-light); border: 1px solid rgba(124,111,255,0.25); }

/* ── Candidate Card ── */
.candidate-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s, background 0.2s;
    backdrop-filter: blur(8px);
}
.candidate-card:hover { border-color: rgba(124,111,255,0.35); background: rgba(255,255,255,0.06); }
.candidate-header { display: flex; align-items: flex-start; gap: 1rem; margin-bottom: 0.75rem; }
.candidate-id {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: var(--accent-light);
    background: rgba(124,111,255,0.12);
    padding: 2px 8px;
    border-radius: 6px;
}
.candidate-score {
    font-size: 1.4rem;
    font-weight: 700;
    color: white;
}
.candidate-score-label {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.candidate-reasoning {
    font-size: 0.82rem;
    color: var(--text-secondary);
    line-height: 1.6;
    border-top: 1px solid var(--border);
    padding-top: 0.75rem;
    margin-top: 0.5rem;
}

/* ── Score Bar ── */
.score-bar-wrap { width: 100%; background: rgba(255,255,255,0.07); border-radius: 999px; height: 6px; overflow: hidden; margin-top: 4px; }
.score-bar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #7C6FFF, #38bdf8); }

/* ── Tag Pill ── */
.tag-pill {
    display: inline-block;
    background: rgba(124,111,255,0.12);
    border: 1px solid rgba(124,111,255,0.2);
    color: var(--accent-light);
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.72rem;
    font-weight: 500;
    margin: 2px;
}

/* ── Section Header ── */
.section-header {
    color: white;
    font-size: 1.1rem;
    font-weight: 700;
    margin: 1.75rem 0 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
    margin-left: 0.5rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(13,13,24,0.9) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown h3 { color: white; font-weight: 600; }

/* ── Streamlit widget overrides ── */
.stTextInput input, .stSelectbox select, .stSlider {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    color: white !important;
}
.stButton button {
    background: linear-gradient(135deg, #7C6FFF, #5b52d4) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
}
.stButton button:hover { transform: translateY(-1px); box-shadow: 0 4px 15px rgba(124,111,255,0.35) !important; }

/* Download button styling */
[data-testid="stDownloadButton"] button {
    background: rgba(52,211,153,0.12) !important;
    border: 1px solid rgba(52,211,153,0.3) !important;
    color: #34d399 !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
}

/* Table styling */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* Alert boxes */
.stAlert { border-radius: 12px !important; }

/* Divider */
hr { border-color: var(--border) !important; }

/* Metric */
[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] { color: var(--accent-light) !important; font-family: 'JetBrains Mono', monospace !important; }
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# DATA LOADING
# ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    csv_path = "results/final_ranking.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()


def extract_tags(reasoning: str) -> list[str]:
    """Pull out interesting signal tags from the reasoning string."""
    tags = []
    patterns = [
        (r"OTW", "Open to Work"), (r"FAANG", "FAANG+"),
        (r"IIT ", "IIT"), (r"MIT|Stanford|Carnegie|Georgia Tech", "Top University"),
        (r"IR roles", "IR Roles"), (r"Learning to Rank", "LTR"),
        (r"rare IR skills", "Rare IR"), (r"github=(\d+)", None),
        (r"notice=(\d+)d", None), (r"active (\d+)d ago", None),
    ]
    if "OTW" in reasoning:
        tags.append("✅ Open to Work")
    if re.search(r"FAANG|Google|Meta|Amazon|Netflix|Microsoft", reasoning, re.I):
        tags.append("🏢 FAANG+")
    if re.search(r"IIT |MIT|Stanford|Carnegie|Georgia Tech|BITS", reasoning):
        tags.append("🎓 Top Uni")
    if re.search(r"\d+ IR roles", reasoning):
        m = re.search(r"(\d+) IR roles", reasoning)
        if m:
            tags.append(f"🔍 {m.group(1)} IR Roles")
    if "Learning to Rank" in reasoning or "LTR" in reasoning:
        tags.append("⚡ LTR")
    if "rare IR" in reasoning.lower():
        tags.append("💎 Rare IR Skills")
    return tags[:5]


def score_to_pct(score: float) -> float:
    return round(score * 100, 1)


# ──────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 Recto")
    st.markdown("""
    <p style='color:rgba(255,255,255,0.5); font-size:0.82rem; line-height:1.6;'>
    Deterministic AI candidate ranking.<br>
    Zero external APIs. Local CPU. &lt;60s runtime.
    </p>
    """, unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### Filters")
    top_n = st.slider("Show Top N Candidates", min_value=10, max_value=100, value=25, step=5)
    min_score = st.slider("Min Hybrid Score (%)", min_value=0, max_value=100, value=0, step=5)
    search_query = st.text_input("🔎 Search by ID or keyword", placeholder="e.g. CAND_00123 or Zomato")

    st.markdown("---")
    st.markdown("### About the Pipeline")
    st.markdown("""
    <p style='color:rgba(255,255,255,0.5); font-size:0.78rem; line-height:1.7;'>
    <b style='color:rgba(255,255,255,0.8)'>Phase 1</b> — Ingest & flatten schema<br>
    <b style='color:rgba(255,255,255,0.8)'>Phase 2</b> — Hard-kill filters<br>
    <b style='color:rgba(255,255,255,0.8)'>Phase 3</b> — Heuristic IR scoring<br>
    <b style='color:rgba(255,255,255,0.8)'>Phase 4</b> — TF-IDF semantic boost<br>
    <b style='color:rgba(255,255,255,0.8)'>Phase 5</b> — Deterministic sort<br>
    </p>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <p style='color:rgba(255,255,255,0.3); font-size:0.72rem;'>
    Built by <b style='color:rgba(255,255,255,0.5)'>Recto</b><br>
    Aryan Bhargava &amp; Harshit Kudhial
    </p>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────
# MAIN CONTENT
# ──────────────────────────────────────────────────────────────────
df = load_data()

# Hero
st.markdown("""
<div class="hero-banner">
    <p class="hero-title">Recto Ranking System</p>
    <p class="hero-sub">Hybrid heuristic + semantic pipeline · Zero LLM inference · Deterministic output</p>
</div>
""", unsafe_allow_html=True)

if df.empty:
    st.error("⚠️ No ranking data found. Run `python main.py --candidates <path_to_dataset.jsonl>` to generate results.")
    st.stop()

# ── Stats Row ──────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Ranked", f"{len(df)}")
with c2:
    st.metric("Top Score", f"{score_to_pct(df['score'].max())}%")
with c3:
    st.metric("Avg Score", f"{score_to_pct(df['score'].mean())}%")
with c4:
    st.metric("Median Score", f"{score_to_pct(df['score'].median())}%")

st.markdown("")

# ── Filter & Search ────────────────────────────────────────────────
filtered = df.copy()
filtered = filtered[filtered["score"] >= min_score / 100]
if search_query:
    mask = filtered.apply(lambda row: search_query.lower() in str(row.values).lower(), axis=1)
    filtered = filtered[mask]
filtered = filtered.head(top_n)

# ── Download ───────────────────────────────────────────────────────
col_dl, col_info = st.columns([1, 4])
with col_dl:
    st.download_button(
        label="⬇ Download CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="recto_final_ranking.csv",
        mime="text/csv",
    )
with col_info:
    st.markdown(f"""
    <p style='color:rgba(255,255,255,0.4); font-size:0.82rem; margin-top:0.6rem;'>
    Showing <b style='color:rgba(255,255,255,0.7)'>{len(filtered)}</b> candidates
    after filters · {len(df)} total ranked
    </p>
    """, unsafe_allow_html=True)

st.markdown("")

# ── Two-Column Layout: Cards + Score Distribution ──────────────────
col_cards, col_chart = st.columns([3, 2], gap="large")

with col_cards:
    st.markdown('<p class="section-header">🏆 Candidate Ranking</p>', unsafe_allow_html=True)

    for _, row in filtered.iterrows():
        rank = int(row["rank"])
        score_pct = score_to_pct(row["score"])
        score_bar = int(row["score"] * 100)
        tags = extract_tags(str(row.get("reasoning", "")))

        # Rank badge class
        if rank == 1:
            badge_cls = "rank-1"
        elif rank == 2:
            badge_cls = "rank-2"
        elif rank == 3:
            badge_cls = "rank-3"
        else:
            badge_cls = "rank-other"

        tag_html = " ".join([f'<span class="tag-pill">{t}</span>' for t in tags])
        reasoning_text = str(row.get("reasoning", "—"))

        st.markdown(f"""
        <div class="candidate-card">
            <div class="candidate-header">
                <div class="rank-badge {badge_cls}">#{rank}</div>
                <div style="flex:1;">
                    <div style="display:flex; align-items:center; gap:0.75rem; flex-wrap:wrap;">
                        <span class="candidate-id">{row['candidate_id']}</span>
                        {tag_html}
                    </div>
                    <div style="display:flex; align-items:baseline; gap:0.5rem; margin-top:0.5rem;">
                        <span class="candidate-score">{score_pct}%</span>
                        <span class="candidate-score-label">hybrid score</span>
                    </div>
                    <div class="score-bar-wrap">
                        <div class="score-bar-fill" style="width:{score_bar}%;"></div>
                    </div>
                </div>
            </div>
            <div class="candidate-reasoning">{reasoning_text[:280]}{'…' if len(reasoning_text) > 280 else ''}</div>
        </div>
        """, unsafe_allow_html=True)

with col_chart:
    st.markdown('<p class="section-header">📊 Score Distribution</p>', unsafe_allow_html=True)

    try:
        import plotly.express as px
        import plotly.graph_objects as go

        # Score histogram
        fig = px.histogram(
            df,
            x="score",
            nbins=20,
            labels={"score": "Hybrid Score", "count": "Candidates"},
            color_discrete_sequence=["#7C6FFF"],
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="rgba(255,255,255,0.6)",
            xaxis=dict(showgrid=False, color="rgba(255,255,255,0.3)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.06)", color="rgba(255,255,255,0.3)"),
            margin=dict(l=0, r=0, t=20, b=0),
            bargap=0.05,
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Top 10 bar chart
        st.markdown('<p class="section-header">🎯 Top 10 Scores</p>', unsafe_allow_html=True)
        top10 = df.head(10).copy()
        top10["label"] = top10["rank"].astype(str).apply(lambda r: f"#{r}")
        fig2 = go.Figure(go.Bar(
            x=top10["score"],
            y=top10["label"],
            orientation="h",
            marker=dict(
                color=top10["score"],
                colorscale=[[0, "#4f46e5"], [1, "#38bdf8"]],
                showscale=False,
            ),
            text=[f"{score_to_pct(s)}%" for s in top10["score"]],
            textposition="outside",
            textfont=dict(color="rgba(255,255,255,0.7)", size=11),
        ))
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="rgba(255,255,255,0.6)",
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(autorange="reversed", color="rgba(255,255,255,0.5)"),
            margin=dict(l=0, r=40, t=10, b=0),
            height=280,
        )
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    except ImportError:
        # Fallback if plotly not installed
        st.bar_chart(df.head(20).set_index("rank")["score"])

# ── Full Table View ────────────────────────────────────────────────
st.markdown('<p class="section-header">📋 Full Table View</p>', unsafe_allow_html=True)
st.dataframe(
    df[["rank", "candidate_id", "score", "reasoning"]].head(100).style.format({"score": "{:.4f}"}),
    use_container_width=True,
    height=350,
)

# ── Footer ─────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center; color:rgba(255,255,255,0.2); font-size:0.75rem; padding:1rem 0;'>
    Built by <strong style='color:rgba(255,255,255,0.4)'>Team Recto</strong> · 
    Aryan Bhargava &amp; Harshit Kudhial · 
    Redrob AI Hackathon 2025
</div>
""", unsafe_allow_html=True)
