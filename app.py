"""
Recto — Streamlit Dashboard

A lightweight web UI for viewing ranked candidates after the pipeline runs.
Can be used as the hackathon sandbox demo link.

Usage:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Recto — AI Candidate Ranking", layout="wide")
st.title("🚀 Recto: AI Candidate Ranking System")
st.markdown(
    "Displays the top 100 Information Retrieval candidates "
    "ranked by the Recto hybrid scoring pipeline."
)


@st.cache_data
def load_data():
    csv_path = "results/final_ranking.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()


df = load_data()

if df.empty:
    st.error("No data found. Run the pipeline first: `python main.py --candidates candidates.jsonl`")
else:
    st.success(f"Loaded {len(df)} ranked candidates.")

    st.header("Top 100 Candidates")
    st.dataframe(df, use_container_width=True, height=600)

    st.header("Top 10 Deep Dive")
    for _, row in df.head(10).iterrows():
        with st.expander(f"Rank {row['rank']} — {row['candidate_id']} (Score: {row['score']:.4f})"):
            st.markdown("**Reasoning:**")
            st.info(row["reasoning"])

    st.markdown("---")
    st.caption("Built by Team Recto for the India Runs Data & AI Challenge.")
