import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Recto - AI Candidate Ranking", layout="wide")

st.title("🚀 Recto: AI Candidate Ranking System")
st.markdown("This dashboard displays the top 100 Information Retrieval and Search Engineering candidates ranked by the Sherlock pipeline.")

@st.cache_data
def load_data():
    csv_path = 'results/final_ranking.csv'
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    return pd.DataFrame()

df = load_data()

if df.empty:
    st.error("No data found! Please run the pipeline first.")
else:
    st.success(f"Loaded {len(df)} ranked candidates.")
    
    st.header("Top 100 Candidates")
    st.dataframe(df, use_container_width=True, height=600)
    
    st.header("Top 10 Deep Dive")
    top10 = df.head(10)
    for _, row in top10.iterrows():
        with st.expander(f"Rank {row['rank']} - {row['candidate_id']} (Score: {row['score']:.1f})"):
            st.markdown(f"**Reasoning Audit Trail:**")
            st.info(row['reasoning'])
            
    st.markdown("---")
    st.markdown("*Built by Team Recto for the India Runs Hack2Skill Data & AI Challenge.*")
