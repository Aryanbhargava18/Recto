import pandas as pd
df = pd.read_csv('results/final_ranking.csv')
print("Ghost candidate CAND_0092278 in top 15?", 'CAND_0092278' in df['candidate_id'].head(15).tolist())
if 'CAND_0092278' in df['candidate_id'].values:
    print("Ghost candidate rank:", df[df['candidate_id'] == 'CAND_0092278']['rank'].iloc[0])
else:
    print("Ghost candidate not in top 100.")
print("NDCG Score check complete.")
