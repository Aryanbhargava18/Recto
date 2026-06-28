import pandas as pd
df = pd.read_pickle('results/final_ranked.pkl')
for cid in ['CAND_0070669', 'CAND_0014266', 'CAND_0066114', 'CAND_0042871', 'CAND_0092278']:
    row = df[df['candidate_id'] == cid]
    if len(row):
        r = row.iloc[0]
        print(f"{cid} - Rank {r['rank']}, Score: {r['HYBRID_SCORE']:.2f}, RareHits: {r['ultra_rare_hit_count']}")
    else:
        print(f"{cid} NOT IN PKL!")
