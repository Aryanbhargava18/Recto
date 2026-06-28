import pandas as pd
import sys

df = pd.read_csv('results/final_ranking.csv')

top_15_ids = df['candidate_id'].head(15).tolist()
top_8_expected = [
    'CAND_0006567', 'CAND_0037980', 'CAND_0080766', 'CAND_0018499',
    'CAND_0070669', 'CAND_0014266', 'CAND_0066114', 'CAND_0042871'
]
ghost_candidate = 'CAND_0092278'

errors = []
for cid in top_8_expected:
    if cid not in top_15_ids:
        errors.append(f'Missing {cid} in top 15!')

if ghost_candidate in top_15_ids:
    errors.append(f'Ghost candidate {ghost_candidate} is in top 15! (Rank {top_15_ids.index(ghost_candidate) + 1})')

if errors:
    print("Errors:")
    for e in errors:
        print("  ❌ " + e)
    sys.exit(1)
else:
    print("✅ All expected candidates are in the Top 15!")
    print("✅ Ghost candidate was successfully downranked out of the Top 15!")
