import pandas as pd
df = pd.read_json('candidates.jsonl', lines=True)
cids = ['CAND_0014266', 'CAND_0042871']
for c in cids:
    row = df[df['candidate_id'] == c]
    if len(row):
        print(row.iloc[0].to_dict())
