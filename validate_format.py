import pandas as pd, re, sys

df = pd.read_csv('results/final_ranking.csv')

errors = []

# Exactly 100 rows
if len(df) != 100:
    errors.append(f'Row count: {len(df)}, need 100')

# Required columns in order
required = ['candidate_id', 'rank', 'score', 'reasoning']
if list(df.columns[:4]) != required:
    errors.append(f'Column order wrong: {df.columns.tolist()}')

# All CAND_XXXXXXX
bad_ids = df[~df['candidate_id'].str.match(r'^CAND_\d{7}$')]
if len(bad_ids):
    errors.append(f'{len(bad_ids)} invalid candidate_ids: {bad_ids["candidate_id"].head(3).tolist()}')

# Ranks 1-100 each exactly once
if sorted(df['rank'].tolist()) != list(range(1, 101)):
    errors.append('Ranks are not exactly 1-100 each once')

# Unique candidate_ids
if df['candidate_id'].duplicated().any():
    errors.append('Duplicate candidate_ids found')

# Score non-increasing
if not (df.sort_values('rank')['score'].diff().dropna() <= 0).all():
    errors.append('Scores not monotonically non-increasing with rank')

# No empty reasoning
empty_reasoning = df['reasoning'].isna() | (df['reasoning'].str.strip() == '')
if empty_reasoning.any():
    errors.append(f'{empty_reasoning.sum()} empty reasoning entries')

# Reasoning diversity check (not all identical)
unique_reasoning = df['reasoning'].nunique()
if unique_reasoning < 50:
    errors.append(f'Only {unique_reasoning} unique reasoning strings out of 100 — too templated')

if errors:
    print('ERRORS FOUND:')
    for e in errors: print(f'  ❌ {e}')
    sys.exit(1)
else:
    print('✅ All validation checks passed. Safe to submit.')
    print(f'   Unique reasoning entries: {unique_reasoning}/100')
    print(f'   Score range: {df["score"].min():.4f} to {df["score"].max():.4f}')
