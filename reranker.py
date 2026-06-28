import os
import json
import time
import pickle
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import scorer

CACHE_FILE = 'rerank_cache.json'
MAX_RETRIES = 4
BACKOFF_TIMES = [4, 8, 16, 32]
BATCH_SIZE = 5

BOUNDARY_PROMPT = """
You are making one critical hiring decision.

Candidate A (currently rank {rank_a}):
{profile_a}

Candidate B (currently rank {rank_b}):
{profile_b}

For a Senior AI Engineer role building hybrid search + embeddings + ranking systems at a product company:
Return ONLY raw JSON, no markdown, exactly this format: {{"winner": "A", "confidence": "high", "deciding_factor": "..."}}
"""

def load_rerank_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_rerank_cache(cache):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)

def get_jd_summary():
    if not os.path.exists('jd_cache.json'):
        return "Information Retrieval Engineer. Requires Search, NLP, and relevance tuning."
    with open('jd_cache.json', 'r', encoding='utf-8') as f:
        try:
            cache = json.load(f)
            if not cache: return "Information Retrieval Engineer. Requires Search, NLP, and relevance tuning."
            first_key = list(cache.keys())[0]
            return json.dumps(cache[first_key], indent=2)
        except Exception:
            return "Information Retrieval Engineer. Requires Search, NLP, and relevance tuning."

def construct_candidates_text(candidates):
    text_blocks = []
    for cand in candidates:
        cid = str(cand['candidate_id'])
        desc = str(cand.get('career_description', ''))[:500]
        roles_count = cand.get('ir_roles_count', 0)
        rare = cand.get('rare_skills', [])
        scores_raw = cand.get('assessment_scores', {})
        if isinstance(scores_raw, dict):
            sorted_scores = dict(sorted(scores_raw.items(), key=lambda item: item[1], reverse=True)[:3])
        else:
            sorted_scores = {}

        block = (
            f"Candidate ID: {cid}\n"
            f"Career Description (First 500 chars): {desc}\n"
            f"IR Roles Count: {roles_count}\n"
            f"Rare Skills: {rare}\n"
            f"Top Assessment Scores: {sorted_scores}\n"
        )
        text_blocks.append(block)
    return "\n---\n".join(text_blocks)

def evaluate_batch(model, candidates, jd_summary, cache):
    if not candidates: return []
    candidates_text = construct_candidates_text(candidates)
    
    user_prompt = f"""Job Requirements Summary: {jd_summary}

Evaluate these {len(candidates)} candidates and return ONLY a JSON array, no markdown:
[{{
  "candidate_id": "...",
  "semantic_score": 0-100,
  "career_narrative_quality": "strong|moderate|weak",
  "genuine_ir_depth": true/false,
  "key_strengths": ["max 3 items"],
  "recruiter_summary": "2 sentence summary a recruiter would trust",
  "rank_within_batch": 1-5
}}]

Candidates:
{candidates_text}
"""
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = model.generate_content(user_prompt)
            result_text = response.text.strip()
            
            if result_text.startswith("```json"): result_text = result_text[7:]
            if result_text.startswith("```"): result_text = result_text[3:]
            if result_text.endswith("```"): result_text = result_text[:-3]
            
            parsed_array = json.loads(result_text.strip())
            
            for item in parsed_array:
                cid = str(item.get('candidate_id'))
                cache[cid] = item
            return parsed_array
        except Exception as e:
            if "429" in str(e) or "Quota" in str(e):
                if attempt < MAX_RETRIES:
                    time.sleep(BACKOFF_TIMES[attempt])
                else:
                    return []
            else:
                return []
    return []

def call_gemini_pairwise(model, cand_a, cand_b, cache):
    cid_a = str(cand_a['candidate_id'])
    cid_b = str(cand_b['candidate_id'])
    pair_key = f"pair_{cid_a}_{cid_b}"
    if pair_key in cache: return cache[pair_key]

    prompt = BOUNDARY_PROMPT.format(
        rank_a=cand_a['rank'], profile_a=str(cand_a.get('career_description', ''))[:300],
        rank_b=cand_b['rank'], profile_b=str(cand_b.get('career_description', ''))[:300]
    )

    for attempt in range(MAX_RETRIES + 1):
        try:
            response = model.generate_content(prompt)
            txt = response.text.strip()
            if txt.startswith("```json"): txt = txt[7:]
            if txt.startswith("```"): txt = txt[3:]
            if txt.endswith("```"): txt = txt[:-3]
            
            result = json.loads(txt.strip())
            cache[pair_key] = result
            save_rerank_cache(cache)
            return result
        except Exception as e:
            if "429" in str(e) or "Quota" in str(e):
                if attempt < MAX_RETRIES:
                    time.sleep(BACKOFF_TIMES[attempt])
                else: return {'winner': 'A', 'confidence': 'low'}
            else: return {'winner': 'A', 'confidence': 'low'}
    return {'winner': 'A', 'confidence': 'low'}

def boundary_calibration(model, df_ranked, cache):
    if len(df_ranked) < 14:
        return df_ranked
        
    boundary_zone = df_ranked.iloc[7:14].copy()
    swaps = []
    for i in range(len(boundary_zone) - 1):
        cand_a = boundary_zone.iloc[i]
        cand_b = boundary_zone.iloc[i+1]
        
        if abs(cand_a['HYBRID_SCORE'] - cand_b['HYBRID_SCORE']) < 5:
            result = call_gemini_pairwise(model, cand_a, cand_b, cache)
            if result.get('winner') == 'B' and result.get('confidence') == 'high':
                swaps.append((i, i+1))
                
    for idx_a, idx_b in swaps:
        row_a = boundary_zone.iloc[idx_a].copy()
        row_b = boundary_zone.iloc[idx_b].copy()
        boundary_zone.iloc[idx_a] = row_b
        boundary_zone.iloc[idx_b] = row_a
        
    df_ranked.iloc[7:14] = boundary_zone.values
    return df_ranked

def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment.")
        return
        
    genai.configure(api_key=api_key)
    system_instruction = "You are an expert technical recruiter evaluating candidates for an Information Retrieval / Search engineering role."
    model = genai.GenerativeModel(model_name="gemini-2.5-flash", system_instruction=system_instruction, generation_config={"response_mime_type": "application/json"})
    
    if not os.path.exists('top200_candidates.pkl'):
        return
    with open('top200_candidates.pkl', 'rb') as f:
        df = pickle.load(f)
    
    jd_summary = get_jd_summary()
    cache = load_rerank_cache()
    
    # 1. Evaluate un-cached candidates
    uncached_candidates = []
    results = []
    for _, row in df.iterrows():
        cid = str(row['candidate_id'])
        if cid in cache: results.append(cache[cid])
        else: uncached_candidates.append(row.to_dict())
            
    for i in range(0, len(uncached_candidates), BATCH_SIZE):
        batch = uncached_candidates[i:i+BATCH_SIZE]
        results.extend(evaluate_batch(model, batch, jd_summary, cache))
        save_rerank_cache(cache)
        if i + BATCH_SIZE < len(uncached_candidates): time.sleep(5)
            
    # 2. Merge & calculate HYBRID_SCORE
    semantic_df = pd.DataFrame(results)
    if not semantic_df.empty:
        semantic_df['candidate_id'] = semantic_df['candidate_id'].astype(str)
    df['candidate_id'] = df['candidate_id'].astype(str)
    df = df.rename(columns={'final_score': 'rule_score'})
    merged_df = pd.merge(df, semantic_df, on='candidate_id', how='left')
    
    merged_df['semantic_score'] = merged_df.get('semantic_score', pd.Series(0)).fillna(0).astype(float)
    merged_df['genuine_practitioner'] = merged_df.get('genuine_ir_depth', pd.Series(False)).fillna(False).astype(bool)
    
    merged_df['HYBRID_SCORE'] = (merged_df['rule_score'] * 0.6) + (merged_df['semantic_score'] * 0.4)
    merged_df = merged_df.sort_values(by='HYBRID_SCORE', ascending=False).reset_index(drop=True)
    merged_df['rank'] = merged_df.index + 1
    
    # 3. Pairwise boundary calibration
    print("Running Pairwise Boundary Calibration on ranks 8-14...")
    merged_df = boundary_calibration(model, merged_df, cache)
    
    # 4. NDCG Optimal Sort
    print("Applying NDCG-Optimal Sort...")
    merged_df = scorer.ndcg_optimal_sort(merged_df)
    
    merged_df['rank'] = merged_df.index + 1
    merged_df.to_pickle('final_ranked.pkl')
    
    csv_cols = ['rank', 'candidate_id', 'name', 'HYBRID_SCORE', 'rule_score', 'semantic_score', 'recruiter_summary', 'key_strengths', 'ultra_rare_hit_count', 'genuine_practitioner', 'wrong_domain']
    csv_df = merged_df.rename(columns={'name': 'candidate_name'})
    csv_cols = [c if c != 'name' else 'candidate_name' for c in csv_cols]
    csv_cols = [c for c in csv_cols if c in csv_df.columns]
    csv_df[csv_cols].to_csv('final_ranking.csv', index=False)
    
if __name__ == "__main__":
    main()
