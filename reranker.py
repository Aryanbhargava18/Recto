import os
import json
import time
import pickle
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

CACHE_FILE = 'rerank_cache.json'
MAX_RETRIES = 4
BACKOFF_TIMES = [4, 8, 16, 32]
BATCH_SIZE = 5

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
    """Loads the JD summary from jd_cache.json."""
    if not os.path.exists('jd_cache.json'):
        print("Warning: jd_cache.json not found. Using generic fallback JD.")
        return "Information Retrieval Engineer. Requires Search, NLP, and relevance tuning."
    
    with open('jd_cache.json', 'r', encoding='utf-8') as f:
        try:
            cache = json.load(f)
            if not cache:
                return "Information Retrieval Engineer. Requires Search, NLP, and relevance tuning."
            
            # Since jd_cache is keyed by hash, we'll grab the first one assuming 1 active JD
            first_key = list(cache.keys())[0]
            jd_data = cache[first_key]
            return json.dumps(jd_data, indent=2)
        except Exception as e:
            print(f"Error loading JD cache: {e}")
            return "Information Retrieval Engineer. Requires Search, NLP, and relevance tuning."

def construct_candidates_text(candidates):
    """Formats the candidate data for the LLM prompt."""
    text_blocks = []
    for cand in candidates:
        cid = str(cand['candidate_id'])
        desc = str(cand.get('career_description', ''))[:500]
        roles_count = cand.get('ir_roles_count', 0)
        rare = cand.get('rare_skills', [])
        
        # safely extract assessment scores
        scores_raw = cand.get('assessment_scores', {})
        if isinstance(scores_raw, dict):
            # Sort by highest scores
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
    """Sends a batch of candidates to the LLM and parses the JSON response."""
    if not candidates:
        return []
        
    system_instruction = (
        "You are an expert technical recruiter evaluating candidates for an Information Retrieval / Search engineering role. "
        "You understand the difference between genuine IR expertise and keyword stuffing. Score fairly."
    )
    
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
            # We override the model's system prompt for this specific task
            response = model.generate_content(
                user_prompt,
                # system_instruction doesn't directly override on generate_content in some API versions,
                # but we instantiated the model with it earlier if possible, or pass it as first part.
                # Since the prompt requires the exact SYSTEM string, we let the model object handle it.
            )
            result_text = response.text.strip()
            
            # Defensive clean up
            if result_text.startswith("```json"): result_text = result_text[7:]
            if result_text.startswith("```"): result_text = result_text[3:]
            if result_text.endswith("```"): result_text = result_text[:-3]
            
            parsed_array = json.loads(result_text.strip())
            
            # Cache the results
            for item in parsed_array:
                cid = str(item.get('candidate_id'))
                cache[cid] = item
            
            return parsed_array
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "Quota" in error_msg or "Too Many Requests" in error_msg:
                if attempt < MAX_RETRIES:
                    wait_time = BACKOFF_TIMES[attempt]
                    print(f"Rate limit hit. Retrying in {wait_time}s... (Attempt {attempt+1}/{MAX_RETRIES})")
                    time.sleep(wait_time)
                else:
                    print("Max retries reached for batch. Returning empty.")
                    return []
            else:
                print(f"API/Parsing error: {e}\nRaw output: {result_text if 'result_text' in locals() else 'None'}")
                return []
    return []

def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in environment.")
        return
        
    genai.configure(api_key=api_key)
    
    system_instruction = (
        "You are an expert technical recruiter evaluating candidates for an Information Retrieval / Search engineering role. "
        "You understand the difference between genuine IR expertise and keyword stuffing. Score fairly."
    )
    
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=system_instruction,
        generation_config={"response_mime_type": "application/json"}
    )
    
    if not os.path.exists('top200_candidates.pkl'):
        print("Error: top200_candidates.pkl not found. Run scorer.py first.")
        return
        
    with open('top200_candidates.pkl', 'rb') as f:
        df = pickle.load(f)
    
    jd_summary = get_jd_summary()
    cache = load_rerank_cache()
    
    print(f"Loaded {len(df)} candidates to rerank.")
    
    # 1. Separate cached vs uncached candidates
    uncached_candidates = []
    results = []
    
    for _, row in df.iterrows():
        cid = str(row['candidate_id'])
        if cid in cache:
            results.append(cache[cid])
        else:
            uncached_candidates.append(row.to_dict())
            
    print(f"Found {len(results)} in cache. {len(uncached_candidates)} need LLM evaluation.")
    
    # 2. Process uncached in batches of 5
    api_calls_used = 0
    total_processed = len(results)
    
    if uncached_candidates:
        print(f"Processing in {len(uncached_candidates) // BATCH_SIZE + (1 if len(uncached_candidates) % BATCH_SIZE != 0 else 0)} batches...")
        
    for i in range(0, len(uncached_candidates), BATCH_SIZE):
        batch = uncached_candidates[i:i+BATCH_SIZE]
        
        batch_results = evaluate_batch(model, batch, jd_summary, cache)
        results.extend(batch_results)
        
        # Save cache progressively
        save_rerank_cache(cache)
        
        api_calls_used += 1
        total_processed += len(batch)
        
        # Print progress
        if total_processed % 10 == 0 or total_processed == len(df):
            print(f"Progress: {total_processed}/{len(df)} candidates evaluated.")
            
        # Enforce sleep to stay under 15 RPM constraints (5 seconds between batches)
        if i + BATCH_SIZE < len(uncached_candidates):
            time.sleep(5)
            
    print(f"\nCompleted reranking! Total API calls made this run: {api_calls_used}")
    
    # 3. Merge semantic results back into DataFrame
    semantic_df = pd.DataFrame(results)
    
    # Ensure IDs match types for merging
    semantic_df['candidate_id'] = semantic_df['candidate_id'].astype(str)
    df['candidate_id'] = df['candidate_id'].astype(str)
    
    # Rename 'final_score' from phase 1 to 'rule_score' for clarity
    df = df.rename(columns={'final_score': 'rule_score'})
    
    merged_df = pd.merge(df, semantic_df, on='candidate_id', how='left')
    
    # Handle any potential missing semantic scores if an API call failed completely
    merged_df['semantic_score'] = merged_df['semantic_score'].fillna(0).astype(float)
    
    # 4. Compute Hybrid Score
    merged_df['HYBRID_SCORE'] = (merged_df['rule_score'] * 0.6) + (merged_df['semantic_score'] * 0.4)
    
    # 5. Re-sort and Rank
    merged_df = merged_df.sort_values(by='HYBRID_SCORE', ascending=False).reset_index(drop=True)
    merged_df['rank'] = merged_df.index + 1
    
    # 6. Save final outputs
    merged_df.to_pickle('final_ranked.pkl')
    print("Saved 'final_ranked.pkl'")
    
    csv_cols = [
        'rank', 'candidate_id', 'name', 'HYBRID_SCORE', 
        'rule_score', 'semantic_score', 'recruiter_summary', 'key_strengths'
    ]
    
    # Rename 'name' to 'candidate_name' in CSV if requested (user asked for 'candidate_name')
    csv_df = merged_df.rename(columns={'name': 'candidate_name'})
    csv_cols = [c if c != 'name' else 'candidate_name' for c in csv_cols]
    
    # Keep only available columns
    csv_cols = [c for c in csv_cols if c in csv_df.columns]
    
    csv_df[csv_cols].to_csv('final_ranking.csv', index=False)
    print("Saved 'final_ranking.csv'")
    
    print("\n=== TOP 5 FINAL RANKED CANDIDATES ===")
    print(csv_df[['rank', 'candidate_id', 'candidate_name', 'HYBRID_SCORE', 'rule_score', 'semantic_score']].head(5).to_string(index=False))

if __name__ == "__main__":
    main()
