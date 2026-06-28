import pandas as pd
import numpy as np
import re
import pickle
import os
import time

ULTRA_RARE_TERMS = ['bm25', 'ndcg', 'mrr', 'embedding', 'bi-encoder', 'cross-encoder', 'dense retrieval', 'sparse retrieval', 'hybrid retrieval', 'reranking', 'colbert', 'dpr', 'splade', 'hnsw', 'ann index', 'vector search', 'recall@k', 'precision@k']
HIGH_VALUE_TERMS = ['elasticsearch', 'solr', 'lucene', 'opensearch', 'faiss', 'pinecone', 'milvus', 'weaviate', 'qdrant', 'chroma']

PRODUCT_SIGNALS = ['founding', 'lead', 'staff', 'principal', 'head of', 'architect']
GROWTH_TITLES = ['engineer', 'scientist', 'researcher', 'developer']
STAGNANT_SIGNALS = ['analyst', 'consultant', 'associate', 'executive', 'specialist']

def load_data(filepath='filtered_candidates.pkl'):
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return pd.DataFrame()
    with open(filepath, 'rb') as f:
        return pickle.load(f)

# ANTI-GAMING: Coherence ratio
def coherence_score(text, term_list):
    text_lower = str(text).lower()
    total_words = max(len(text_lower.split()), 1)
    
    total_occurrences = sum(text_lower.count(t) for t in term_list)
    unique_terms = sum(1 for t in term_list if t in text_lower)
    
    if total_occurrences == 0:
        return 1.0  # no IR terms = not a stuffer, just not relevant
    
    repetition_ratio = total_occurrences / max(unique_terms, 1)
    coherence_multiplier = 1.0 if repetition_ratio <= 3 else max(0.4, 1.0 - (repetition_ratio - 3) * 0.1)
    
    density = total_occurrences / max(total_words / 100, 1)
    density_multiplier = 1.0 if density <= 12 else max(0.5, 1.0 - (density - 12) * 0.05)
    
    return coherence_multiplier * density_multiplier

# Career trajectory scoring
def trajectory_score(career_text, duration_months):
    text_lower = str(career_text).lower()
    
    product_mentions = sum(1 for s in PRODUCT_SIGNALS if s in text_lower)
    growth_role_count = sum(1 for s in GROWTH_TITLES if s in text_lower)
    stagnant_count = sum(1 for s in STAGNANT_SIGNALS if s in text_lower)
    
    company_indicators = text_lower.count(' at ') + text_lower.count('joined ')
    if duration_months > 0 and company_indicators > 0:
        avg_tenure = duration_months / max(company_indicators, 1)
        tenure_score = 10 if avg_tenure >= 24 else 5 if avg_tenure >= 18 else -5
    else:
        tenure_score = 0
    
    arc_score = (product_mentions * 5) + (growth_role_count * 3) - (stagnant_count * 4) + tenure_score
    return max(-20, min(25, arc_score))

def score_candidates(df, weights=None):
    if weights is None:
        weights = {'ultra_rare_weight': 20, 'services_penalty': -20, 'cv_penalty': -8}
        
    start_time = time.time()
    
    # 1. Rare skill extraction (from career_description as skills array is synthetic noise)
    def count_ultra_rare(text):
        if not isinstance(text, str): return 0
        text_lower = text.lower()
        return sum(1 for t in ULTRA_RARE_TERMS if t in text_lower)
    
    df['ultra_rare_hit_count'] = df['career_description'].apply(count_ultra_rare)
    df['rare_skill_bonus'] = df['ultra_rare_hit_count'] * weights['ultra_rare_weight']
    
    # 2. Coherence multiplier (Anti-Gaming)
    df['coherence_mult'] = df['career_description'].apply(
        lambda x: coherence_score(str(x), ULTRA_RARE_TERMS + HIGH_VALUE_TERMS)
    )
    
    # 3. Trajectory Score
    df['trajectory_score'] = df.apply(
        lambda row: trajectory_score(str(row['career_description']), row.get('duration_months', 0)), axis=1
    )
    
    # 4. Assessment Bonus & CV Trap Penalty
    def eval_assessment(scores):
        if not isinstance(scores, dict): return 0, False
        ltr = scores.get('LTR', 0)
        sent = scores.get('SentTrans', 0)
        yolo = scores.get('YOLO', 0)
        cnn = scores.get('CNN', 0)
        opencv = scores.get('OpenCV', 0)
        weight = scores.get('Weight', 0)
        
        bonus = 0
        if ltr > 70 or sent > 70:
            bonus += 6
            
        max_ir = max(ltr, sent)
        max_cv = max(yolo, cnn, opencv, weight)
        is_wrong = False
        if max_cv > 0 and max_cv >= max_ir:
            bonus += weights['cv_penalty']
            is_wrong = True
        return bonus, is_wrong
        
    assessment_results = df['assessment_scores'].apply(eval_assessment)
    df['assessment_bonus'] = assessment_results.apply(lambda x: x[0])
    df['wrong_domain'] = assessment_results.apply(lambda x: x[1])
    
    # 5. Services penalty recalculation (if we want dynamic weights)
    services_regex = re.compile(r'(?i)\b(TCS|Infosys|Wipro|Accenture|Cognizant)\b')
    df['services_flag'] = df['career_description'].apply(lambda x: bool(services_regex.search(str(x))))
    df['dynamic_services_penalty'] = np.where(df['services_flag'], weights['services_penalty'], 0)
    
    # LAYER 2 BASE SCORE
    df['core_ir_score'] = (
        df['rare_skill_bonus'] + 
        df['assessment_bonus'] + 
        df['dynamic_services_penalty']
    )
    
    # Apply Coherence & Trajectory
    df['core_ir_score'] = df['core_ir_score'] * df['coherence_mult']
    df['core_ir_score'] += df['trajectory_score'] * 0.5
    
    # LAYER 3: Behavioral
    active = df['active_days_since_last_login'].fillna(999)
    resp = df['response_rate'].fillna(0)
    
    m_act = np.where(active < 30, 1.0, np.where(active <= 60, 0.85, np.where(active <= 90, 0.70, 0.50)))
    m_otw = np.where(df['open_to_work'] == True, 1.0, 0.9)
    m_resp = np.where(resp > 0.7, 1.0, np.where(resp >= 0.5, 0.95, 0.75))
             
    notice = df['notice_period_days'].fillna(30)
    m_not = np.where(notice < 30, 0.92, np.where(notice <= 60, 1.0, np.where(notice <= 90, 0.85, 0.75)))
            
    stars = df['github_stars'].fillna(0)
    m_git = np.where(stars > 50, 1.05, 1.0)
    
    # Specific hackathon ghost penalty constraint
    # CAND_0092278 has 1417 search appearances but saved < 5 and rr < 0.2
    appearances = df.get('search_appearance', df.get('search_appearances', pd.Series(0))).fillna(0)
    saved = df.get('recruiter_saves', df.get('saved', pd.Series(0))).fillna(0)
    
    ghost_penalty = np.where(
        (appearances > 800) & (saved < 5) & (resp < 0.2),
        0.5,
        1.0
    )
    
    df['behavioral_multiplier'] = m_act * m_otw * m_resp * m_not * m_git * ghost_penalty
    df['final_score'] = df['core_ir_score'] * df['behavioral_multiplier']
    
    # Ghost Detection for sorting tier logic
    GHOST_THRESHOLD = (df['active_days_since_last_login'] > 180) & (df['response_rate'] < 0.1)
    df['ghost_flag'] = GHOST_THRESHOLD
    
    # Deterministic Genuine Practitioner Check (replacing Gemini)
    df['genuine_practitioner'] = (df['trajectory_score'] > 0) & (df['wrong_domain'] == False) & (df['coherence_mult'] > 0.8)
    
    # Map final_score to HYBRID_SCORE for compatibility with legacy formatter
    df['HYBRID_SCORE'] = df['final_score']
    
    df = df.sort_values(by='final_score', ascending=False).reset_index(drop=True)
    return df

def ndcg_optimal_sort(df_top30):
    """
    For positions 1-3: require ultra_rare_hit_count >= 2 AND genuine_practitioner=True
    For positions 4-7: ultra_rare_hit_count >= 1
    For positions 8-10: any genuine IR candidate
    """
    tier_1 = df_top30[
        (df_top30['ultra_rare_hit_count'] >= 2) & 
        (df_top30['genuine_practitioner'] == True) &
        (~df_top30['ghost_flag'].fillna(False))
    ].sort_values('HYBRID_SCORE', ascending=False).head(3)
    
    remaining = df_top30[~df_top30['candidate_id'].isin(tier_1['candidate_id'])]
    
    tier_2 = remaining[
        (remaining['ultra_rare_hit_count'] >= 1) &
        (remaining['wrong_domain'] == False)
    ].sort_values('HYBRID_SCORE', ascending=False).head(4)
    
    remaining2 = remaining[~remaining['candidate_id'].isin(tier_2['candidate_id'])]
    
    tier_3 = remaining2[
        remaining2['wrong_domain'] == False
    ].sort_values('HYBRID_SCORE', ascending=False).head(3)
    
    remaining3 = remaining2[~remaining2['candidate_id'].isin(tier_3['candidate_id'])].sort_values('HYBRID_SCORE', ascending=False)
    
    # Concatenate exactly as described, resulting in NDCG-optimized list
    return pd.concat([tier_1, tier_2, tier_3, remaining3]).reset_index(drop=True)

if __name__ == "__main__":
    df = load_data('filtered_candidates.pkl')
    if not df.empty:
        scored_df = score_candidates(df)
        scored_df.to_pickle('scored_candidates.pkl')
        scored_df.head(200).to_pickle('top200_candidates.pkl')
        print("Completed local score generation.")
