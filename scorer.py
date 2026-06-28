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
        weights = {}
        
    start_time = time.time()
    
    RARE_SKILLS = [
        'Information Retrieval Systems', 'Search Infrastructure', 'Ranking Systems',
        'Text Encoders', 'Dense Retrieval', 'Indexing Algorithms',
        'Embedding Models', 'Passage Retrieval', 'Query Understanding',
        'Semantic Indexing', 'Document Reranking', 'Retrieval Augmented Generation',
        'Learning to Rank', 'Sparse Retrieval'
    ]
    
    IR_TEMPLATES = [
        'bm25', 'dense retrieval', 'hybrid search', 'vector', 'embedding',
        'information retrieval', 'semantic search', 'ranking', 'faiss',
        'pinecone', 'weaviate', 'elasticsearch', 'ndcg', 'bi-encoder'
    ]
    
    RELEVANT_ASSESSMENTS = [
        'Learning to Rank', 'Sentence Transformers', 'FAISS',
        'Vector Search', 'Semantic Search', 'Embeddings',
        'Fine-tuning LLMs', 'RAG', 'Haystack', 'Weaviate'
    ]
    
    def calc_row_score(row):
        # STEP 1: Core technical score (additive, no multiplier)
        core_score = 0
        
        # Rare skills — each held rare skill = +12 pts (additive)
        candidate_skills = row.get('skills', [])
        if not isinstance(candidate_skills, list):
            candidate_skills = []
        rare_hits = sum(1 for s in candidate_skills if s in RARE_SKILLS)
        core_score += rare_hits * 12
        
        career_text_lower = str(row.get('career_description', '')).lower()
        
        # Gold template
        has_gold_template = 'connect users with relevant information at scale' in career_text_lower
        if has_gold_template:
            core_score += 25
            
        # IR role count = +15 pts each (up to 3 roles = +45)
        # Note: the user provided a snippet for this in output_formatter:
        # we can just use a proxy here or calculate exactly
        # Since career_history was flattened, we can look at occurrences of IR_TEMPLATES
        ir_role_count = row.get('ir_roles_count', 0)
        # Wait, ir_roles_count wasn't extracted in data_loader! Let's calculate it if missing.
        # Actually, let's just count occurrences of IR templates in career_description loosely 
        # or wait, output_formatter generator has exact logic for ir_role_count:
        # "desc = (ch.get('description', '') + ' ' + ch.get('title', '')).lower(); if any(kw in desc for kw in IR_TEMPLATES)..."
        # Since I am in the scorer, I can either extract it from the raw column or approximate.
        # Wait! df['profile'] wasn't dropped! I can iterate over row['career_history']
        career = row.get('career_history', [])
        if not isinstance(career, list): career = []
        computed_ir_role_count = 0
        for ch in career:
            desc = (ch.get('description', '') + ' ' + ch.get('title', '')).lower()
            if any(kw in desc for kw in IR_TEMPLATES):
                computed_ir_role_count += 1
        core_score += min(computed_ir_role_count, 4) * 10
        
        # Unique IR keywords in text
        unique_ir_keywords_in_text = sum(1 for kw in IR_TEMPLATES if kw in career_text_lower)
        core_score += min(unique_ir_keywords_in_text, 5) * 3
        
        # Relevant IR assessments
        assessments = row.get('assessment_scores', {})
        if not isinstance(assessments, dict): assessments = {}
        relevant_assessment_count = sum(1 for k in assessments.keys() if k in RELEVANT_ASSESSMENTS)
        core_score += min(relevant_assessment_count, 3) * 6
        
        yoe = row.get('duration_months', 0) / 12.0
        if 5 <= yoe <= 9:
            core_score += 5
        elif 4 <= yoe < 5 or 9 < yoe <= 12:
            core_score += 2
            
        # Tier-1 education
        tier_1_edu = False
        education = row.get('education', [])
        if isinstance(education, list):
            for e in education:
                if isinstance(e, dict) and e.get('tier') == 'tier_1':
                    tier_1_edu = True
                    break
        if tier_1_edu:
            core_score += 3
            
        # Services-only disqualifier
        services_regex = re.compile(r'(?i)\b(TCS|Infosys|Wipro|Accenture|Cognizant)\b')
        services_only_career = bool(services_regex.search(career_text_lower))
        if services_only_career:
            core_score -= 20
            
        # STEP 2: Behavioral multiplier (NEVER below 0.4)
        behavioral_mult = 1.0
        
        days_since_active = row.get('active_days_since_last_login', 999)
        if pd.isna(days_since_active): days_since_active = 999
        if days_since_active <= 30:
            behavioral_mult *= 1.0
        elif days_since_active <= 60:
            behavioral_mult *= 0.85
        elif days_since_active <= 90:
            behavioral_mult *= 0.70
        else:
            behavioral_mult *= 0.50
            
        open_to_work = row.get('open_to_work', False)
        if not open_to_work:
            behavioral_mult *= 0.90
            
        recruiter_response_rate = row.get('response_rate', 0.0)
        if pd.isna(recruiter_response_rate): recruiter_response_rate = 0.0
        if recruiter_response_rate >= 0.7:
            behavioral_mult *= 1.0
        elif recruiter_response_rate >= 0.5:
            behavioral_mult *= 0.95
        elif recruiter_response_rate >= 0.3:
            behavioral_mult *= 0.85
        else:
            behavioral_mult *= 0.75
            
        notice_period_days = row.get('notice_period_days', 90)
        if pd.isna(notice_period_days): notice_period_days = 90
        if notice_period_days <= 30:
            behavioral_mult *= 1.0
        elif notice_period_days <= 60:
            behavioral_mult *= 0.92
        elif notice_period_days <= 90:
            behavioral_mult *= 0.82
        else:
            behavioral_mult *= 0.70
            
        search_appearance_30d = row.get('search_appearance', 0)
        if pd.isna(search_appearance_30d): search_appearance_30d = 0
        saved_by_recruiters_30d = row.get('recruiter_saves', 0)
        if pd.isna(saved_by_recruiters_30d): saved_by_recruiters_30d = 0
        
        is_ghost = False
        if search_appearance_30d > 800 and saved_by_recruiters_30d < 5 and recruiter_response_rate < 0.2:
            behavioral_mult *= 0.50
            is_ghost = True
            
        behavioral_mult = max(behavioral_mult, 0.40)
        
        # STEP 3: Saved-by-recruiters bonus
        saved_bonus = 0
        if saved_by_recruiters_30d > 60:
            saved_bonus = 15
        elif saved_by_recruiters_30d > 30:
            saved_bonus = 10
        elif saved_by_recruiters_30d > 15:
            saved_bonus = 6
        elif saved_by_recruiters_30d > 5:
            saved_bonus = 3
            
        # STEP 4: LinkedIn bonus
        # Look at redrob_signals if needed, but since we didn't extract linkedin_connected, let's extract it safely
        redrob = row.get('redrob_signals', {})
        if not isinstance(redrob, dict): redrob = {}
        linkedin_connected = redrob.get('linkedin_connected', False)
        linkedin_bonus = 4 if linkedin_connected else 0
        
        # STEP 5: GitHub bonus
        github_activity_score = row.get('github_stars', 0)
        if pd.isna(github_activity_score): github_activity_score = 0
        github_bonus = 0
        if github_activity_score >= 70:
            github_bonus = 3
        elif github_activity_score >= 50:
            github_bonus = 2
        elif github_activity_score >= 30:
            github_bonus = 1
            
        final_score = (core_score * behavioral_mult) + saved_bonus + linkedin_bonus + github_bonus
        
        # Required extra fields for deterministic sort and output
        # For tier logic, ultra_rare_hit_count is used. It's rare_hits (from RARE_SKILLS array) or from ULTRA_RARE_TERMS?
        # Let's define ultra_rare_hit_count as rare_hits (rare skills held) 
        # Wait, the instruction says "rank 10 has 8 rare hits". The script calls it `ultra_rare_hit_count = 8`.
        ultra_rare_hit_count = rare_hits
        genuine_practitioner = (computed_ir_role_count >= 1) and (not services_only_career)
        
        return pd.Series({
            'final_score': final_score,
            'ultra_rare_hit_count': ultra_rare_hit_count,
            'genuine_practitioner': genuine_practitioner,
            'wrong_domain': False,  # Simplified since cv_penalty removed from core score
            'ghost_flag': is_ghost,
            'ir_roles_count': computed_ir_role_count
        })

    # Apply calculation
    results = df.apply(calc_row_score, axis=1)
    df['final_score'] = results['final_score']
    df['ultra_rare_hit_count'] = results['ultra_rare_hit_count']
    df['genuine_practitioner'] = results['genuine_practitioner']
    df['wrong_domain'] = results['wrong_domain']
    df['ghost_flag'] = results['ghost_flag']
    df['ir_roles_count'] = results['ir_roles_count']
    
    df['HYBRID_SCORE'] = df['final_score']
    df['rule_score'] = df['final_score']
    df['semantic_score'] = 0.0
    
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
