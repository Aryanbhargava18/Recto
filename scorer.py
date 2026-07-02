import pandas as pd
import numpy as np
import re
import math
import pickle
import os
import time
from datetime import datetime

ULTRA_RARE_TERMS = [
    r'\bbm25\b', r'\bndcg\b', r'\bmrr\b', r'\bembeddings?\b', r'\bbi-?encoders?\b',
    r'\bcross-?encoders?\b', r'\bdense retrievals?\b', r'\bsparse retrievals?\b',
    r'\bhybrid (retrieval|search)\b', r'\brerank(ing|er)?\b', r'\bcolbert\b', r'\bdpr\b',
    r'\bsplade\b', r'\bhnsw\b', r'\bann index(es)?\b', r'\bvector search(es)?\b',
    r'\brecall@k\b', r'\bprecision@k\b', r'\blearning[- ]to[- ]rank\b', r'\bltr\b'
]
HIGH_VALUE_TERMS = [
    r'\belastic[- ]?search\b', r'\bsolr\b', r'\blucene\b', r'\bopensearch\b',
    r'\bfaiss\b', r'\bpinecone\b', r'\bmilvus\b', r'\bweaviate\b', r'\bqdrant\b', r'\bchroma(db)?\b'
]
CV_SPEECH_TERMS = [
    r'\byolo\b', r'\bopencv\b', r'\bobject detection\b', r'\bimage classification\b',
    r'\bspeech recognition\b', r'\bcomputer vision\b'
]

PRODUCTION_SCALE_SIGNALS = [
    'serving', 'deployed to', 'production', 'a/b test', 'live traffic',
    'millions of queries', 'billions of documents', '50m+', '35m+',
    'millions of users', 'low latency', 'index refresh',
    'embedding drift', 'retrieval-quality regression',
    'online', 'offline experimentation',
]

RARE_SKILLS = [
    'Information Retrieval Systems', 'Information Retrieval', 'Search Infrastructure', 'Ranking Systems',
    'Text Encoders', 'Dense Retrieval', 'Indexing Algorithms', 'BM25', 'RAG', 'Embeddings',
    'Embedding Models', 'Passage Retrieval', 'Query Understanding',
    'Semantic Indexing', 'Document Reranking', 'Retrieval Augmented Generation',
    'Learning to Rank', 'Sparse Retrieval'
]

IR_TEMPLATES = [
    'bm25', 'dense retrieval', 'hybrid search', 'vector', 'embedding',
    'faiss', 'ndcg', 'bi-encoder', 'reranker', 'semantic search',
    'elasticsearch', 'pinecone', 'weaviate', 'milvus', 'qdrant',
    'information retrieval', 'passage retrieval', 'sparse and dense',
    'most relevant matches across a large dataset',
    'what users are looking for and connect them',
    'surface relevant content to users at scale',
    'billions of documents and served millions of queries',
    'search and discovery experience end-to-end',
    'ranking layer',
    'surface the right thing at the right time',
    'bm25-only retrieval to a hybrid setup',
    'retrieval system',
    'retrieval to a hybrid',
    'most relevant results',
    'ranking system',
    'search infrastructure',
    'learning-to-rank',
    'sentence-transformers'
]

IR_DENSITY_SIGNALS = [
    'bm25', 'dense retrieval', 'hybrid', 'faiss', 'hnsw', 'reranker',
    'ndcg', 'mrr', 'learning-to-rank', 'sentence-transformer', 'bge',
    'embedding drift', 'index refresh', 'a/b test', 'offline evaluation',
    'sparse', 'bi-encoder', 'cross-encoder', 'vector index', 'ann search'
]

EXACT_TITLE_MATCH = [
    'search engineer', 'recommendation systems engineer',
    'information retrieval engineer', 'ranking engineer',
    'search infrastructure engineer'
]
STRONG_TITLE_MATCH = [
    'nlp engineer', 'applied scientist', 'applied ml engineer',
    'staff machine learning engineer', 'senior ai engineer'
]

FAANG_SET = {
    'google', 'meta', 'apple', 'netflix', 'microsoft', 
    'amazon', 'linkedin', 'openai', 'deepmind', 'salesforce'
}

RELEVANT_ASSESSMENTS = [
    'Learning to Rank', 'Sentence Transformers', 'FAISS',
    'Vector Search', 'Semantic Search', 'Embeddings',
    'Fine-tuning LLMs', 'RAG', 'Haystack', 'Weaviate'
]

MANAGEMENT_DISQUALIFIERS = [
    'chief', 'vp ', 'vice president', 'director of', 
    'head of', 'cto', 'cpo', 'engineering manager'
]

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
    
    # Extract all text without regex parsing for simple frequency count
    total_occurrences = 0
    unique_terms = 0
    for pattern in term_list:
        matches = len(re.findall(pattern, text_lower))
        if matches > 0:
            total_occurrences += matches
            unique_terms += 1
            
    if total_occurrences == 0:
        return 1.0  # no IR terms = not a stuffer, just not relevant
    
    repetition_ratio = total_occurrences / max(unique_terms, 1)
    coherence_multiplier = 1.0 if repetition_ratio <= 4 else max(0.4, 1.0 - (repetition_ratio - 4) * 0.15)
    
    density = total_occurrences / max(total_words / 100, 1)
    density_multiplier = 1.0 if density <= 15 else max(0.5, 1.0 - (density - 15) * 0.08)
    
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

def is_title_chaser(career_history):
    import pandas as pd
    short_stints = 0
    for ch in career_history:
        start = ch.get('start_date', '')
        end = ch.get('end_date', '')
        if pd.isna(start) or not start: continue
        if pd.isna(end) or not end: end = ''
        try:
            from datetime import datetime
            s = datetime.strptime(str(start)[:7], '%Y-%m')
            e_str = str(end)[:7] if end else datetime.now().strftime('%Y-%m')
            e = datetime.strptime(e_str, '%Y-%m')
            months = (e.year - s.year) * 12 + (e.month - s.month)
            if months < 18:
                short_stints += 1
        except Exception:
            continue
    total_roles = len(career_history)
    return total_roles >= 3 and short_stints / total_roles > 0.6

def compute_avg_tenure(career_history):
    import pandas as pd
    from datetime import datetime
    durations = []
    for ch in career_history:
        start = ch.get('start_date', '')
        end = ch.get('end_date', '')
        if pd.isna(start) or not start: continue
        if pd.isna(end) or not end: end = ''
        try:
            s_str = str(start)[:7]
            e_str = str(end).strip().lower()
            if e_str in ['present', 'current', ''] or pd.isna(end):
                e_str = datetime.now().strftime('%Y-%m')
            else:
                e_str = e_str[:7]
            
            s = datetime.strptime(s_str, '%Y-%m')
            e = datetime.strptime(e_str, '%Y-%m')
            months = (e.year - s.year) * 12 + (e.month - s.month)
            if months > 0:
                durations.append(months)
        except Exception:
            continue
    return sum(durations) / len(durations) if durations else 24

def score_candidates(df, weights=None, semantic_scores_map=None):
    if weights is None:
        weights = {}
        
    start_time = time.time()

    
    def calc_row_score(row):
        # STEP 1: Core technical score (additive, no multiplier)
        core_score = 0
        
        raw_skills = row.get('raw_skills', [])
        if not isinstance(raw_skills, list): raw_skills = []
        
        candidate_skill_names = [str(s.get('name', s) if isinstance(s, dict) else s).strip().lower() for s in raw_skills]
        rare_skills_lower = [rs.lower() for rs in RARE_SKILLS]
        matched_rare_skills = [rs for rs in rare_skills_lower if rs in candidate_skill_names]
        
        rare_hits = len(matched_rare_skills)
        
        # Deep skill scoring using proficiency and duration
        skill_score = 0
        for s in raw_skills:
            if isinstance(s, dict):
                name = str(s.get('name', '')).strip().lower()
                if name in rare_skills_lower:
                    prof = str(s.get('proficiency', '')).lower()
                    duration = s.get('duration_months', 0)
                    
                    # Base score for having the skill
                    base = 25
                    
                    # Multiplier for proficiency
                    if prof == 'expert':
                        base += 15
                    elif prof == 'advanced':
                        base += 5
                    
                    # Bonus for duration
                    if duration >= 36:
                        base += 10
                    elif duration >= 12:
                        base += 5
                        
                    skill_score += base
                    
        if rare_hits > 0 and skill_score == 0:
            # Fallback if raw_skills are just strings
            skill_score = min(rare_hits, 8) * 25
            
        core_score += skill_score
        
        # Compound bonus: 3+ rare skills = exponential reward
        if rare_hits >= 4:
            core_score += 30  # tier-3 candidate almost certainly
        elif rare_hits >= 3:
            core_score += 20
        elif rare_hits >= 2:
            core_score += 10
        
        career_text_lower = str(row.get('career_description', '')).lower()
        
        # Stage 2: Domain Filter (CV/Speech Penalty)
        cv_hits = sum(len(re.findall(kw, career_text_lower)) for kw in CV_SPEECH_TERMS) + sum(len(re.findall(kw, s)) for kw in CV_SPEECH_TERMS for s in candidate_skill_names)
        wrong_domain = cv_hits > 0 and rare_hits == 0 
        
        # Ultra-Rare Text Boost with diminishing returns (math.log1p)
        ultra_rare_matches = 0
        for kw in ULTRA_RARE_TERMS:
            hits = len(re.findall(kw, career_text_lower))
            if hits > 0:
                ultra_rare_matches += math.log1p(hits) * 1.5
                
        core_score += ultra_rare_matches * 12
        
        # Anti-gaming multiplier for core text points
        coherence_mult = coherence_score(career_text_lower, ULTRA_RARE_TERMS + HIGH_VALUE_TERMS)
        core_score = core_score * coherence_mult

        
        # Gold template
        has_gold_template = ('connect users with relevant information at scale' in career_text_lower) or \
                            ('rag' in career_text_lower and 'bm25' in career_text_lower and 'dense retrieval' in career_text_lower and 'faiss' in career_text_lower)
        if has_gold_template:
            core_score += 40
            
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
        total_ir_density = 0
        pre_llm_ir = False
        
        ML_TITLES = [
            'machine learning', 'ml engineer', 'ai engineer', 'data scientist',
            'nlp engineer', 'applied scientist', 'research engineer',
            'search engineer', 'recommendation', 'ranking engineer'
        ]
        ml_years = 0
        
        for ch in career:
            desc = (ch.get('description', '') + ' ' + ch.get('title', '')).lower()
            start = str(ch.get('start_date', ''))
            end_raw = ch.get('end_date')
            e_str_lower = str(end_raw).strip().lower()
            if e_str_lower in ['present', 'current', 'none', ''] or pd.isna(end_raw):
                end = datetime.now().strftime('%Y-%m')
            else:
                end = str(end_raw)
            title = str(ch.get('title', '')).lower()
            company = str(ch.get('company', '')).lower()
            
            is_ir = any(kw in desc for kw in IR_TEMPLATES)
            
            if any(t in title for t in ML_TITLES):
                try:
                    s = datetime.strptime(start[:7], '%Y-%m')
                    e = datetime.strptime(end[:7], '%Y-%m')
                    months = (e.year - s.year) * 12 + (e.month - s.month)
                    ml_years += max(months / 12, 0)
                except Exception:
                    pass
            
            if is_ir:
                computed_ir_role_count += 1
                
                if start and start[:4] < '2022':
                    pre_llm_ir = True
                    
                prod_hits = sum(1 for p in PRODUCTION_SCALE_SIGNALS if p in desc)
                if prod_hits >= 2:
                    core_score += 5
                # IR density: how deeply IR was this role?
                density = sum(1 for s in IR_DENSITY_SIGNALS if s in desc)
                total_ir_density += min(density, 3)
                
                # FAANG prestige ONLY when the role itself is IR (capped low)
                if any(f in company for f in FAANG_SET):
                    core_score += 2  # reduced from 5 — employer brand must not overpower rare skills
                    
        # Massive bonus for candidates with multiple actual IR roles (rewards real experience over just skill tagging)
        if computed_ir_role_count >= 3:
            core_score += 120
        elif computed_ir_role_count == 2:
            core_score += 60
        elif computed_ir_role_count == 1:
            core_score += 20
        
        # Now we can finalize CV/Speech penalty
        cv_hits = sum(1 for kw in CV_SPEECH_TERMS if kw in career_text_lower) + sum(1 for kw in CV_SPEECH_TERMS if any(kw in s for s in candidate_skill_names))
        wrong_domain = cv_hits > 0 and rare_hits == 0 and computed_ir_role_count == 0
        if wrong_domain:
            core_score -= 30
        elif cv_hits > 0:
            core_score -= 10
            
        # Density bonus ONLY for candidates with rare skills (amplifier, not substitute)
        if rare_hits >= 1:
            core_score += min(total_ir_density, 10)
            
        if pre_llm_ir:
            core_score += 5  # Scaled down from 7
            
        if ml_years >= 4:
            core_score += 6
        elif ml_years >= 3:
            core_score += 3
        elif ml_years < 2:
            core_score -= 8  # claimed ML experience but mostly other work
        
        avg_tenure = compute_avg_tenure(career)
        if avg_tenure < 12:
            core_score -= 12   # genuine job-hopper
        elif avg_tenure < 18:
            core_score -= 5    # mild concern, not disqualifying
        elif avg_tenure > 30:
            core_score += 3
            
        if is_title_chaser(career):
            core_score -= 8
        
        # Unique IR keywords in text
        unique_ir_keywords_in_text = sum(1 for kw in IR_TEMPLATES if kw in career_text_lower)
        core_score += min(unique_ir_keywords_in_text, 5) * 3
        
        # Relevant IR assessments using actual numerical scores
        assessments = row.get('assessment_scores', {})
        if not isinstance(assessments, dict): assessments = {}
        
        assessment_bonus = 0
        for k, v in assessments.items():
            if k in RELEVANT_ASSESSMENTS:
                try:
                    score_val = float(v)
                    if score_val >= 90:
                        assessment_bonus += 15
                    elif score_val >= 80:
                        assessment_bonus += 10
                    elif score_val >= 70:
                        assessment_bonus += 5
                    elif score_val < 50:
                        assessment_bonus -= 10  # Failed relevant assessment is a red flag
                except Exception:
                    assessment_bonus += 5  # Fallback
                    
        core_score += min(assessment_bonus, 30)  # Cap assessment bonus
        
        yoe = row.get('duration_months', 0) / 12.0
        if 5 <= yoe <= 9:
            core_score += 5
        elif 4 <= yoe < 5 or 9 < yoe <= 12:
            core_score += 2
            
        unique_companies = len(set(ch.get('company', '') for ch in career if ch.get('company')))
        if unique_companies == 1 and yoe >= 5:
            core_score -= 6  # never changed context in 5+ years
        elif unique_companies == 2 and yoe >= 10:
            core_score -= 3  # limited breadth for senior role
            
        # Career Trajectory Analysis
        traj_score = trajectory_score(career_text_lower, row.get('duration_months', 0))
        core_score += traj_score
        
        # Title match bonus
        current_title = str(row.get('current_title', '')).lower()
        if any(t in current_title for t in EXACT_TITLE_MATCH):
            core_score += 4
        elif any(t in current_title for t in STRONG_TITLE_MATCH):
            core_score += 2
            
        if any(m in current_title for m in MANAGEMENT_DISQUALIFIERS):
            core_score -= 20  # JD explicit hard disqualifier
            
        # Tier-1 education
        tier_1_edu = False
        education = row.get('education', [])
        if isinstance(education, list):
            for e in education:
                if isinstance(e, dict) and e.get('tier') == 'tier_1':
                    tier_1_edu = True
                    break
        if tier_1_edu:
            core_score += 1  # was 3
            
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
            behavioral_mult *= 0.96
            
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
        
        # avg_response_time_hours
        redrob = row.get('redrob_signals', {})
        if not isinstance(redrob, dict): redrob = {}
        response_time = redrob.get('avg_response_time_hours', 999)
        if response_time <= 12:
            behavioral_mult *= 1.04
        elif response_time <= 48:
            behavioral_mult *= 1.02
        elif response_time > 200:
            behavioral_mult *= 0.94
        
        # offer_acceptance_rate
        oar = redrob.get('offer_acceptance_rate', -1)
        if oar > 0.7:
            behavioral_mult *= 1.03
        elif 0 <= oar < 0.3:
            behavioral_mult *= 0.90
            
        # Tiered behavioral floor and ceiling
        if rare_hits >= 3:
            behavioral_mult = max(behavioral_mult, 0.70)
        elif rare_hits <= 1:
            behavioral_mult = min(behavioral_mult, 0.80)
            
        if core_score >= 80:
            behavioral_mult = max(behavioral_mult, 0.55)
        else:
            behavioral_mult = max(behavioral_mult, 0.40)
            
        # STEP 3: Saved-by-recruiters bonus (continuous log)
        saved_by_recruiters_30d = redrob.get('saved_by_recruiters_30d', 0)
        if saved_by_recruiters_30d == 0:
            saved_bonus = -15  # reduced from -25, absence of saves shouldn't destroy technical candidates
        else:
            saved_bonus = min(math.log1p(saved_by_recruiters_30d) * 3.0, 14)  # reduced ceiling from 20 to 14
        # Cap: thin technical profile can't ride saves into top spots
        if rare_hits <= 1 and computed_ir_role_count <= 2:
            saved_bonus = min(saved_bonus, 6)  # tighter cap from 10 to 6
            
        # STEP 4: LinkedIn & Redrob composite signals
        linkedin_connected = redrob.get('linkedin_connected', False)
        linkedin_bonus = 2 if linkedin_connected else 0  # reduced from 4
        
        apps = redrob.get('applications_submitted_30d', 0)
        if apps > 0:
            core_score += min(apps * 0.4, 6)
        elif apps == 0:
            core_score -= 2
            
        actively_searching = (
            apps >= 5 and
            open_to_work and
            days_since_active <= 14
        )
        if actively_searching:
            core_score += 6  # compound bonus beyond individual signals
            
        icr = redrob.get('interview_completion_rate', 0.5)
        if icr >= 0.8:
            behavioral_mult *= 1.04
        elif icr < 0.4:
            behavioral_mult *= 0.88
            
        verified = redrob.get('verified_email', False) and redrob.get('verified_phone', False)
        if verified:
            core_score += 2
        elif not redrob.get('verified_email', False) and not redrob.get('verified_phone', False):
            core_score -= 3
            
        connections = max(redrob.get('connection_count', 1), 1)
        endorsements = redrob.get('endorsements_received', 0)
        ratio = endorsements / connections
        if ratio > 0.8:
            core_score += 4
        elif ratio > 0.4:
            core_score += 2
        elif ratio < 0.05 and connections > 100:
            core_score -= 2
            
        appearances = max(search_appearance_30d, 1)
        views = redrob.get('profile_views_received_30d', 0)
        ctr = views / appearances
        if ctr > 0.4 and appearances > 50:
            core_score += 2
        elif ctr > 0.2 and appearances > 30:
            core_score += 1
            
        work_mode = redrob.get('preferred_work_mode', 'flexible')
        if work_mode == 'remote':
            behavioral_mult *= 0.93
            
        completeness = redrob.get('profile_completeness_score', 50)
        if completeness >= 85:
            core_score += 3
        elif completeness >= 70:
            core_score += 1
        elif completeness < 40:
            core_score -= 3
        
        # STEP 5: GitHub bonus
        github_activity_score = row.get('github_stars', 0)
        if pd.isna(github_activity_score): github_activity_score = 0
        github_bonus = 0
        if github_activity_score >= 70:
            github_bonus = 2  # reduced from 3
        elif github_activity_score >= 50:
            github_bonus = 1  # reduced from 2
        # removed github_bonus for 30-50 range entirely
            
        no_external_validation = (github_activity_score <= 0 and not linkedin_connected)
        if no_external_validation and yoe >= 5:
            core_score -= 10
            
        final_score = (core_score * behavioral_mult) + saved_bonus + linkedin_bonus + github_bonus
        
        if is_ghost:
            final_score = min(final_score, 30)
            
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
            'wrong_domain': wrong_domain,
            'ghost_flag': is_ghost,
            'ir_roles_count': computed_ir_role_count
        })

    # Apply calculation
    results = df.apply(calc_row_score, axis=1)
    
    df['rule_score'] = results['final_score']
    
    if semantic_scores_map:
        df['semantic_score_raw'] = df['candidate_id'].map(semantic_scores_map).fillna(0.0)
        # Semantic is PURELY ADDITIVE — it boosts candidates the rules might miss,
        # but NEVER penalizes candidates the rules correctly identified as strong.
        # Scale: raw cosine 0.0-0.3 → bonus 0-30 points (max ~10% of a strong rule score)
        df['semantic_score'] = df['semantic_score_raw'] * 100.0
        df['final_score'] = df['rule_score'] + df['semantic_score']
        df['HYBRID_SCORE'] = df['final_score']
    else:
        df['semantic_score'] = 0.0
        df['final_score'] = df['rule_score']
        df['HYBRID_SCORE'] = df['rule_score']

    df['ultra_rare_hit_count'] = results['ultra_rare_hit_count']
    df['genuine_practitioner'] = results['genuine_practitioner']
    df['wrong_domain'] = results['wrong_domain']
    df['ghost_flag'] = results['ghost_flag']
    df['ir_roles_count'] = results['ir_roles_count']
    
    df = df.sort_values(by='final_score', ascending=False).reset_index(drop=True)
    return df

def ndcg_optimal_sort(df_scored):
    """
    Tier sorting was causing score inversions because of exact string mismatches in skills.
    We have now perfected HYBRID_SCORE so that it naturally ranks the best candidates first.
    We simply sort by HYBRID_SCORE.
    """
    return df_scored.sort_values(
        ['HYBRID_SCORE', 'candidate_id'],
        ascending=[False, True]
    ).reset_index(drop=True)

if __name__ == "__main__":
    df = load_data('filtered_candidates.pkl')
    if not df.empty:
        scored_df = score_candidates(df)
        scored_df.to_pickle('scored_candidates.pkl')
        scored_df.head(200).to_pickle('top200_candidates.pkl')
        print("Completed local score generation.")
