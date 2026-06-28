import pandas as pd
import numpy as np
import re
import pickle
import os
import time

def load_data(filepath='filtered_candidates.pkl'):
    """
    Loads the filtered dataset. If not found, generates a dummy dataset 
    to demonstrate the exact logic running fast.
    """
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found. Generating dummy dataset for testing...")
        return generate_dummy_data()
    
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def generate_dummy_data(n=100_000):
    """Generates 100k rows to test performance."""
    np.random.seed(42)
    df = pd.DataFrame({
        'candidate_id': range(1, n+1),
        'name': [f"Candidate_{i}" for i in range(1, n+1)],
        'career_description': np.random.choice([
            "I use BGE and FAISS for dense retrieval and reranker models.",
            "Worked on YOLO and CNN for image classification. OpenCV expert.",
            "Information retrieval using Elasticsearch, BM25, and Solr.",
            "I have experience in LLM, ColBERT, SPLADE, and hybrid search with NDCG evaluation.",
            "Sales manager with good communication."
        ], n),
        'rare_skills': [
            [['FAISS', 'LLM'], ['YOLO', 'OpenCV'], ['ColBERT', 'SPLADE', 'LTR'], [], ['BM25']][i]
            for i in np.random.randint(0, 5, n)
        ],
        'assessment_scores': [
            [
                {'LTR': 80, 'SentTrans': 75, 'YOLO': 20},
                {'LTR': 40, 'YOLO': 90, 'CNN': 85},
                {'LTR': 85, 'SentTrans': 88},
                {}
            ][i]
            for i in np.random.randint(0, 4, n)
        ],
        'active_days_since_last_login': np.random.randint(1, 120, n),
        'open_to_work': np.random.choice([True, False], n),
        'response_rate': np.random.uniform(0.1, 1.0, n),
        'notice_period_days': np.random.choice([15, 30, 45, 60, 90, 120], n),
        'github_stars': np.random.randint(0, 200, n),
        'ir_roles_count': np.random.randint(0, 6, n),
        'duration_months': np.random.randint(10, 240, n),
        'penalty_score': np.random.choice([0, -20], n, p=[0.9, 0.1])
    })
    return df

def score_candidates(df):
    start_time = time.time()
    print(f"Scoring {len(df)} candidates...")
    
    # ==========================================
    # LAYER 2: Core IR Score
    # ==========================================
    
    # 1. Rare skill bonus (+8 pts each)
    rare_keywords = set([x.lower() for x in [
        'FAISS','LLM','reranking','ColBERT','DPR','SPLADE','LambdaMART','BM25',
        'LTR','dense retrieval','bi-encoder','cross-encoder','HNSW','ANN','vector search'
    ]])
    def count_rare(skills):
        if not isinstance(skills, list): return 0
        return sum(1 for s in skills if str(s).lower() in rare_keywords)
    
    # Vectorized via apply (very fast for lists)
    df['rare_skill_bonus'] = df['rare_skills'].apply(count_rare) * 8
    
    # 2. Gold template bonus (+25 pts for 3+)
    # Pre-compiling regex makes this lightning fast
    gold_terms = ['BGE','FAISS','LLM','reranker','dense','sparse','hybrid','recall','precision','NDCG']
    gold_pattern = re.compile(r'(?i)\b(' + '|'.join(map(re.escape, gold_terms)) + r')\b')
    def count_unique_gold(text):
        if not isinstance(text, str): return 0
        return len(set(match.lower() for match in gold_pattern.findall(text)))
        
    gold_counts = df['career_description'].apply(count_unique_gold)
    df['gold_template_bonus'] = np.where(gold_counts >= 3, 25, 0)
    
    # 3. IR Roles Score (capped at 60)
    df['ir_roles_score'] = (df['ir_roles_count'].fillna(0) * 15).clip(upper=60)
    
    # 4. Unique IR Keywords Count (capped at 30)
    ir_terms = [
        'elasticsearch', 'solr', 'lucene', 'opensearch', 'bm25', 'tf-idf', 'vector search', 
        'ann', 'hnsw', 'faiss', 'pinecone', 'milvus', 'weaviate', 'qdrant', 'chroma', 
        'learning-to-rank', 'ltr', 'lambdamart', 'xgboost', 'lightgbm', 'catboost', 'ranklib', 
        'bert', 'sentence-transformers', 'bi-encoder', 'cross-encoder', 'colbert', 'dpr', 
        'splade', 'bge', 'e5', 'instructor', 'cohere', 'voyage', 'jina', 'reranking', 
        'semantic search', 'lexical search', 'hybrid search', 'rag'
    ]
    ir_pattern = re.compile(r'(?i)\b(' + '|'.join(map(re.escape, ir_terms)) + r')\b')
    def count_unique_ir(text):
        if not isinstance(text, str): return 0
        return len(set(match.lower() for match in ir_pattern.findall(text)))
        
    df['unique_ir_keywords_count'] = df['career_description'].apply(count_unique_ir)
    df['unique_ir_score'] = (df['unique_ir_keywords_count'] * 2).clip(upper=30)
    
    # 5. Assessment Bonus & CV Trap Penalty
    def get_assessment_bonus(scores):
        if not isinstance(scores, dict): return 0
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
        if max_cv > 0 and max_cv >= max_ir:
            bonus -= 5
        return bonus
        
    df['assessment_bonus'] = df['assessment_scores'].apply(get_assessment_bonus)
    
    # Fetch penalty score from Layer 1 if it exists
    penalty = df.get('penalty_score', 0)
    
    # Sum Layer 2
    df['layer2_score'] = (
        df['rare_skill_bonus'] + 
        df['gold_template_bonus'] + 
        df['ir_roles_score'] + 
        df['unique_ir_score'] + 
        df['assessment_bonus'] + 
        penalty
    )
    
    # ==========================================
    # LAYER 3: Behavioral Multiplier
    # ==========================================
    
    # Vectorized operations for multiplier components
    active = df['active_days_since_last_login'].fillna(999)
    m_act = np.where(active < 30, 1.0, 
            np.where(active <= 60, 0.85, 
            np.where(active <= 90, 0.70, 0.50)))
            
    m_otw = np.where(df['open_to_work'] == True, 1.0, 0.9)
    
    resp = df['response_rate']
    m_resp = np.where(resp.isnull(), 0.9,
             np.where(resp > 0.7, 1.0,
             np.where(resp >= 0.5, 0.95, 0.75)))
             
    notice = df['notice_period_days'].fillna(30)
    m_not = np.where(notice < 30, 0.92,
            np.where(notice <= 60, 1.0,
            np.where(notice <= 90, 0.85, 0.75)))
            
    stars = df['github_stars'].fillna(0)
    m_git = np.where(stars > 50, 1.05, 1.0)
    
    df['behavioral_multiplier'] = m_act * m_otw * m_resp * m_not * m_git
    
    # Note on Overqualified candidates: 
    # Because there are NO negative penalties applied natively to duration_months, 
    # candidates with >192 months who are "deep" simply retain their high Layer 2 scores normally!
    
    # ==========================================
    # FINAL SCORE & TWIN TIEBREAKER
    # ==========================================
    df['final_score'] = df['layer2_score'] * df['behavioral_multiplier']
    
    # Tiebreaker logic: Score within 0.5 points uses behavioral_multiplier as tiebreaker
    # We round score to nearest 0.5 to create bins for exact behavioral sorting within the bin
    df['score_bin'] = (df['final_score'] * 2).round() / 2
    
    df = df.sort_values(
        by=['score_bin', 'behavioral_multiplier', 'final_score'], 
        ascending=[False, False, False]
    ).reset_index(drop=True)
    
    df = df.drop(columns=['score_bin'])
    
    duration = time.time() - start_time
    print(f"Finished scoring in {duration:.2f} seconds.\n")
    return df

def main():
    df = load_data('filtered_candidates.pkl')
    scored_df = score_candidates(df)
    
    # Save outputs
    scored_df.to_pickle('scored_candidates.pkl')
    print("-> Saved FULL scored list to 'scored_candidates.pkl'")
    
    top200 = scored_df.head(200)
    top200.to_pickle('top200_candidates.pkl')
    print("-> Saved TOP 200 to 'top200_candidates.pkl'\n")
    
    # Print Distribution Stats
    print("=== FINAL SCORE DISTRIBUTION ===")
    print(scored_df['final_score'].describe().round(2))
    
    # Print Top 10 Summary
    print("\n=== TOP 10 CANDIDATES SUMMARY ===")
    cols_to_print = ['candidate_id', 'name', 'layer2_score', 'behavioral_multiplier', 'final_score']
    print(scored_df[cols_to_print].head(10).to_string(index=False))
    
    # Print Separation Metrics (Top 10 vs Rank 11-30)
    print("\n=== SEPARATION: TOP 10 vs TOP 30 ===")
    if len(scored_df) >= 30:
        top10_mean = scored_df['final_score'].head(10).mean()
        top30_mean = scored_df['final_score'].iloc[10:30].mean()
        diff = top10_mean - top30_mean
        print(f"Avg Score Top 10: {top10_mean:.2f}")
        print(f"Avg Score Rank 11-30: {top30_mean:.2f}")
        print(f"Difference: {diff:.2f} pts")
        
        # Dive deeper into IR Depth
        top10_ir = scored_df['unique_ir_keywords_count'].head(10).mean()
        top30_ir = scored_df['unique_ir_keywords_count'].iloc[10:30].mean()
        print(f"Avg Unique IR Keywords (Top 10): {top10_ir:.1f}  |  (Rank 11-30): {top30_ir:.1f}")

if __name__ == "__main__":
    main()
