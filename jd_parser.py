import os
import json
import time
import hashlib
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Hard-coded known facts for Sherlock hackathon project
KNOWN_REQUIRED = [
    'production embeddings', 'sentence-transformers', 'BGE', 'E5', 'vector database', 
    'Pinecone', 'Weaviate', 'Qdrant', 'Milvus', 'FAISS', 'Elasticsearch', 
    'ranking evaluation', 'NDCG', 'MRR', 'MAP', 'hybrid search', 'BM25', 
    'dense retrieval', 'neural reranking'
]

KNOWN_NICE_TO_HAVE = [
    'BERT', 'transformers', 'LLM', 'Python', 'MLOps', 'A/B testing', 
    'product company experience'
]

KNOWN_DISQUALIFIERS = [
    'TCS', 'Infosys', 'Wipro', 'Accenture', 'Cognizant', 'Capgemini', 
    'YOLO', 'OpenCV', 'speech recognition', 'computer vision', 'robotics', 
    'image classification'
]

ULTRA_RARE_SIGNALS = [
    'bm25', 'ndcg', 'mrr', 'embedding', 'bi-encoder', 'cross-encoder', 
    'dense retrieval', 'sparse retrieval', 'hybrid retrieval', 'reranking', 
    'ColBERT', 'DPR', 'SPLADE', 'HNSW', 'ANN index', 'vector search', 
    'recall@k', 'precision@k'
]

# Rate Limiting & Cache Config
MAX_RETRIES = 4
BACKOFF_TIMES = [4, 8, 16, 32]
CACHE_FILE = 'jd_cache.json'
REQUESTS_PER_MINUTE_LIMIT = 14

_last_request_time = 0

def enforce_rate_limit():
    global _last_request_time
    min_interval = 60.0 / REQUESTS_PER_MINUTE_LIMIT
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < min_interval:
        time.sleep(min_interval - elapsed)
    _last_request_time = time.time()

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_cache(cache):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, indent=2)

def get_jd_hash(jd_text):
    return hashlib.sha256(jd_text.encode('utf-8')).hexdigest()

def merge_signals(gemini_output):
    """
    3. Merge the hard-coded known facts with Gemini's parsed output.
    Hard-coded facts take priority and ensure baseline intelligence.
    """
    gemini_core = gemini_output.get("core_skills_needed", [])
    gemini_explicit = gemini_output.get("explicit_disqualifiers", [])
    gemini_implicit = gemini_output.get("implicit_disqualifiers", [])
    
    # Merge and deduplicate, prioritizing known facts
    merged_required = list(dict.fromkeys(KNOWN_REQUIRED + gemini_core))
    merged_disqualifiers = list(dict.fromkeys(KNOWN_DISQUALIFIERS + gemini_explicit + gemini_implicit))
    
    merged = {
        "seniority": gemini_output.get("seniority", "N/A"),
        "domain": gemini_output.get("domain", "N/A"),
        "is_product_company": gemini_output.get("is_product_company", False),
        "location_preference": gemini_output.get("location_preference", "N/A"),
        "yoe_min": gemini_output.get("yoe_min", None),
        "yoe_max": gemini_output.get("yoe_max", None),
        "required_skills": merged_required,
        "nice_to_have_skills": KNOWN_NICE_TO_HAVE,
        "disqualifiers": merged_disqualifiers,
        "ultra_rare_signals": ULTRA_RARE_SIGNALS
    }
    return merged

def parse_jd(jd_text: str) -> dict:
    # 5. Rate limiting & Caching
    cache = load_cache()
    jd_hash = get_jd_hash(jd_text)
    
    if jd_hash in cache:
        print("Using cached result from jd_cache.json")
        gemini_raw = cache[jd_hash]
    else:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("Warning: GEMINI_API_KEY not found in environment.")
            
        genai.configure(api_key=api_key)
        
        system_instruction = (
            "You are a technical recruiter expert in IR/Search/NLP roles. "
            "Analyze the JD and return ONLY raw JSON, no markdown."
        )
        
        user_prompt = f"""Extract from this JD:
{{
  "core_skills_needed": ["..."],
  "seniority": "junior|mid|senior|staff",
  "domain": "search|recommendations|nlp|cv|general",
  "is_product_company": true/false,
  "location_preference": "...",
  "yoe_min": 0,
  "yoe_max": 0,
  "explicit_disqualifiers": ["..."],
  "implicit_disqualifiers": ["..."]
}}
JD: {jd_text}
"""
        model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=system_instruction,
            generation_config={"response_mime_type": "application/json"}
        )
        
        gemini_raw = {}
        # Exponential backoff loop
        for attempt in range(MAX_RETRIES + 1):
            try:
                enforce_rate_limit()
                response = model.generate_content(user_prompt)
                result_text = response.text.strip()
                
                # Cleanup markdown just in case
                if result_text.startswith("```json"): result_text = result_text[7:]
                if result_text.startswith("```"): result_text = result_text[3:]
                if result_text.endswith("```"): result_text = result_text[:-3]
                
                gemini_raw = json.loads(result_text.strip())
                
                # Cache raw result
                cache[jd_hash] = gemini_raw
                save_cache(cache)
                break
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "Quota" in error_msg:
                    if attempt < MAX_RETRIES:
                        wait_time = BACKOFF_TIMES[attempt]
                        print(f"Rate limit hit. Retrying in {wait_time}s... (Attempt {attempt+1}/{MAX_RETRIES})")
                        time.sleep(wait_time)
                    else:
                        print("Max retries reached.")
                else:
                    print(f"API/Parsing Error: {e}")
                    break
    
    # 3. Merge facts
    merged_signals = merge_signals(gemini_raw)
    
    # 4. Save result to jd_signals.json
    with open('jd_signals.json', 'w', encoding='utf-8') as f:
        json.dump(merged_signals, f, indent=2)
        
    return merged_signals

def print_summary_table(merged):
    """6. Print a clean 'JD Signal Summary' table."""
    def wrap_text(text, width=74):
        words = text.split(', ')
        lines, current = [], []
        for word in words:
            if len(", ".join(current + [word])) > width:
                lines.append(", ".join(current) + ",")
                current = [word]
            else:
                current.append(word)
        if current:
            lines.append(", ".join(current))
        return lines

    print("+" + "-"*78 + "+")
    print(f"| {'JD SIGNAL SUMMARY':^76} |")
    print("+" + "-"*78 + "+")
    
    yoe = f"{merged.get('yoe_min', '?')} - {merged.get('yoe_max', '?')}"
    print(f"| Seniority: {str(merged['seniority']).upper():<25} | Domain: {str(merged['domain']).upper():<29} |")
    print(f"| YOE: {yoe:<31} | Product Co: {str(merged['is_product_company']):<25} |")
    print("+" + "-"*78 + "+")
    
    sections = [
        ("REQUIRED SKILLS (Hard-coded + Gemini):", merged['required_skills']),
        ("NICE-TO-HAVE SKILLS (Hard-coded):", merged['nice_to_have_skills']),
        ("DISQUALIFIERS (Hard-coded + Gemini):", merged['disqualifiers']),
        ("ULTRA RARE SIGNALS (Tier 4/5 markers):", merged['ultra_rare_signals'])
    ]
    
    for title, items in sections:
        print(f"| {title:<76} |")
        print("+" + "-"*78 + "+")
        for line in wrap_text(", ".join(items)):
            print(f"| {line:<76} |")
        print("+" + "-"*78 + "+")
    print("\n -> Complete signals saved to 'jd_signals.json'")

if __name__ == "__main__":
    load_dotenv()
    
    # Sample run with the specific Redrob role requirements
    sample_jd = (
        "We are looking for a Senior AI Engineer at Redrob (founding team) in Pune/Noida "
        "(hybrid). Requires 5-9 years of experience. You will build and scale our hybrid "
        "search, embedding models, and ranking systems to match candidates with roles."
    )
    
    print("Parsing JD for Sherlock Ranker...")
    merged = parse_jd(sample_jd)
    print_summary_table(merged)
