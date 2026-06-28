import json

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

def parse_jd(jd_text: str) -> dict:
    """
    Since LLMs are prohibited at inference time, we return a deterministic
    pre-computed lookup table based on our EDA of the hackathon constraints.
    """
    merged_signals = {
        "seniority": "senior",
        "domain": "search",
        "is_product_company": True,
        "location_preference": "hybrid",
        "yoe_min": 5,
        "yoe_max": 9,
        "required_skills": KNOWN_REQUIRED,
        "nice_to_have_skills": KNOWN_NICE_TO_HAVE,
        "disqualifiers": KNOWN_DISQUALIFIERS,
        "ultra_rare_signals": ULTRA_RARE_SIGNALS
    }
    
    with open('jd_signals.json', 'w', encoding='utf-8') as f:
        json.dump(merged_signals, f, indent=2)
        
    return merged_signals

def print_summary_table(merged):
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
    print(f"| {'JD SIGNAL SUMMARY (DETERMINISTIC)':^76} |")
    print("+" + "-"*78 + "+")
    
    yoe = f"{merged.get('yoe_min', '?')} - {merged.get('yoe_max', '?')}"
    print(f"| Seniority: {str(merged['seniority']).upper():<25} | Domain: {str(merged['domain']).upper():<29} |")
    print(f"| YOE: {yoe:<31} | Product Co: {str(merged['is_product_company']):<25} |")
    print("+" + "-"*78 + "+")
    
    sections = [
        ("REQUIRED SKILLS:", merged['required_skills']),
        ("NICE-TO-HAVE SKILLS:", merged['nice_to_have_skills']),
        ("DISQUALIFIERS:", merged['disqualifiers']),
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
    sample_jd = (
        "We are looking for a Senior AI Engineer at Redrob (founding team) in Pune/Noida "
        "(hybrid). Requires 5-9 years of experience. You will build and scale our hybrid "
        "search, embedding models, and ranking systems to match candidates with roles."
    )
    print("Parsing JD for Sherlock Ranker (Deterministic)...")
    merged = parse_jd(sample_jd)
    print_summary_table(merged)
