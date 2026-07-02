import os
import time
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rich.console import Console

console = Console()

def precompute_semantic_scores(input_pkl='results/filtered_candidates.pkl', output_pkl='results/semantic_scores.pkl'):
    console.print("[cyan]Loading filtered candidates...[/cyan]")
    if not os.path.exists(input_pkl):
        raise FileNotFoundError(f"Input file {input_pkl} not found. Run data_loader first.")
        
    df = pd.read_pickle(input_pkl)
    console.print(f"[green]Loaded {len(df)} candidates for semantic scoring.[/green]")
    
    # Construct the ideal query for this specific hackathon job description
    target_query = (
        "Information Retrieval Engineer Senior Machine Learning Engineer Search Infrastructure "
        "Ranking Systems Dense Retrieval Sparse Retrieval BM25 TF-IDF "
        "vector search Learning to Rank LTR sentence transformers bi-encoders cross-encoders "
        "Elasticsearch FAISS Pinecone Milvus Weaviate Qdrant "
        "Connecting users relevant information scale billions documents "
        "discovery algorithm matching engine recommendation core query understanding"
    )
    
    # Prepare candidate text blobs
    console.print("[cyan]Preparing candidate text blobs...[/cyan]")
    candidate_texts = []
    candidate_ids = []
    
    for _, row in df.iterrows():
        title = str(row.get('title', ''))
        skills = " ".join([str(s.get('name', s) if isinstance(s, dict) else s) for s in row.get('raw_skills', [])])
        career = str(row.get('career_description', ''))
        
        blob = f"{title} {skills} {career}"
        candidate_texts.append(blob)
        candidate_ids.append(row['candidate_id'])
        
    console.print(f"[cyan]Building TF-IDF Sparse Embedding Matrix for {len(candidate_texts)} candidates...[/cyan]")
    t0 = time.time()
    
    # Use n-grams to capture semantic phrases (e.g. "information retrieval")
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=25000)
    
    # Fit on the candidate corpus and transform
    candidate_vectors = vectorizer.fit_transform(candidate_texts)
    query_vector = vectorizer.transform([target_query])
    
    console.print(f"[green]Matrix built in {time.time() - t0:.1f} seconds.[/green]")
    
    console.print("[cyan]Computing Cosine Similarities...[/cyan]")
    t0 = time.time()
    similarities = cosine_similarity(candidate_vectors, query_vector).flatten()
    console.print(f"[green]Similarity computed in {time.time() - t0:.1f} seconds.[/green]")
    
    # Save the scores to a dictionary map
    scores_dict = {cid: float(score) for cid, score in zip(candidate_ids, similarities)}
    
    # Save to disk
    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    with open(output_pkl, 'wb') as f:
        pickle.dump(scores_dict, f)
        
    console.print(f"[bold green]Successfully saved {len(scores_dict)} semantic scores to {output_pkl}.[/bold green]")
    
    console.print("[bold]Top 5 Semantic Scores:[/bold]")
    top_5 = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)[:5]
    for cid, score in top_5:
        name = df[df['candidate_id'] == cid].iloc[0].get('name', '?')
        console.print(f"  [yellow]{cid}[/yellow] ({name}): {score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Precompute Semantic Scores")
    parser.add_argument("--input", type=str, default="results/filtered_candidates.pkl", help="Path to filtered_candidates.pkl")
    parser.add_argument("--output", type=str, default="results/semantic_scores.pkl", help="Path to output semantic_scores.pkl")
    args = parser.parse_args()
    
    precompute_semantic_scores(input_pkl=args.input, output_pkl=args.output)
