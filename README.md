# Recto — Ranking candidates the way a great recruiter would

Recto is an advanced, multi-layered pipeline designed to identify the absolute best Information Retrieval (IR) and Search Engineering candidates from a dataset of 100,000+ profiles. 

It was built strictly under the hackathon constraints: **Zero external APIs (no LLMs), < 5 minutes execution time, <= 16GB RAM, pure CPU execution.**

## Architecture

```text
[ Raw Data: 100,000 Candidates ]
         │
         ▼
 ┌─────────────────────┐
 │ PHASE 1: Data Load  │──▶ Fallback handling, JSON schema flattening
 └─────────────────────┘
         │
         ▼
 ┌─────────────────────┐
 │ PHASE 2: Hard Kills │──▶ Salary traps, Honeypots, Keyword-stuffing penalties
 └─────────────────────┘
         │
         ▼
 ┌─────────────────────┐
 │ PHASE 3: Core Score │──▶ Rare Skills (BM25, Pinecone), Trajectory, IR Density
 └─────────────────────┘
         │
         ▼
 ┌─────────────────────┐
 │ PHASE 4: Semantic   │──▶ TF-IDF Sparse Similarity Matrix (Cosine Similarity)
 └─────────────────────┘
         │
         ▼
 ┌─────────────────────┐
 │ PHASE 5: Rank Sort  │──▶ Deterministic Hybrid Score Sort (Zero Inversions)
 └─────────────────────┘
         │
         ▼
[ Final Top 100 Shortlist ]
```

## Quick Start (Evaluation Command)

As required by the submission spec, the pipeline runs entirely locally with a single command:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full end-to-end pipeline
python main.py --candidates /path/to/candidates.jsonl --output results/
```
*(Note: the `--jd` parameter is optional and defaults to the internal deterministic IR JD logic. Do not worry if you don't supply it.)*

## Key Innovations

* **Anti-Gaming Architecture**: Uses Coherence Ratios to penalize keyword stuffers. If someone dumps 15 buzzwords into a 1-sentence career description, their score collapses.
* **The "3+ IR Role" Boost**: A systemic fix that prevents junior candidates with keyword-stuffed "skills" arrays from outscoring 10-year veterans who have actually built IR systems at scale.
* **Semantic Safety Net**: A pre-computed TF-IDF Cosine Similarity engine that acts purely as an additive boost to catch candidates the heuristics might have missed. 
* **Zero Score Inversions**: We enforce a strict deterministic mathematical rank sort, meaning `score_at_rank_1 >= score_at_rank_2` natively, with no artificial tiering required.

## Tech Stack
* **Python 3.11+**
* **Pandas / Numpy** (for lightning-fast vectorized scoring)
* **Scikit-Learn** (for TF-IDF Sparse Matrices and Cosine Similarity)
* **Rich** (for beautiful CLI UX)

---
