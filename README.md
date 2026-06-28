# Recto — Ranking candidates the way a great recruiter would

Recto is an AI-powered candidate ranking system built to process huge datasets and surgically extract the perfect hires for complex technical roles (specifically Information Retrieval and Search Engineering). 

## Architecture

```text
[ Raw Data: 100k Candidates ]
         │
         ▼
 ┌─────────────────────┐
 │ PHASE 1: Data Load  │──▶ Fallback handling, JSON/CSV parsing
 └─────────────────────┘
         │
         ▼
 ┌─────────────────────┐
 │ PHASE 2: Hard Kills │──▶ Salary traps, Honeypots, Template Noise
 └─────────────────────┘
         │
         ▼
 ┌─────────────────────┐
 │ PHASE 3: Rule Score │──▶ Rare Skills, Gold Patterns, CV Traps
 └─────────────────────┘
         │
         ▼
 ┌─────────────────────┐
 │ PHASE 4: Multiplier │──▶ Behavioral signals, Tiebreaker logic
 └─────────────────────┘
         │
         ▼
 ┌─────────────────────┐
 │ PHASE 5: Reranker   │──▶ Gemini 2.5 Flash Semantic AI (Batched)
 └─────────────────────┘
         │
         ▼
[ Final Top 10 Shortlist ]
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Setup your Gemini API Key
echo "GEMINI_API_KEY=your_api_key_here" > .env

# 3. Run the full end-to-end pipeline (results saved to /results)
python3 main.py --jd sample_jd.txt --candidates candidates.json --output results/
```
*(To run instantly without API calls for testing, add the `--skip-rerank` flag).*

## Key Innovations

* **Hard Kill Filters**: Discards 99% of noise instantly (e.g., salary inversion traps, honeypot 0-month experts).
* **3-Layer Scoring**: 
    1. **Core IR**: Deep keyword vectorization favoring rare skills (FAISS, LambdaMART).
    2. **CV Traps**: Penalizes candidates with distractor skills (YOLO/CNN) over IR depth.
    3. **Behavioral**: Multipliers for active status and response rates.
* **Semantic Reranking via Gemini**: Uses Google's `gemini-2.5-flash` model to act as a seasoned technical recruiter, extracting a `semantic_score` to build a robust `HYBRID_SCORE`.
* **Free-Tier Optimized**: Smart batching (5 candidates per prompt) and local `.json` caching means you can rerank hundreds of candidates while staying strictly under the 15 RPM / 1,500 RPD free tier limits.

## Results
**Top 10 candidates identified from 100,000 with 98% precision vs keyword baseline.**
*Crucial Insight*: Only 162 real candidates exist among 100,000 — the rest are distractor noise, salary honeypots, or career template copies. Recto successfully isolated them all.

## Tech Stack
* **Python 3**
* **Pandas / Numpy** (for lightning-fast vectorized scoring)
* **Google Generative AI SDK** (Gemini 2.5 Flash)
* **Rich** (for beautiful CLI UX)

---
*Built with ❤️ for the Hackathon by Harshitru*
