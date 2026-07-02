"""
Recto — AI Candidate Ranking Pipeline

Ranks candidates from a large candidate pool for the Information Retrieval
Engineer role using a hybrid heuristic + semantic scoring engine.

Usage:
    python main.py --candidates candidates.jsonl [--output results/]
"""

import argparse
import os
import time
import pickle

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

import data_loader
import jd_parser
import scorer
import output_formatter

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="Recto: AI Candidate Ranking Pipeline"
    )
    parser.add_argument(
        "--candidates", type=str, required=True,
        help="Path to candidates JSONL file"
    )
    parser.add_argument(
        "--jd", type=str, required=False, default=None,
        help="Path to Job Description text file (optional; uses built-in IR logic if omitted)"
    )
    parser.add_argument(
        "--output", type=str, default="results",
        help="Output directory for ranked CSV and reports"
    )
    parser.add_argument(
        "--skip-rerank", action="store_true",
        help="Skip LLM reranking (Note: Recto is 100% deterministic, so this is a no-op)"
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    console.print(Panel(
        "[bold blue]🚀 RECTO Pipeline Started[/bold blue]\n"
        "[italic]Ranking candidates the way a great recruiter would.[/italic]"
    ))

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:

        # ── Step 1: Load & Filter ──────────────────────────────────
        task1 = progress.add_task("[cyan]Step 1: Loading & Filtering Data...", total=None)
        t0 = time.time()

        data_loader.FILTER_CONFIG['input_json'] = args.candidates
        data_loader.FILTER_CONFIG['input_csv'] = args.candidates
        filtered_path = os.path.join(args.output, 'filtered_candidates.pkl')
        data_loader.FILTER_CONFIG['output_pkl'] = filtered_path

        df = data_loader.load_data(data_loader.FILTER_CONFIG)
        filtered_df, summary = data_loader.apply_filters(df, data_loader.FILTER_CONFIG)

        with open(filtered_path, 'wb') as f:
            pickle.dump(filtered_df, f)

        progress.update(task1, description=(
            f"[green]✔ Step 1: Loaded & Filtered ({time.time()-t0:.2f}s) "
            f"| Valid candidates: {summary['final_count']}[/green]"
        ))
        
        import json
        clean_summary = {k: int(v) for k, v in summary.items()}
        with open(os.path.join(args.output, 'pipeline_stats.json'), 'w') as f:
            json.dump(clean_summary, f)

        # ── Step 2: Parse JD ───────────────────────────────────────
        task2 = progress.add_task("[cyan]Step 2: Parsing Job Description...", total=None)
        t0 = time.time()

        jd_text = ""
        if args.jd and os.path.exists(args.jd):
            with open(args.jd, 'r', encoding='utf-8') as f:
                jd_text = f.read()
        else:
            progress.print("[yellow]⚠ No JD file provided. Using built-in IR domain logic.[/yellow]")

        parsed_jd = jd_parser.parse_jd(jd_text)
        progress.update(task2, description=(
            f"[green]✔ Step 2: JD Parsed ({time.time()-t0:.2f}s) "
            f"| Extracted required skills.[/green]"
        ))

        # ── Step 3: Hybrid Scoring ─────────────────────────────────
        task3 = progress.add_task("[cyan]Step 3: Hybrid Scoring (Heuristic + Semantic)...", total=None)
        t0 = time.time()

        semantic_scores_map = None
        semantic_path = os.path.join(args.output, 'semantic_scores.pkl')
        if os.path.exists(semantic_path):
            with open(semantic_path, 'rb') as f:
                semantic_scores_map = pickle.load(f)
            progress.print(f"[dim]Loaded pre-computed semantic scores for {len(semantic_scores_map)} candidates.[/dim]")

        scored_df = scorer.score_candidates(filtered_df, semantic_scores_map=semantic_scores_map)
        progress.update(task3, description=(
            f"[green]✔ Step 3: Scored ({time.time()-t0:.2f}s) "
            f"| {len(scored_df)} candidates ranked.[/green]"
        ))

        # ── Step 4: Deterministic Rank Sort ────────────────────────
        task4 = progress.add_task("[cyan]Step 4: Deterministic Rank Sort...", total=None)
        t0 = time.time()

        final_df = scorer.ndcg_optimal_sort(scored_df)
        from output_formatter import generate_key_strengths
        final_df['key_strengths'] = final_df.apply(generate_key_strengths, axis=1)
        final_df['rank'] = range(1, len(final_df) + 1)
        final_df.to_pickle(os.path.join(args.output, 'final_ranked.pkl'))

        progress.update(task4, description=(
            f"[green]✔ Step 4: Deterministic Sort ({time.time()-t0:.2f}s) "
            f"| Sorted by verified Hybrid Score[/green]"
        ))

        # ── Step 5: Report Generation ──────────────────────────────
        task5 = progress.add_task("[cyan]Step 5: Generating Reports...", total=None)
        t0 = time.time()
        output_formatter.generate_reports(args.output)
        progress.update(task5, description=f"[green]✔ Step 5: Reports Generated ({time.time()-t0:.2f}s)[/green]")

    console.print(
        f"\n[bold green]🎉 Pipeline Complete! "
        f"Check [underline]{args.output}/[/underline] for final_ranking.csv and recto_report.md[/bold green]"
    )


if __name__ == "__main__":
    main()
