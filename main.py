import argparse
import os
import time
import shutil
import pickle
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

import data_loader
import jd_parser
import scorer
import output_formatter

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Recto: AI Candidate Ranking Pipeline")
    parser.add_argument("--jd", type=str, required=True, help="Path to Job Description txt file")
    parser.add_argument("--candidates", type=str, required=True, help="Path to candidates json/csv")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--sensitivity", action="store_true", help="Run sensitivity analysis and exit")
    args = parser.parse_args()

    # Ensure environment variables are loaded so jd_parser and reranker have the API key
    load_dotenv()

    os.makedirs(args.output, exist_ok=True)
    
    console.print(Panel("[bold blue]🚀 RECTO Pipeline Started[/bold blue]\n[italic]Ranking candidates the way a great recruiter would.[/italic]"))

    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        
        # Step 1: Load and Filter
        task1 = progress.add_task("[cyan]Step 1: Loading & Filtering Data...", total=None)
        t0 = time.time()
        
        # Override data loader paths dynamically
        data_loader.FILTER_CONFIG['input_json'] = args.candidates
        data_loader.FILTER_CONFIG['input_csv'] = args.candidates
        filtered_path = os.path.join(args.output, 'filtered_candidates.pkl')
        data_loader.FILTER_CONFIG['output_pkl'] = filtered_path
        
        df = data_loader.load_data(data_loader.FILTER_CONFIG)
        filtered_df, summary = data_loader.apply_filters(df, data_loader.FILTER_CONFIG)
        
        with open(filtered_path, 'wb') as f:
            pickle.dump(filtered_df, f)
            
        progress.update(task1, description=f"[green]✔ Step 1: Loaded & Filtered ({time.time()-t0:.2f}s) | Valid candidates: {summary['final_count']}[/green]")
        
        # Step 2: Parse JD
        task2 = progress.add_task("[cyan]Step 2: Parsing Job Description...", total=None)
        t0 = time.time()
        with open(args.jd, 'r', encoding='utf-8') as f:
            jd_text = f.read()
        parsed_jd = jd_parser.parse_jd(jd_text)
        progress.update(task2, description=f"[green]✔ Step 2: JD Parsed ({time.time()-t0:.2f}s) | Extracted required skills.[/green]")
        
        if args.sensitivity:
            progress.stop()
            run_sensitivity_analysis(filtered_df)
            return

        # Step 3: Score
        task3 = progress.add_task("[cyan]Step 3: Rule-based Scoring...", total=None)
        t0 = time.time()
        scored_df = scorer.score_candidates(filtered_df)
        scored_df.to_pickle(os.path.join(args.output, 'scored_candidates.pkl'))
        
        top200_path = os.path.join(args.output, 'top200_candidates.pkl')
        scored_df.head(200).to_pickle(top200_path)
        progress.update(task3, description=f"[green]✔ Step 3: Scored ({time.time()-t0:.2f}s) | Top 200 isolated.[/green]")
        
        # Step 4: Deterministic Rerank
        t0 = time.time()
        task4 = progress.add_task("[yellow]Step 4: LLM Reranking (REMOVED per Hackathon Rules)[/yellow]", total=None)
        
        final_df = scored_df.copy()
        final_df['rule_score'] = final_df['final_score']
        final_df['semantic_score'] = 0.0
        final_df['key_strengths'] = "N/A"
        
        # Apply NDCG optimal sort directly using deterministic score
        final_df = scorer.ndcg_optimal_sort(final_df.head(200))
        final_df['rank'] = final_df.index + 1
        
        final_df.to_pickle(os.path.join(args.output, 'final_ranked.pkl'))
        
        # Save CSV logic that was previously in reranker.py
        csv_cols = ['rank', 'candidate_id', 'name', 'HYBRID_SCORE', 'rule_score', 'semantic_score', 'recruiter_summary', 'key_strengths', 'ultra_rare_hit_count', 'genuine_practitioner', 'wrong_domain']
        csv_df = final_df.rename(columns={'name': 'candidate_name'})
        csv_cols = [c if c != 'name' else 'candidate_name' for c in csv_cols]
        csv_cols = [c for c in csv_cols if c in csv_df.columns]
        csv_df[csv_cols].to_csv(os.path.join(args.output, 'final_ranking.csv'), index=False)
        
        progress.update(task4, description=f"[yellow]✔ Step 4: Reranking replaced with deterministic optimal sort ({time.time()-t0:.2f}s)[/yellow]")
                
        # Step 5: Output Formatter
        task5 = progress.add_task("[cyan]Step 5: Generating Reports...", total=None)
        t0 = time.time()
        output_formatter.generate_reports(args.output)
        progress.update(task5, description=f"[green]✔ Step 5: Reports Generated ({time.time()-t0:.2f}s)[/green]")
        
    console.print(f"\n[bold green]🎉 Pipeline Complete! Check the [underline]{args.output}/[/underline] folder for shortlist_top10.csv and recto_report.md[/bold green]")

def run_sensitivity_analysis(df_filtered):
    console.print("\n[bold yellow]Running Sensitivity Analysis...[/bold yellow]")
    results = {}
    
    configs = {
        'baseline': {'ultra_rare_weight': 20, 'services_penalty': -20, 'cv_penalty': -8},
        'rare_heavy': {'ultra_rare_weight': 30, 'services_penalty': -20, 'cv_penalty': -8},
        'rare_light': {'ultra_rare_weight': 12, 'services_penalty': -20, 'cv_penalty': -8},
        'lenient_services': {'ultra_rare_weight': 20, 'services_penalty': -10, 'cv_penalty': -8},
    }
    
    baseline_top10 = None
    for config_name, weights in configs.items():
        top10_ids = scorer.score_candidates(df_filtered.copy(), weights).head(10)['candidate_id'].tolist()
        results[config_name] = top10_ids
        if config_name == 'baseline':
            baseline_top10 = set(top10_ids)
    
    print("\nSensitivity Analysis:")
    print(f"{'Config':<20} {'Overlap with baseline top-10'}")
    for name, ids in results.items():
        overlap = len(set(ids) & baseline_top10)
        print(f"{name:<20} {overlap}/10 candidates in common")

if __name__ == "__main__":
    main()
