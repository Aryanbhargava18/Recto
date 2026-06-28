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
import reranker
import output_formatter

console = Console()

def main():
    parser = argparse.ArgumentParser(description="Recto: AI Candidate Ranking Pipeline")
    parser.add_argument("--jd", type=str, required=True, help="Path to Job Description txt file")
    parser.add_argument("--candidates", type=str, required=True, help="Path to candidates json/csv")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--skip-rerank", action="store_true", help="Skip the LLM semantic reranking phase")
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
        
        # Step 3: Score
        task3 = progress.add_task("[cyan]Step 3: Rule-based Scoring...", total=None)
        t0 = time.time()
        scored_df = scorer.score_candidates(filtered_df)
        scored_df.to_pickle(os.path.join(args.output, 'scored_candidates.pkl'))
        
        top200_path = os.path.join(args.output, 'top200_candidates.pkl')
        scored_df.head(200).to_pickle(top200_path)
        progress.update(task3, description=f"[green]✔ Step 3: Scored ({time.time()-t0:.2f}s) | Top 200 isolated.[/green]")
        
        # Step 4: Rerank
        t0 = time.time()
        if args.skip_rerank:
            task4 = progress.add_task("[yellow]Step 4: Semantic Reranking (SKIPPED)...", total=None)
            final_df = scored_df.copy()
            final_df['rule_score'] = final_df['final_score']
            final_df['HYBRID_SCORE'] = final_df['rule_score']
            final_df['semantic_score'] = 0.0
            final_df['recruiter_summary'] = "Reranking skipped."
            final_df['key_strengths'] = "N/A"
            final_df = final_df.sort_values(by='HYBRID_SCORE', ascending=False).reset_index(drop=True)
            final_df['rank'] = final_df.index + 1
            final_df.to_pickle(os.path.join(args.output, 'final_ranked.pkl'))
            progress.update(task4, description="[yellow]✔ Step 4: Reranking Skipped (Testing mode)[/yellow]")
        else:
            task4 = progress.add_task("[cyan]Step 4: Gemini Semantic Reranking (Batched API calls)...", total=None)
            
            # reranker.py hardcodes 'top200_candidates.pkl' in local dir, copy it there temporarily
            shutil.copy(top200_path, 'top200_candidates.pkl')
            try:
                reranker.main()
                # Move outputs to results dir
                if os.path.exists('final_ranked.pkl'):
                    shutil.move('final_ranked.pkl', os.path.join(args.output, 'final_ranked.pkl'))
                if os.path.exists('final_ranking.csv'):
                    shutil.move('final_ranking.csv', os.path.join(args.output, 'final_ranking.csv'))
                if os.path.exists('top200_candidates.pkl'):
                    os.remove('top200_candidates.pkl')
                    
                progress.update(task4, description=f"[green]✔ Step 4: Reranked via Gemini ({time.time()-t0:.2f}s)[/green]")
            except Exception as e:
                progress.update(task4, description=f"[red]✖ Step 4 Failed: {e}[/red]")
                
        # Step 5: Output Formatter
        task5 = progress.add_task("[cyan]Step 5: Generating Reports...", total=None)
        t0 = time.time()
        output_formatter.generate_reports(args.output)
        progress.update(task5, description=f"[green]✔ Step 5: Reports Generated ({time.time()-t0:.2f}s)[/green]")
        
    console.print(f"\n[bold green]🎉 Pipeline Complete! Check the [underline]{args.output}/[/underline] folder for shortlist_top10.csv and recto_report.md[/bold green]")

if __name__ == "__main__":
    main()
