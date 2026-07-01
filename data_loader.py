import pandas as pd
import json
import re
import os
import pickle
import numpy as np
from rich.console import Console

console = Console()

# Configuration dictionary to make filter logic and thresholds tunable
FILTER_CONFIG = {
    'input_json': 'candidates.json',
    'input_csv': 'candidates.csv',
    'output_pkl': 'filtered_candidates.pkl',
    
    'noise_patterns': [
        r'\btemplate pattern\b',
        r'\bgeneric description\b'
    ],
    
    # 2d. Patterns for non-technical titles
    'non_technical_titles': [
        r'\bhr\b',
        r'\brecruiter\b',
        r'\bsales\b',
        r'\bmanager\b',
        r'\baccountant\b'
    ],
    
    # 2d. IR keywords to salvage candidates with non-technical titles
    'ir_keywords': [
        r'\binformation retrieval\b',
        r'\bsearch engine\b',
        r'\belasticsearch\b',
        r'\bsolr\b',
        r'\blucene\b',
        r'\branking\b',
        r'\brecommender\b'
    ],
    
    # 2f. Services companies to flag and penalize
    'services_companies': [
        r'\bTCS\b',
        r'\bInfosys\b',
        r'\bWipro\b',
        r'\bAccenture\b',
        r'\bCognizant\b'
    ],
    'services_penalty': -20
}

def load_data(config):
    json_path = config['input_json']
    csv_path = config['input_csv']
    
    if os.path.exists(json_path):
        console.print(f"[cyan]Loading data from JSON: {json_path}[/cyan]")
        try:
            df = pd.read_json(json_path)
        except ValueError:
            df = pd.read_json(json_path, lines=True)
    elif os.path.exists(csv_path):
        console.print(f"[yellow]Loading data from CSV fallback: {csv_path}[/yellow]")
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"Could not find {json_path} or {csv_path}. Please ensure the dataset exists.")
        
    # FLATTEN RAW DATASET SCHEMA
    if 'profile' in df.columns:
        console.print("[cyan]Flattening raw nested dataset schema...[/cyan]")
        import datetime
        
        df['name'] = df['profile'].apply(lambda x: x.get('anonymized_name', '') if isinstance(x, dict) else '')
        df['title'] = df['profile'].apply(lambda x: x.get('current_title', '') if isinstance(x, dict) else '')
        df['country'] = df['profile'].apply(lambda x: x.get('country', '') if isinstance(x, dict) else '')
        df['duration_months'] = df['profile'].apply(lambda x: int(x.get('years_of_experience', 0)*12) if isinstance(x, dict) else 0)
        
        def extract_career(history):
            if not isinstance(history, list): return ''
            return ' '.join([item.get('description', '') for item in history if isinstance(item, dict)])
        df['career_description'] = df.get('career_history', pd.Series([])).apply(extract_career)
        
        def calc_career_span_months(history):
            if not isinstance(history, list): return 0
            return sum(item.get('duration_months', 0) for item in history if isinstance(item, dict))
        df['career_span_months'] = df.get('career_history', pd.Series([])).apply(calc_career_span_months)
        
        def extract_skills(skills):
            if not isinstance(skills, list): return []
            return [item.get('name', '') for item in skills if isinstance(item, dict)]
        df['skills_list'] = df.get('skills', pd.Series([])).apply(extract_skills)
        
        # Preserve raw skill dicts for proficiency/duration scoring
        df['raw_skills'] = df.get('skills', pd.Series([]))
        
        def check_skill_fabrication(skills):
            if not isinstance(skills, list): return False
            for s in skills:
                if isinstance(s, dict):
                    if s.get('proficiency') == 'expert' and s.get('duration_months', -1) == 0:
                        return True
            return False
        df['skill_fabrication_flag'] = df.get('skills', pd.Series([])).apply(check_skill_fabrication)
        
        df['skills'] = df['skills_list']
        
        if 'redrob_signals' in df.columns:
            def safe_get(d, key, default=None):
                return d.get(key, default) if isinstance(d, dict) else default
                
            df['salary_min'] = df['redrob_signals'].apply(lambda x: safe_get(safe_get(x, 'expected_salary_range_inr_lpa', {}), 'min', 0))
            df['salary_max'] = df['redrob_signals'].apply(lambda x: safe_get(safe_get(x, 'expected_salary_range_inr_lpa', {}), 'max', 0))
            df['willing_to_relocate'] = df['redrob_signals'].apply(lambda x: safe_get(x, 'willing_to_relocate', False))
            df['onsite_preference'] = df['redrob_signals'].apply(lambda x: safe_get(x, 'preferred_work_mode', 'flexible'))
            df['open_to_work'] = df['redrob_signals'].apply(lambda x: safe_get(x, 'open_to_work_flag', False))
            df['response_rate'] = df['redrob_signals'].apply(lambda x: safe_get(x, 'recruiter_response_rate', 0.0))
            df['notice_period_days'] = df['redrob_signals'].apply(lambda x: safe_get(x, 'notice_period_days', 30))
            df['github_stars'] = df['redrob_signals'].apply(lambda x: safe_get(x, 'github_activity_score', 0))
            df['search_appearance'] = df['redrob_signals'].apply(lambda x: safe_get(x, 'search_appearance_30d', 0))
            df['recruiter_saves'] = df['redrob_signals'].apply(lambda x: safe_get(x, 'saved_by_recruiters_30d', 0))
            df['assessment_scores'] = df['redrob_signals'].apply(lambda x: safe_get(x, 'skill_assessment_scores', {}))
            
            def calc_active_days(x):
                date_str = safe_get(x, 'last_active_date')
                if not date_str: return 999
                try:
                    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                    return (datetime.datetime.now() - dt).days
                except Exception:
                    return 999
            df['active_days_since_last_login'] = df['redrob_signals'].apply(calc_active_days)
    
    return df

def apply_filters(df, config):
    """
    2. Applies HARD KILL filters in exact order and flags candidates.
    Returns the filtered DataFrame and a dictionary of summary statistics.
    """
    summary = {
        'initial_count': len(df),
        'removed_salary_inversion': 0,
        'removed_noise': 0,
        'removed_honeypot': 0,
        'removed_non_technical': 0,
        'removed_notice_period': 0,
        'removed_location_mismatch': 0,
        'flagged_services_company': 0,
        'final_count': 0
    }
    
    if df.empty:
        return df, summary
        
    # a. salary_min > salary_max (salary inversion trap)
    if 'salary_min' in df.columns and 'salary_max' in df.columns:
        condition_a = df['salary_min'] > df['salary_max']
        summary['removed_salary_inversion'] = condition_a.sum()
        df = df[~condition_a].copy()
    
    # b. career_description is null/empty OR matches any IR/search template pattern (non-IR noise)
    if 'career_description' in df.columns:
        # Check null or empty
        is_null_empty = df['career_description'].isnull() | (df['career_description'].str.strip() == '')
        
        # Check noise patterns
        noise_regex = '|'.join(config['noise_patterns'])
        is_noise = df['career_description'].str.contains(noise_regex, na=False, regex=True, case=False)
        
        condition_b = is_null_empty | is_noise
        summary['removed_noise'] = condition_b.sum()
        df = df[~condition_b].copy()
        
    # c. Skill fabrication flag (expert + 0 duration) and Timeline impossibility
    condition_fabrication = df.get('skill_fabrication_flag', pd.Series([False]*len(df))) == True
    
    # Timeline impossibility: YoE > career span + 2 years
    condition_timeline = (df.get('duration_months', 0) / 12.0) > ((df.get('career_span_months', 0) / 12.0) + 2.0)
    
    condition_c = condition_fabrication | condition_timeline
    summary['removed_honeypot'] = condition_c.sum()
    df = df[~condition_c].copy()
    # d. title is non-technical AND no IR keyword in career
    if 'title' in df.columns and 'career_description' in df.columns:
        non_tech_regex = '|'.join(config['non_technical_titles'])
        ir_keyword_regex = '|'.join(config['ir_keywords'])
        
        is_non_tech = df['title'].str.contains(non_tech_regex, na=False, regex=True, case=False)
        has_ir_keyword = df['career_description'].str.contains(ir_keyword_regex, na=False, regex=True, case=False)
        
        condition_d = is_non_tech & (~has_ir_keyword)
        summary['removed_non_technical'] = condition_d.sum()
        df = df[~condition_d].copy()
        
    # d2. Notice period > 90 days (explicit JD disqualifier)
    if 'notice_period_days' in df.columns:
        condition_notice = df['notice_period_days'] > 90
        summary['removed_notice_period'] = condition_notice.sum()
        df = df[~condition_notice].copy()
        
    # e. Hard disqualify: outside India + unwilling to relocate
    if 'country' in df.columns and 'willing_to_relocate' in df.columns:
        is_outside_india = (df['country'].str.lower() != 'india') & (df['country'].str.strip() != '')
        condition_e = is_outside_india & (df['willing_to_relocate'] == False)
        summary['removed_location_mismatch'] = condition_e.sum()
        df = df[~condition_e].copy()
        
    # f. services_company flag -> assign -20 pts penalty (don't eliminate, just flag)
    # We initialize the penalty score column
    df['penalty_score'] = 0
    if 'career_description' in df.columns:
        services_regex = '|'.join(config['services_companies'])
        is_services_company = df['career_description'].str.contains(services_regex, na=False, regex=True, case=False)
        
        df.loc[is_services_company, 'penalty_score'] = config['services_penalty']
        summary['flagged_services_company'] = is_services_company.sum()
        
    summary['final_count'] = len(df)
    return df, summary

def print_summary(summary):
    """
    3. Prints a summary of total removed by each rule and candidates remaining.
    """
    console.print("\n[bold cyan]--- Filter Summary ---[/bold cyan]")
    console.print(f"Initial Candidate Count: [yellow]{summary['initial_count']}[/yellow]")
    console.print(f"Removed by (a) Salary Inversion Trap: [red]{summary['removed_salary_inversion']}[/red]")
    console.print(f"Removed by (b) Non-IR Noise/Empty Career: [red]{summary['removed_noise']}[/red]")
    console.print(f"Removed by (c) Honeypot Flag (Fabrication/Timeline): [red]{summary['removed_honeypot']}[/red]")
    console.print(f"Removed by (d) Non-Technical Title + No IR keyword: [red]{summary['removed_non_technical']}[/red]")
    console.print(f"Removed by (e) Notice Period > 90 days: [red]{summary['removed_notice_period']}[/red]")
    console.print(f"Removed by (f) Location Mismatch (India/No-Relo/Onsite): [red]{summary['removed_location_mismatch']}[/red]")
    console.print("[cyan]----------------------[/cyan]")
    console.print(f"Flagged with Services Company Penalty ({FILTER_CONFIG['services_penalty']} pts): [magenta]{summary['flagged_services_company']}[/magenta]")
    console.print(f"Final Candidates Remaining: [bold green]{summary['final_count']}[/bold green]")
    console.print("[cyan]----------------------[/cyan]\n")

def get_sample(n=50, pickle_file=FILTER_CONFIG['output_pkl']):
    """
    5. Returns a random sample for testing from the filtered dataset.
    """
    if not os.path.exists(pickle_file):
        raise FileNotFoundError(f"Pickle file '{pickle_file}' does not exist. Run the pipeline first.")
        
    with open(pickle_file, 'rb') as f:
        df = pickle.load(f)
        
    sample_size = min(n, len(df))
    return df.sample(n=sample_size)

def main():
    try:
        # 1. Load Data
        df = load_data(FILTER_CONFIG)
        
        # 2. Apply Filters
        filtered_df, summary = apply_filters(df, FILTER_CONFIG)
        
        # 3. Print Summary
        print_summary(summary)
        
        # 4. Save to Pickle
        output_path = FILTER_CONFIG['output_pkl']
        with open(output_path, 'wb') as f:
            pickle.dump(filtered_df, f)
        console.print(f"[bold green]Filtered dataset saved to {output_path}[/bold green]")
        
    except FileNotFoundError as e:
        console.print(f"[bold red]Error: {e}[/bold red]")
        # Note: In a real scenario you might create a mock dataset here to test if file is absent

if __name__ == "__main__":
    main()
