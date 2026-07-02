import json
import random
import uuid
import datetime

def generate_mock_data(n=2000, output_path='candidates.jsonl'):
    import os
    
    first_names = ["Aryan", "Harshit", "Mira", "Aarav", "Priya", "Rahul", "Neha", "Amit", "Rohan", "Anjali"]
    last_names = ["Sharma", "Verma", "Gupta", "Singh", "Patel", "Kumar", "Iyer", "Nair", "Reddy", "Das"]
    
    tech_titles = ["Software Engineer", "Machine Learning Engineer", "Frontend Developer", 
                   "Backend Developer", "Data Scientist", "Search Engineer", "Information Retrieval Engineer"]
    
    companies = ["Google", "Amazon", "Flipkart", "TCS", "Infosys", "Startup Inc", "Zomato", "Swiggy", "Razorpay"]
    
    ir_skills = ["Learning to Rank", "BM25", "Vector Search", "Embeddings", "FAISS", "Elasticsearch", "Dense Retrieval"]
    generic_skills = ["Python", "Java", "React", "SQL", "Docker", "AWS", "Git"]
    
    with open(output_path, 'w') as f:
        for _ in range(n):
            is_ir_expert = random.random() < 0.05
            is_stuffer = random.random() < 0.02
            is_services = random.random() < 0.20
            
            c_id = f"CAND_{random.randint(10000, 99999)}"
            name = f"{random.choice(first_names)} {random.choice(last_names)}"
            
            if is_ir_expert:
                title = random.choice(["Search Engineer", "Information Retrieval Engineer", "Machine Learning Engineer"])
                skill_set = random.sample(ir_skills, k=random.randint(3, 6)) + random.sample(generic_skills, k=2)
                comp = random.choice(["Google", "Flipkart", "Amazon", "Razorpay"])
                desc = f"Built hybrid retrieval systems using {', '.join(random.sample(ir_skills, 2))} to serve millions of queries."
                score = random.randint(70, 95)
            elif is_stuffer:
                title = "AI Expert"
                skill_set = ir_skills + generic_skills
                comp = "Startup Inc"
                desc = "I do bm25 ndcg mrr embeddings bi-encoder cross-encoder dense retrieval learning to rank over and over."
                score = random.randint(10, 30)
            elif is_services:
                title = "System Engineer"
                skill_set = random.sample(generic_skills, k=4)
                comp = random.choice(["TCS", "Infosys", "Wipro"])
                desc = "Worked on enterprise java applications and maintained sql databases."
                score = random.randint(10, 40)
            else:
                title = random.choice(tech_titles)
                skill_set = random.sample(generic_skills, k=5)
                comp = random.choice(companies)
                desc = "Developed web applications and scalable backend APIs."
                score = random.randint(20, 60)
                
            skills_objs = [{"name": s, "proficiency": random.choice(["intermediate", "advanced", "expert"]), "duration_months": random.randint(12, 60)} for s in skill_set]
            
            record = {
                "candidate_id": c_id,
                "profile": {
                    "anonymized_name": name,
                    "current_title": title,
                    "country": "India",
                    "years_of_experience": random.uniform(2, 10)
                },
                "career_history": [
                    {"title": title, "company": comp, "description": desc, "start_date": "2020-01", "end_date": "2024-01"}
                ],
                "skills": skills_objs,
                "redrob_signals": {
                    "expected_salary_range_inr_lpa": {"min": 10, "max": 20},
                    "willing_to_relocate": True,
                    "preferred_work_mode": "hybrid",
                    "open_to_work_flag": True,
                    "recruiter_response_rate": random.uniform(0.5, 1.0),
                    "notice_period_days": 30,
                    "github_activity_score": random.randint(10, 100),
                    "search_appearance_30d": random.randint(10, 500),
                    "saved_by_recruiters_30d": random.randint(0, 20),
                    "skill_assessment_scores": {"Semantic Search": score} if is_ir_expert else {}
                }
            }
            
            f.write(json.dumps(record) + "\n")
            
    print(f"Generated {n} candidates at {output_path}")

if __name__ == "__main__":
    generate_mock_data(2000)
