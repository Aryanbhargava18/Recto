import json

with open("candidates.jsonl", "r") as f:
    candidates = [json.loads(line) for line in f]

killed = 0
genuine = []

for c in candidates:
    yoe = c.get('profile', {}).get('years_of_experience', 0)
    yoe_months = yoe * 12
    
    is_fake = False
    for skill in c.get('skills', []):
        if skill.get('duration_months', 0) > yoe_months + 24:
            is_fake = True
            break
    
    if is_fake:
        killed += 1
    else:
        genuine.append(c)

print(f"Total: {len(candidates)}")
print(f"Killed by skill timeline fabrication: {killed}")
print(f"Survived: {len(genuine)}")

# Let's find Aarav with ~7 years experience in the survivors
for c in genuine:
    if "Aarav" in c['profile']['anonymized_name'] and 6.5 < c['profile']['years_of_experience'] < 8:
        print(f"Found Aarav: {c['candidate_id']} - {c['profile']['anonymized_name']} - YoE: {c['profile']['years_of_experience']:.1f}")
