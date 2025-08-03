import csv
import os
from typing import Any

# --- Prompt Templates (customize as needed) ---
TOM_PROMPT_TEMPLATE = (
    "You are the manager of Team {team}. Your roster: RBs={rb_depth}, WRs={wr_depth}, TEs={te_depth}, QBs={qb_depth}. "
    "Your trade need score: {trade_need_score}.\n"
    "Your opponent (Team {opp_team}) has: RBs={opp_rb_depth}, WRs={opp_wr_depth}, TEs={opp_te_depth}, QBs={opp_qb_depth}. "
    "Their trade need score: {opp_trade_need_score}.\n"
    "What trade would you propose, and why? Consider what your opponent needs and what they believe about your needs."
)

BASELINE_PROMPT_TEMPLATE = (
    "You are the manager of Team {team}. Your roster: RBs={rb_depth}, WRs={wr_depth}, TEs={te_depth}, QBs={qb_depth}. "
    "Your trade need score: {trade_need_score}.\n"
    "Your opponent (Team {opp_team}) has: RBs={opp_rb_depth}, WRs={opp_wr_depth}, TEs={opp_te_depth}, QBs={opp_qb_depth}. "
    "Their trade need score: {opp_trade_need_score}.\n"
    "What trade would you propose?"
)

# --- File Paths ---
INPUT_CSV = os.path.join("data", "03_primary", "simulated_rosters_with_scores.csv")
OUTPUT_CSV = os.path.join("data", "03_primary", "generated_prompts.csv")

# --- Read and Pair Rosters ---
def read_rosters(input_csv: str) -> dict[str, dict[str, dict[str, str]]]:
    scenarios: dict[str, dict[str, dict[str, str]]] = {}
    with open(input_csv, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rid = str(row['roster_id'])
            team = str(row['team'])
            if rid not in scenarios:
                scenarios[rid] = {}
            scenarios[rid][team] = {k: str(v) for k, v in row.items()}
    return scenarios

# --- Generate Prompts ---
def generate_prompts(scenarios: dict[str, dict[str, dict[str, str]]]) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for rid, teams in scenarios.items():
        if '1' not in teams or '2' not in teams:
            continue  # skip incomplete pairs
        for team_id, opp_id in [('1', '2'), ('2', '1')]:
            team = teams[team_id]
            opp = teams[opp_id]
            # ToM prompt
            tom_prompt = TOM_PROMPT_TEMPLATE.format(
                team=str(team['team']),
                rb_depth=str(team['rb_depth']),
                wr_depth=str(team['wr_depth']),
                te_depth=str(team['te_depth']),
                qb_depth=str(team['qb_depth']),
                trade_need_score=str(team['trade_need_score']),
                opp_team=str(opp['team']),
                opp_rb_depth=str(opp['rb_depth']),
                opp_wr_depth=str(opp['wr_depth']),
                opp_te_depth=str(opp['te_depth']),
                opp_qb_depth=str(opp['qb_depth']),
                opp_trade_need_score=str(opp['trade_need_score'])
            )
            prompts.append({
                'roster_id': str(rid),
                'team': str(team['team']),
                'prompt_type': 'ToM',
                'prompt_text': tom_prompt
            })
            # Baseline prompt
            baseline_prompt = BASELINE_PROMPT_TEMPLATE.format(
                team=str(team['team']),
                rb_depth=str(team['rb_depth']),
                wr_depth=str(team['wr_depth']),
                te_depth=str(team['te_depth']),
                qb_depth=str(team['qb_depth']),
                trade_need_score=str(team['trade_need_score']),
                opp_team=str(opp['team']),
                opp_rb_depth=str(opp['rb_depth']),
                opp_wr_depth=str(opp['wr_depth']),
                opp_te_depth=str(opp['te_depth']),
                opp_qb_depth=str(opp['qb_depth']),
                opp_trade_need_score=str(opp['trade_need_score'])
            )
            prompts.append({
                'roster_id': str(rid),
                'team': str(team['team']),
                'prompt_type': 'Baseline',
                'prompt_text': baseline_prompt
            })
    return prompts

# --- Write Prompts to CSV ---
def write_prompts(prompts: list[dict[str, Any]], output_csv: str) -> None:
    fieldnames = ['roster_id', 'team', 'prompt_type', 'prompt_text']
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for prompt in prompts:
            writer.writerow(prompt)

if __name__ == "__main__":
    scenarios = read_rosters(INPUT_CSV)
    prompts = generate_prompts(scenarios)
    write_prompts(prompts, OUTPUT_CSV)
    print(f"Generated {len(prompts)} prompts and saved to {OUTPUT_CSV}")
