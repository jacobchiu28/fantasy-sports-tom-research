import pandas as pd
from pathlib import Path

# Load player data
players_df = pd.read_csv(Path("data") / "raw" / "fantasy_players_2025.csv", dtype=str)

# Exclude Defense and Kickers
players_df = players_df[~players_df["position"].isin(["DEF", "K"])]

# Group by position
qb_pool = players_df[players_df["position"] == "QB"]
rb_pool = players_df[players_df["position"] == "RB"]
wr_pool = players_df[players_df["position"] == "WR"]
te_pool = players_df[players_df["position"] == "TE"]
flex_pool = players_df[players_df["position"].isin(["RB", "WR", "TE"])]

NUM_ROSTERS = 100
TEAM_SIZE = 12
POSITION_COUNTS = {"QB": 1, "RB": 2, "WR": 4, "TE": 2, "FLEX": 3}  # FLEX: RB/WR/TE

rosters = []

for i in range(NUM_ROSTERS):
    for team_num in [1, 2]:
        roster = []
        used_ids = set()
        # Add QBs
        qbs = qb_pool.sample(POSITION_COUNTS["QB"])
        roster.extend(qbs["name"].tolist())
        used_ids.update(qbs["player_id"].tolist())
        # Add RBs
        rbs = rb_pool[~rb_pool["player_id"].isin(used_ids)].sample(POSITION_COUNTS["RB"])
        roster.extend(rbs["name"].tolist())
        used_ids.update(rbs["player_id"].tolist())
        # Add WRs
        wrs = wr_pool[~wr_pool["player_id"].isin(used_ids)].sample(POSITION_COUNTS["WR"])
        roster.extend(wrs["name"].tolist())
        used_ids.update(wrs["player_id"].tolist())
        # Add TEs
        tes = te_pool[~te_pool["player_id"].isin(used_ids)].sample(POSITION_COUNTS["TE"])
        roster.extend(tes["name"].tolist())
        used_ids.update(tes["player_id"].tolist())
        # Add FLEX
        flex_candidates = flex_pool[~flex_pool["player_id"].isin(used_ids)]
        flex = flex_candidates.sample(POSITION_COUNTS["FLEX"])
        roster.extend(flex["name"].tolist())
        used_ids.update(flex["player_id"].tolist())
        rosters.append({"roster_id": i+1, "team": team_num, "players": roster})

# Save to CSV
output_path = Path("simulated_rosters.csv")
with output_path.open("w", encoding="utf-8") as f:
    f.write("roster_id,team,player_num,player_name\n")
    for r in rosters:
        for idx, player in enumerate(r["players"], 1):
            f.write(f"{r['roster_id']},{r['team']},{idx},\"{player}\"\n")

print(f"Generated {len(rosters)} rosters and saved to {output_path}")
