import pandas as pd
from pathlib import Path


def main():
    # Load player data
    players_df = pd.read_csv(
        Path("data") / "01_raw" / "football_players_positions.csv", dtype=str
    )
    players_df = players_df[~players_df["position"].isin(["DEF", "K"])]

    # Load all player stats CSVs into DataFrames
    stats_dir = Path("data") / "02_intermediate" / "2024" / "player"
    stats_files = [
        "fumbles.csv",
        "kickoff-returns.csv",
        "passing.csv",
        "punt-returns.csv",
        "receiving.csv",
        "rushing.csv",
        "tackles.csv",
    ]
    stats_dfs = {}
    for fname in stats_files:
        fpath = stats_dir / fname
        if fpath.exists():
            stats_dfs[fname] = pd.read_csv(fpath, dtype=str)
        else:
            stats_dfs[fname] = pd.DataFrame()

    eligible_names = set()
    for df in stats_dfs.values():
        if not df.empty and "Player" in df.columns:
            stat_cols = [col for col in df.columns if col != "Player"]
            # Replace empty, whitespace, and '0' with pd.NA
            stat_df = df[stat_cols].replace({r"^\s*$": pd.NA, "0": pd.NA}, regex=True)
            mask = stat_df.notna().any(axis=1)
            for player_name in df.loc[mask, "Player"]:
                eligible_names.add(player_name.strip())

    eligible_players_df = players_df[players_df["name"].isin(eligible_names)].copy()

    # Position constraints per roster (total must be 12)
    POSITION_COUNTS = {"QB": 3, "RB": 4, "WR": 8, "TE": 3}
    NUM_ROSTERS = 100

    qb_pool = eligible_players_df[eligible_players_df["position"] == "QB"]
    rb_pool = eligible_players_df[eligible_players_df["position"] == "RB"]
    wr_pool = eligible_players_df[eligible_players_df["position"] == "WR"]
    te_pool = eligible_players_df[eligible_players_df["position"] == "TE"]

    # Collect all stat columns (excluding Player column)
    all_stat_cols = set()
    for df in stats_dfs.values():
        if not df.empty:
            all_stat_cols.update([col for col in df.columns if col != "Player"])
    all_stat_cols = sorted(all_stat_cols)

    rosters = []
    for i in range(NUM_ROSTERS):
        for team_num in [1, 2]:
            roster = []
            used_ids = set()
            # Add QBs
            qbs = qb_pool.sample(POSITION_COUNTS["QB"])
            roster.extend(qbs[["name"]].to_dict("records"))
            used_ids.update(qbs["player_id"].tolist())
            # Add RBs
            rbs = rb_pool[~rb_pool["player_id"].isin(used_ids)].sample(
                POSITION_COUNTS["RB"]
            )
            roster.extend(rbs[["name"]].to_dict("records"))
            used_ids.update(rbs["player_id"].tolist())
            # Add WRs
            wrs = wr_pool[~wr_pool["player_id"].isin(used_ids)].sample(
                POSITION_COUNTS["WR"]
            )
            roster.extend(wrs[["name"]].to_dict("records"))
            used_ids.update(wrs["player_id"].tolist())
            # Add TEs
            tes = te_pool[~te_pool["player_id"].isin(used_ids)].sample(
                POSITION_COUNTS["TE"]
            )
            roster.extend(tes[["name"]].to_dict("records"))
            used_ids.update(tes["player_id"].tolist())

    # Save to CSV
    output_path = (
        Path(__file__).parent / "data" / "03_primary" / "simulated_rosters.csv"
    )
    with output_path.open("w", encoding="utf-8") as f:
        header = ["roster_id", "team", "player_num", "player_name"] + all_stat_cols
        f.write(",".join(header) + "\n")
        for r in rosters:
            for idx, player in enumerate(r["players"], 1):
                row = [r["roster_id"], r["team"], idx, player["name"]]
                # Merge stats for this player from all stat files
                player_stats = {
                    col: "0" for col in all_stat_cols
                }
                for df in stats_dfs.values():
                    if not df.empty and "Player" in df.columns:
                        stat_row = df[df["Player"].str.strip() == player["name"]]
                        if not stat_row.empty:
                            for col in stat_row.columns:
                                if col != "Player" and col in all_stat_cols:
                                    value = stat_row.iloc[0][col]
                                    if (
                                        pd.isna(value)
                                        or str(value).strip() == ""
                                        or str(value).strip().lower() == "nan"
                                    ):
                                        continue
                                    player_stats[col] = value
                row += [player_stats[col] for col in all_stat_cols]
                f.write(",".join(map(str, row)) + "\n")
    print(f"Generated {len(rosters)} rosters and saved to {output_path}")


if __name__ == "__main__":
    main()
