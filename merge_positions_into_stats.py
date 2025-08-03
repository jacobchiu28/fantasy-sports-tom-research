from pathlib import Path
import pandas as pd

def main():
    positions_path = Path("data") / "01_raw" / "football_players_positions.csv"
    player_stats_dir = Path("data") / "01_raw" / "2024" / "player"
    output_player_dir = Path("data") / "02_intermediate" / "2024" / "player"
    output_player_dir.mkdir(parents=True, exist_ok=True)

    positions_df = pd.read_csv(positions_path, dtype=str)
    positions_df["name"] = positions_df["name"].str.strip()
    positions_df["position"] = positions_df["position"].fillna("")

    stats_files = [
        "field-goals.csv",
        "fumbles.csv",
        "interceptions.csv",
        "kickoff-returns.csv",
        "kickoffs.csv",
        "passing.csv",
        "punt-returns.csv",
        "punts.csv",
        "receiving.csv",
        "rushing.csv",
        "tackles.csv",
    ]

    for stats_file in stats_files:
        stats_path = player_stats_dir / stats_file
        df = pd.read_csv(stats_path, dtype=str)
        if "Player" not in df.columns:
            print(f"Skipping {stats_file}: no 'Player' column.")
            continue
        df["Player"] = df["Player"].str.strip()
        merged = df.merge(
            positions_df[["name", "position"]],
            left_on="Player",
            right_on="name",
            how="left",
        )
        merged.drop(columns=["name"], inplace=True)
        cols = list(merged.columns)
        if "position" in cols:
            cols.insert(cols.index("Player") + 1, cols.pop(cols.index("position")))
            merged = merged[cols]
        merged = merged[merged["position"].isin(["QB", "RB", "WR", "TE"])]
        # Only save if there are players after filtering
        if not merged.empty:
            output_path = output_player_dir / stats_file
            merged.to_csv(output_path, index=False)
            unmatched = merged[merged["position"].isnull()]["Player"].tolist()
            if unmatched:
                print(f"{stats_file}: {len(unmatched)} unmatched players (position missing)")

if __name__ == "__main__":
    main()
