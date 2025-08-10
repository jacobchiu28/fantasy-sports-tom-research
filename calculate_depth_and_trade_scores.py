import pandas as pd
from pathlib import Path

def main():
    # Configurable threshold for positional depth
    POINTS_THRESHOLD = 8

    # File paths
    INPUT_PATH = Path("data") / "03_primary" / "simulated_rosters.csv"
    OUTPUT_PATH = Path("data") / "03_primary" / "simulated_rosters_with_scores.csv"

    # Ensure output directory exists
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Read the primary dataset
    rosters_df = pd.read_csv(INPUT_PATH)

    # Use 'Yds' as a proxy for projected points
    POINTS_COL = "Yds"

    # Prepare results
    results = []

    # Group by roster_id and team
    for (roster_id, team), group in rosters_df.groupby(["roster_id", "team"]):
        # Count number of RBs and WRs above threshold
        rb_count = group[(group["position"] == "RB") & (pd.to_numeric(group[POINTS_COL], errors="coerce") > POINTS_THRESHOLD)].shape[0]
        wr_count = group[(group["position"] == "WR") & (pd.to_numeric(group[POINTS_COL], errors="coerce") > POINTS_THRESHOLD)].shape[0]
        te_count = group[(group["position"] == "TE") & (pd.to_numeric(group[POINTS_COL], errors="coerce") > POINTS_THRESHOLD)].shape[0]
        qb_count = group[(group["position"] == "QB") & (pd.to_numeric(group[POINTS_COL], errors="coerce") > POINTS_THRESHOLD)].shape[0]
        # Positional depth score: store counts
        positional_depth = {
            "roster_id": roster_id,
            "team": team,
            "rb_depth": rb_count,
            "wr_depth": wr_count,
            "te_depth": te_count,
            "qb_depth": qb_count,
        }
        # Trade need score: RB-WR imbalance
        positional_depth["trade_need_score"] = rb_count - wr_count
        results.append(positional_depth)

        # Convert to DataFrame and save
        scores_df = pd.DataFrame(results)
        scores_df.to_csv(OUTPUT_PATH, index=False)

        print(f"Saved positional depth and trade need scores to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()