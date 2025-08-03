import requests
import pandas as pd
from pathlib import Path


def fetch_sleeper_players():
    url = "https://api.sleeper.app/v1/players/nfl"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def preprocess_players(player_data: pd.DataFrame) -> pd.DataFrame:
    players = []
    for player_id, pdata in player_data.items():
        # Only include active NFL players with a valid position
        players.append(
            {
                "player_id": player_id,
                "name": f"{pdata.get('first_name', '')} {pdata.get('last_name', '')}",
                "position": pdata.get("position"),
                "team": pdata.get("team"),
                "injury_status": pdata.get("injury_status"),
                "fantasy_positions": pdata.get("fantasy_positions"),
                "adp": pdata.get("adp", None),
                "proj_points": pdata.get(
                    "proj_points", None
                ),
            }
        )
    return pd.DataFrame(players)


def main():
    print("Fetching player data from SleeperAPI...")
    player_data = fetch_sleeper_players()
    print(f"Fetched {len(player_data)} players.")
    df = preprocess_players(player_data)
    print(f"Processed {len(df)} fantasy-relevant players.")
    output_path = Path(__file__).parent / "data" / "raw" / "football_players_positions.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
