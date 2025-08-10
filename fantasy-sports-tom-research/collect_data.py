import pandas as pd
import requests
from bs4 import BeautifulSoup

def get_injury_data():
    """Scrape injury data from RotoWire"""
    url = "https://www.rotowire.com/football/injury-report.php"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.content, "html.parser")

    injury_dict = {}
    for table in soup.find_all('table'):
        rows = table.find_all('tr')[1:]
        for row in rows:
            cells = row.find_all('td')
            if len(cells) >= 2:
                player_name = cells[0].get_text(strip=True)
                status = cells[-1].get_text(strip=True)
                injury_dict[player_name] = status
    return injury_dict


def scrape_position_data(position, limit):
    """Scrape player projections (name, team, points) from FantasyPros"""
    url_map = {
        'QB': "https://www.fantasypros.com/nfl/projections/qb.php",
        'RB': "https://www.fantasypros.com/nfl/projections/rb.php?week=1&scoring=PPR",
        'WR': "https://www.fantasypros.com/nfl/projections/wr.php?week=1&scoring=PPR",
        'TE': "https://www.fantasypros.com/nfl/projections/te.php?week=1&scoring=PPR"
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url_map[position], headers=headers)
    soup = BeautifulSoup(resp.content, "html.parser")
    table = soup.find('table', id='data')
    if not table:
        return []

    players = []
    rows = table.tbody.find_all('tr')[:limit]
    for row in rows:
        cells = row.find_all('td')
        if len(cells) < 2:
            continue
        # Extract text with space separator to separate name/team
        full = cells[0].get_text(" ", strip=True)
        parts = full.split()
        # Team is last if 2-3 uppercase letters
        team = parts[-1] if parts[-1].isupper() and 2 <= len(parts[-1]) <= 3 else 'Unknown'
        # Name is everything before team
        name = ' '.join(parts[:-1]) if team != 'Unknown' else full
        # Projected points from last cell
        try:
            points = float(cells[-1].get_text(strip=True))
        except (ValueError, IndexError):
            points = 0.0
        players.append((name, position, team, points))
    return players


def match_injury(name, injury_data):
    """Return injury status by relaxed name matching"""
    n = name.lower()
    for player, status in injury_data.items():
        p = player.lower()
        if n == p or n in p or p in n:
            return status
    return 'Healthy'


def create_fantasy_dataset():
    injuries = get_injury_data()
    all_players = []
    for pos, limit in [('QB', 30), ('RB', 70), ('WR', 100), ('TE', 40)]:
        all_players.extend(scrape_position_data(pos, limit))

    records = []
    for name, pos, team, pts in all_players:
        status = match_injury(name, injuries)
        records.append({
            'name': name,
            'position': pos,
            'team': team,
            'projected_points': pts,
            'injury_status': status
        })
    df = pd.DataFrame(records)
    df.to_csv('top_240_fantasy_players.csv', index=False)
    return df

if __name__ == '__main__':
    df = create_fantasy_dataset()
    print(df.head())
