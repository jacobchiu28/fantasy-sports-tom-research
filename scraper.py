from functions import NFLDataScraper

# Create an instance of the NFLDataScraper class
scraper = NFLDataScraper()

# Set the season before calling get_stats
scraper.season = 2024

# Check current season and week
scraper.set_current_season_and_week()

# Call the get_stats method with the season argument
scraper.get_stats(level = "player")
scraper.get_stats(level = "team")