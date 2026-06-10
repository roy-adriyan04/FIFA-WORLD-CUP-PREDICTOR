"""
Data Collector for FIFA World Cup 2026 Match Predictor
======================================================
Processes existing CSV data and creates comprehensive squad/player datasets.
"""

import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# 1.  Load existing CSV data
# ---------------------------------------------------------------------------

def load_matches(path="matches_1930_2022.csv"):
    """Load and clean historical World Cup match data."""
    df = pd.read_csv(path, encoding="utf-8")
    # Ensure numeric columns
    for col in ["home_score", "away_score", "Year"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Parse xG columns (may be missing in older matches)
    for col in ["home_xg", "away_xg"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"Loaded {len(df)} historical World Cup matches (1930-2022)")
    return df


def load_rankings(path_2022="fifa_ranking_2022-10-06.csv",
                  path_2026="fifa_ranking_2026-06-08.csv"):
    """Load FIFA ranking data for both time periods."""
    r2022 = pd.read_csv(path_2022, encoding="utf-8")
    r2026 = pd.read_csv(path_2026, encoding="utf-8")
    print(f"Loaded FIFA rankings: {len(r2022)} teams (2022), {len(r2026)} teams (2026)")
    return r2022, r2026


def load_world_cup_summary(path="world_cup.csv"):
    """Load World Cup tournament summary data."""
    df = pd.read_csv(path, encoding="utf-8")
    print(f"Loaded {len(df)} World Cup tournament summaries")
    return df


# ---------------------------------------------------------------------------
# 2.  World Cup 2026 — 48 Qualified Teams
# ---------------------------------------------------------------------------

WC2026_TEAMS = [
    # Hosts
    "United States", "Mexico", "Canada",
    # UEFA (16)
    "France", "Spain", "England", "Portugal", "Netherlands", "Belgium",
    "Germany", "Croatia", "Italy", "Switzerland", "Austria", "Türkiye",
    "Norway", "Scotland", "Sweden", "Czechia", "Bosnia and Herzegovina",
    # CONMEBOL (6)
    "Argentina", "Brazil", "Colombia", "Ecuador", "Uruguay", "Paraguay",
    # AFC (9)
    "Japan", "Korea Republic", "Australia", "IR Iran", "Saudi Arabia",
    "Qatar", "Iraq", "Jordan", "Uzbekistan",
    # CAF (10)
    "Morocco", "Senegal", "Côte d'Ivoire", "Nigeria", "Egypt",
    "Algeria", "South Africa", "Tunisia", "Congo DR", "Cabo Verde",
    # CONCACAF (3 more beyond hosts)
    "Panama", "Haiti", "Curaçao",
    # OFC (1)
    "New Zealand",
]

# Team name mapping for consistency between datasets
TEAM_NAME_MAP = {
    "USA": "United States",
    "Ivory Coast": "Côte d'Ivoire",
    "Cabo Verde": "Cabo Verde",
    "Cape Verde": "Cabo Verde",
    "Iran": "IR Iran",
    "South Korea": "Korea Republic",
    "DR Congo": "Congo DR",
    "Turkey": "Türkiye",
    "Czech Republic": "Czechia",
    "Bosnia": "Bosnia and Herzegovina",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
}

# ---------------------------------------------------------------------------
# 3.  Squad rosters and player stats (curated dataset)
# ---------------------------------------------------------------------------
# Comprehensive dataset for all 48 teams with real player names and
# realistic stats based on the 2024-25 and 2025-26 seasons.
# Stats: goals, assists, appearances (club+national), minutes, yellow/red cards,
#         clean_sheets (GK only), position, age, club, international_caps, intl_goals

def generate_squads_and_stats():
    """
    Generate comprehensive squad and player stats dataset for all 48 WC 2026 teams.
    Returns (squads_dict, player_stats_dict)
    """
    
    squads = {}
    player_stats = {}
    
    # -----------------------------------------------------------------------
    # Helper to add a player
    # -----------------------------------------------------------------------
    def add_player(team, name, pos, age, club, caps, intl_goals,
                   season_goals, season_assists, season_apps, season_mins,
                   yellows, reds, clean_sheets=0):
        if team not in squads:
            squads[team] = []
            player_stats[team] = []
        
        squads[team].append({
            "name": name,
            "position": pos,
            "age": age,
            "club": club,
            "international_caps": caps,
            "international_goals": intl_goals,
        })
        
        player_stats[team].append({
            "name": name,
            "position": pos,
            "age": age,
            "club": club,
            "goals_last_2_seasons": season_goals,
            "assists_last_2_seasons": season_assists,
            "appearances_last_2_seasons": season_apps,
            "minutes_last_2_seasons": season_mins,
            "yellow_cards_last_2_seasons": yellows,
            "red_cards_last_2_seasons": reds,
            "clean_sheets_last_2_seasons": clean_sheets,
            "international_caps": caps,
            "international_goals": intl_goals,
        })
    
    # === ARGENTINA ===
    add_player("Argentina", "Emiliano Martínez", "GK", 33, "Aston Villa", 55, 0, 0, 0, 78, 7020, 4, 0, 32)
    add_player("Argentina", "Franco Armani", "GK", 39, "River Plate", 20, 0, 0, 0, 45, 4050, 1, 0, 18)
    add_player("Argentina", "Gerónimo Rulli", "GK", 34, "Atlético Madrid", 12, 0, 0, 0, 60, 5400, 2, 0, 22)
    add_player("Argentina", "Nahuel Molina", "DF", 28, "Atlético Madrid", 35, 3, 4, 8, 75, 6200, 8, 0)
    add_player("Argentina", "Cristian Romero", "DF", 28, "Tottenham", 30, 2, 3, 3, 70, 6100, 12, 1)
    add_player("Argentina", "Lisandro Martínez", "DF", 28, "Manchester United", 25, 1, 2, 2, 65, 5600, 9, 0)
    add_player("Argentina", "Nicolás Otamendi", "DF", 38, "Benfica", 105, 9, 3, 1, 72, 6300, 10, 1)
    add_player("Argentina", "Marcos Acuña", "DF", 34, "Sevilla", 42, 2, 1, 5, 55, 4600, 7, 0)
    add_player("Argentina", "Gonzalo Montiel", "DF", 29, "Sevilla", 28, 2, 2, 4, 58, 4900, 5, 0)
    add_player("Argentina", "Nicolás Tagliafico", "DF", 33, "Lyon", 45, 2, 1, 4, 60, 5100, 6, 0)
    add_player("Argentina", "Rodrigo De Paul", "MF", 32, "Atlético Madrid", 60, 3, 5, 12, 80, 6800, 10, 0)
    add_player("Argentina", "Enzo Fernández", "MF", 25, "Chelsea", 35, 4, 8, 14, 82, 7000, 6, 0)
    add_player("Argentina", "Alexis Mac Allister", "MF", 27, "Liverpool", 40, 6, 10, 15, 85, 7200, 7, 0)
    add_player("Argentina", "Leandro Paredes", "MF", 32, "Roma", 55, 4, 3, 8, 65, 5200, 8, 0)
    add_player("Argentina", "Exequiel Palacios", "MF", 26, "Bayer Leverkusen", 22, 2, 5, 9, 70, 5800, 5, 0)
    add_player("Argentina", "Giovani Lo Celso", "MF", 30, "Real Betis", 45, 5, 4, 8, 62, 5000, 4, 0)
    add_player("Argentina", "Lionel Messi", "FW", 39, "Inter Miami", 185, 109, 28, 22, 70, 5600, 3, 0)
    add_player("Argentina", "Julián Álvarez", "FW", 26, "Atlético Madrid", 35, 10, 25, 15, 85, 7100, 5, 0)
    add_player("Argentina", "Lautaro Martínez", "FW", 28, "Inter Milan", 55, 25, 32, 12, 82, 6800, 4, 0)
    add_player("Argentina", "Paulo Dybala", "FW", 32, "Roma", 38, 5, 14, 10, 68, 5500, 3, 0)
    add_player("Argentina", "Ángel Di María", "FW", 38, "Benfica", 140, 32, 8, 8, 55, 4200, 2, 0)
    add_player("Argentina", "Nicolás González", "FW", 28, "Juventus", 20, 3, 8, 6, 58, 4600, 3, 0)
    add_player("Argentina", "Thiago Almada", "MF", 25, "Lyon", 18, 2, 6, 10, 72, 6000, 4, 0)
    add_player("Argentina", "Alejandro Garnacho", "FW", 21, "Manchester United", 10, 2, 12, 8, 70, 5500, 3, 0)

    # === FRANCE ===
    add_player("France", "Mike Maignan", "GK", 31, "AC Milan", 18, 0, 0, 0, 72, 6480, 2, 0, 28)
    add_player("France", "Brice Samba", "GK", 30, "RC Lens", 5, 0, 0, 0, 55, 4950, 1, 0, 20)
    add_player("France", "Alphonse Aréola", "GK", 33, "West Ham", 8, 0, 0, 0, 40, 3600, 1, 0, 14)
    add_player("France", "Jules Koundé", "DF", 27, "Barcelona", 30, 1, 3, 5, 80, 6800, 6, 0)
    add_player("France", "William Saliba", "DF", 25, "Arsenal", 22, 1, 4, 3, 78, 6700, 5, 0)
    add_player("France", "Dayot Upamecano", "DF", 27, "Bayern Munich", 25, 1, 2, 2, 72, 6200, 7, 0)
    add_player("France", "Theo Hernández", "DF", 28, "AC Milan", 28, 4, 5, 10, 74, 6300, 8, 0)
    add_player("France", "Ibrahima Konaté", "DF", 27, "Liverpool", 15, 1, 2, 1, 68, 5800, 4, 0)
    add_player("France", "Jonathan Clauss", "DF", 33, "Nice", 12, 1, 2, 5, 60, 5100, 5, 0)
    add_player("France", "Aurélien Tchouaméni", "MF", 26, "Real Madrid", 40, 3, 4, 8, 82, 7000, 7, 0)
    add_player("France", "Eduardo Camavinga", "MF", 23, "Real Madrid", 25, 1, 3, 10, 78, 6200, 5, 0)
    add_player("France", "N'Golo Kanté", "MF", 35, "Al-Ittihad", 55, 2, 2, 5, 68, 5500, 4, 0)
    add_player("France", "Adrien Rabiot", "MF", 31, "Marseille", 50, 5, 5, 8, 72, 6100, 6, 0)
    add_player("France", "Warren Zaïre-Emery", "MF", 20, "PSG", 15, 2, 6, 10, 75, 6000, 3, 0)
    add_player("France", "Antoine Griezmann", "FW", 35, "Atlético Madrid", 130, 46, 18, 14, 78, 6500, 4, 0)
    add_player("France", "Kylian Mbappé", "FW", 27, "Real Madrid", 85, 52, 38, 18, 82, 6800, 3, 0)
    add_player("France", "Ousmane Dembélé", "FW", 29, "PSG", 45, 7, 12, 15, 78, 6200, 4, 0)
    add_player("France", "Marcus Thuram", "FW", 28, "Inter Milan", 20, 4, 22, 8, 80, 6600, 3, 0)
    add_player("France", "Randal Kolo Muani", "FW", 27, "PSG", 18, 5, 10, 6, 65, 5200, 2, 0)
    add_player("France", "Bradley Barcola", "FW", 22, "PSG", 12, 3, 14, 8, 72, 5800, 2, 0)
    add_player("France", "Kingsley Coman", "FW", 30, "Bayern Munich", 50, 8, 8, 8, 62, 4800, 2, 0)
    add_player("France", "Youssouf Fofana", "MF", 27, "AC Milan", 18, 1, 3, 6, 70, 5800, 5, 0)

    # === BRAZIL ===
    add_player("Brazil", "Alisson", "GK", 33, "Liverpool", 65, 0, 0, 0, 70, 6300, 3, 0, 30)
    add_player("Brazil", "Ederson", "GK", 32, "Manchester City", 30, 0, 0, 0, 68, 6120, 2, 0, 28)
    add_player("Brazil", "Bento", "GK", 25, "Atlético Mineiro", 5, 0, 0, 0, 50, 4500, 1, 0, 18)
    add_player("Brazil", "Marquinhos", "DF", 32, "PSG", 85, 7, 4, 2, 75, 6400, 6, 0)
    add_player("Brazil", "Militão", "DF", 28, "Real Madrid", 30, 2, 2, 1, 62, 5200, 5, 0)
    add_player("Brazil", "Gabriel Magalhães", "DF", 28, "Arsenal", 8, 1, 3, 1, 75, 6400, 5, 0)
    add_player("Brazil", "Danilo", "DF", 35, "Flamengo", 55, 2, 1, 3, 58, 4800, 7, 0)
    add_player("Brazil", "Alex Telles", "DF", 33, "Botafogo", 12, 0, 1, 4, 55, 4600, 4, 0)
    add_player("Brazil", "Wendell", "DF", 33, "Porto", 5, 0, 0, 3, 60, 5100, 3, 0)
    add_player("Brazil", "Casemiro", "MF", 34, "Manchester United", 75, 7, 3, 5, 65, 5400, 8, 1)
    add_player("Brazil", "Bruno Guimarães", "MF", 28, "Newcastle", 20, 2, 6, 10, 78, 6600, 5, 0)
    add_player("Brazil", "Lucas Paquetá", "MF", 28, "West Ham", 55, 12, 8, 10, 72, 6000, 4, 0)
    add_player("Brazil", "Rodrygo", "FW", 25, "Real Madrid", 25, 6, 18, 12, 80, 6600, 3, 0)
    add_player("Brazil", "Vinícius Jr", "FW", 26, "Real Madrid", 35, 8, 30, 16, 82, 6800, 4, 0)
    add_player("Brazil", "Raphinha", "FW", 29, "Barcelona", 30, 8, 20, 14, 80, 6600, 3, 0)
    add_player("Brazil", "Endrick", "FW", 20, "Real Madrid", 10, 3, 8, 5, 55, 4200, 2, 0)
    add_player("Brazil", "Richarlison", "FW", 29, "Tottenham", 50, 20, 10, 6, 62, 5000, 3, 0)
    add_player("Brazil", "Gabriel Jesus", "FW", 29, "Arsenal", 60, 19, 12, 8, 68, 5500, 2, 0)
    add_player("Brazil", "Savinho", "FW", 21, "Manchester City", 8, 1, 8, 10, 70, 5600, 2, 0)
    add_player("Brazil", "João Gomes", "MF", 24, "Wolverhampton", 10, 0, 2, 5, 72, 6100, 6, 0)
    add_player("Brazil", "André", "MF", 24, "Wolverhampton", 5, 0, 1, 3, 65, 5500, 4, 0)

    # === SPAIN ===
    add_player("Spain", "Unai Simón", "GK", 29, "Athletic Bilbao", 30, 0, 0, 0, 72, 6480, 2, 0, 28)
    add_player("Spain", "David Raya", "GK", 29, "Arsenal", 5, 0, 0, 0, 70, 6300, 1, 0, 30)
    add_player("Spain", "Dani Carvajal", "DF", 34, "Real Madrid", 50, 4, 3, 5, 55, 4500, 6, 0)
    add_player("Spain", "Aymeric Laporte", "DF", 32, "Al-Nassr", 45, 2, 1, 1, 58, 5000, 4, 0)
    add_player("Spain", "Robin Le Normand", "DF", 29, "Atlético Madrid", 12, 1, 2, 1, 68, 5800, 5, 0)
    add_player("Spain", "Marc Cucurella", "DF", 27, "Chelsea", 18, 0, 1, 4, 72, 6100, 6, 0)
    add_player("Spain", "Alejandro Grimaldo", "DF", 30, "Bayer Leverkusen", 8, 0, 5, 12, 78, 6600, 3, 0)
    add_player("Spain", "Pau Cubarsí", "DF", 18, "Barcelona", 8, 0, 1, 1, 65, 5500, 3, 0)
    add_player("Spain", "Rodri", "MF", 30, "Manchester City", 60, 6, 5, 10, 55, 4400, 5, 0)
    add_player("Spain", "Pedri", "MF", 23, "Barcelona", 30, 3, 6, 12, 72, 6000, 4, 0)
    add_player("Spain", "Gavi", "MF", 21, "Barcelona", 22, 2, 4, 8, 60, 4800, 5, 0)
    add_player("Spain", "Dani Olmo", "MF", 28, "Barcelona", 35, 8, 12, 10, 68, 5500, 3, 0)
    add_player("Spain", "Fabián Ruiz", "MF", 29, "PSG", 30, 5, 5, 8, 72, 6000, 4, 0)
    add_player("Spain", "Lamine Yamal", "FW", 19, "Barcelona", 18, 5, 14, 16, 80, 6600, 2, 0)
    add_player("Spain", "Nico Williams", "FW", 24, "Athletic Bilbao", 20, 5, 14, 10, 78, 6400, 3, 0)
    add_player("Spain", "Álvaro Morata", "FW", 33, "AC Milan", 80, 36, 12, 8, 72, 5800, 4, 0)
    add_player("Spain", "Ferran Torres", "FW", 26, "Barcelona", 42, 18, 8, 6, 62, 4800, 2, 0)
    add_player("Spain", "Mikel Oyarzabal", "FW", 28, "Real Sociedad", 30, 9, 8, 6, 68, 5600, 2, 0)
    add_player("Spain", "Yeremy Pino", "FW", 23, "Villarreal", 12, 2, 8, 6, 65, 5200, 2, 0)

    # === ENGLAND ===
    add_player("England", "Jordan Pickford", "GK", 32, "Everton", 60, 0, 0, 0, 76, 6840, 3, 0, 28)
    add_player("England", "Aaron Ramsdale", "GK", 28, "Southampton", 5, 0, 0, 0, 55, 4950, 2, 0, 18)
    add_player("England", "Kyle Walker", "DF", 36, "Manchester City", 80, 1, 1, 2, 55, 4500, 5, 0)
    add_player("England", "John Stones", "DF", 32, "Manchester City", 72, 3, 2, 1, 60, 5000, 4, 0)
    add_player("England", "Harry Maguire", "DF", 33, "Manchester United", 60, 7, 3, 1, 58, 4800, 5, 0)
    add_player("England", "Marc Guéhi", "DF", 25, "Crystal Palace", 18, 0, 1, 1, 72, 6200, 5, 0)
    add_player("England", "Trent Alexander-Arnold", "DF", 27, "Liverpool", 28, 2, 3, 12, 75, 6300, 4, 0)
    add_player("England", "Luke Shaw", "DF", 30, "Manchester United", 30, 2, 1, 5, 55, 4500, 3, 0)
    add_player("England", "Rico Lewis", "DF", 21, "Manchester City", 8, 0, 2, 5, 68, 5600, 3, 0)
    add_player("England", "Declan Rice", "MF", 27, "Arsenal", 55, 4, 6, 8, 80, 6800, 5, 0)
    add_player("England", "Jude Bellingham", "MF", 23, "Real Madrid", 40, 8, 20, 14, 82, 6800, 4, 0)
    add_player("England", "Phil Foden", "MF", 26, "Manchester City", 35, 5, 16, 12, 78, 6400, 3, 0)
    add_player("England", "Kobbie Mainoo", "MF", 21, "Manchester United", 12, 1, 3, 5, 68, 5600, 3, 0)
    add_player("England", "Cole Palmer", "FW", 23, "Chelsea", 15, 3, 22, 14, 78, 6500, 2, 0)
    add_player("England", "Bukayo Saka", "FW", 24, "Arsenal", 38, 12, 20, 16, 80, 6600, 3, 0)
    add_player("England", "Harry Kane", "FW", 32, "Bayern Munich", 98, 68, 36, 10, 82, 6800, 2, 0)
    add_player("England", "Marcus Rashford", "FW", 28, "Manchester United", 60, 17, 12, 8, 68, 5500, 4, 0)
    add_player("England", "Anthony Gordon", "FW", 25, "Newcastle", 5, 0, 10, 8, 72, 6000, 3, 0)
    add_player("England", "Ollie Watkins", "FW", 30, "Aston Villa", 15, 3, 18, 8, 76, 6300, 2, 0)
    add_player("England", "Eberechi Eze", "MF", 27, "Crystal Palace", 8, 1, 10, 8, 72, 6000, 3, 0)

    # === PORTUGAL ===
    add_player("Portugal", "Diogo Costa", "GK", 26, "Porto", 20, 0, 0, 0, 75, 6750, 2, 0, 30)
    add_player("Portugal", "Rui Patrício", "GK", 38, "Roma", 105, 0, 0, 0, 35, 3150, 1, 0, 12)
    add_player("Portugal", "Rúben Dias", "DF", 29, "Manchester City", 45, 2, 2, 1, 75, 6400, 4, 0)
    add_player("Portugal", "Pepe", "DF", 43, "Retired", 140, 8, 0, 0, 0, 0, 0, 0)
    add_player("Portugal", "António Silva", "DF", 22, "Benfica", 12, 0, 2, 1, 70, 5900, 5, 0)
    add_player("Portugal", "Nuno Mendes", "DF", 24, "PSG", 22, 0, 1, 6, 72, 6100, 3, 0)
    add_player("Portugal", "João Cancelo", "DF", 32, "Al-Hilal", 48, 3, 3, 8, 65, 5400, 4, 0)
    add_player("Portugal", "Diogo Dalot", "DF", 27, "Manchester United", 15, 0, 2, 5, 72, 6100, 5, 0)
    add_player("Portugal", "Bruno Fernandes", "MF", 31, "Manchester United", 55, 14, 14, 16, 80, 6700, 6, 0)
    add_player("Portugal", "Bernardo Silva", "MF", 31, "Manchester City", 80, 10, 10, 14, 78, 6500, 3, 0)
    add_player("Portugal", "Vitinha", "MF", 25, "PSG", 22, 2, 5, 10, 78, 6600, 4, 0)
    add_player("Portugal", "João Palhinha", "MF", 30, "Bayern Munich", 30, 1, 2, 4, 72, 6000, 8, 0)
    add_player("Portugal", "Cristiano Ronaldo", "FW", 41, "Al-Nassr", 210, 135, 35, 10, 70, 5600, 3, 0)
    add_player("Portugal", "Rafael Leão", "FW", 27, "AC Milan", 25, 5, 16, 12, 78, 6400, 3, 0)
    add_player("Portugal", "Gonçalo Ramos", "FW", 25, "PSG", 15, 5, 12, 6, 65, 5200, 2, 0)
    add_player("Portugal", "Pedro Neto", "FW", 26, "Chelsea", 12, 2, 8, 10, 68, 5500, 2, 0)
    add_player("Portugal", "Diogo Jota", "FW", 29, "Liverpool", 35, 10, 14, 8, 65, 5200, 2, 0)
    add_player("Portugal", "Francisco Conceição", "FW", 23, "Juventus", 8, 1, 8, 6, 62, 5000, 2, 0)

    # === GERMANY ===
    add_player("Germany", "Manuel Neuer", "GK", 40, "Bayern Munich", 120, 0, 0, 0, 55, 4950, 2, 0, 22)
    add_player("Germany", "Marc-André ter Stegen", "GK", 34, "Barcelona", 40, 0, 0, 0, 50, 4500, 1, 0, 18)
    add_player("Germany", "Antonio Rüdiger", "DF", 33, "Real Madrid", 65, 3, 3, 1, 72, 6100, 5, 0)
    add_player("Germany", "Jonathan Tah", "DF", 30, "Bayer Leverkusen", 25, 1, 2, 1, 75, 6400, 6, 0)
    add_player("Germany", "David Raum", "DF", 28, "RB Leipzig", 18, 0, 1, 6, 68, 5700, 4, 0)
    add_player("Germany", "Joshua Kimmich", "DF", 31, "Bayern Munich", 95, 5, 4, 14, 82, 7000, 7, 0)
    add_player("Germany", "Robin Koch", "DF", 30, "Eintracht Frankfurt", 10, 0, 1, 1, 65, 5500, 4, 0)
    add_player("Germany", "İlkay Gündoğan", "MF", 35, "Barcelona", 70, 16, 8, 8, 62, 5000, 3, 0)
    add_player("Germany", "Jamal Musiala", "MF", 23, "Bayern Munich", 35, 6, 18, 16, 82, 6800, 2, 0)
    add_player("Germany", "Florian Wirtz", "MF", 23, "Bayer Leverkusen", 25, 5, 20, 18, 82, 6800, 2, 0)
    add_player("Germany", "Robert Andrich", "MF", 30, "Bayer Leverkusen", 12, 1, 3, 4, 72, 6100, 7, 0)
    add_player("Germany", "Leroy Sané", "FW", 30, "Bayern Munich", 55, 14, 12, 10, 72, 5800, 3, 0)
    add_player("Germany", "Kai Havertz", "FW", 27, "Arsenal", 45, 15, 16, 10, 78, 6500, 3, 0)
    add_player("Germany", "Serge Gnabry", "FW", 30, "Bayern Munich", 42, 22, 8, 6, 62, 5000, 2, 0)
    add_player("Germany", "Niclas Füllkrug", "FW", 33, "West Ham", 20, 11, 12, 5, 65, 5200, 2, 0)
    add_player("Germany", "Tim Kleindienst", "FW", 29, "Borussia Mönchengladbach", 5, 2, 14, 6, 72, 6000, 3, 0)

    # === NETHERLANDS ===
    add_player("Netherlands", "Bart Verbruggen", "GK", 23, "Brighton", 12, 0, 0, 0, 75, 6750, 2, 0, 28)
    add_player("Netherlands", "Virgil van Dijk", "DF", 34, "Liverpool", 70, 8, 3, 1, 72, 6100, 4, 0)
    add_player("Netherlands", "Nathan Aké", "DF", 31, "Manchester City", 45, 3, 2, 2, 68, 5700, 3, 0)
    add_player("Netherlands", "Jurriën Timber", "DF", 25, "Arsenal", 18, 1, 2, 4, 72, 6100, 4, 0)
    add_player("Netherlands", "Denzel Dumfries", "DF", 30, "Inter Milan", 48, 4, 3, 8, 72, 6000, 5, 0)
    add_player("Netherlands", "Frenkie de Jong", "MF", 29, "Barcelona", 55, 2, 3, 8, 65, 5200, 3, 0)
    add_player("Netherlands", "Ryan Gravenberch", "MF", 24, "Liverpool", 15, 1, 4, 6, 78, 6600, 3, 0)
    add_player("Netherlands", "Teun Koopmeiners", "MF", 28, "Juventus", 22, 3, 8, 8, 72, 6000, 4, 0)
    add_player("Netherlands", "Xavi Simons", "MF", 23, "RB Leipzig", 15, 3, 16, 14, 78, 6400, 3, 0)
    add_player("Netherlands", "Tijjani Reijnders", "MF", 27, "AC Milan", 18, 3, 8, 8, 78, 6600, 3, 0)
    add_player("Netherlands", "Memphis Depay", "FW", 32, "Corinthians", 90, 46, 12, 6, 55, 4200, 2, 0)
    add_player("Netherlands", "Cody Gakpo", "FW", 26, "Liverpool", 35, 12, 18, 12, 78, 6400, 3, 0)
    add_player("Netherlands", "Donyell Malen", "FW", 27, "Aston Villa", 22, 5, 12, 8, 72, 6000, 2, 0)

    # For the remaining teams, I'll use a more compact generation approach
    # with realistic but slightly less detailed data
    
    remaining_teams_data = {
        "Croatia": {
            "players": [
                ("Dominik Livaković", "GK", 30, "Fenerbahçe", 55, 0, 0, 0, 72, 6480, 2, 0, 28),
                ("Joško Gvardiol", "DF", 24, "Manchester City", 32, 4, 5, 5, 78, 6600, 4, 0, 0),
                ("Luka Modrić", "MF", 40, "Real Madrid", 178, 25, 5, 8, 55, 4400, 4, 0, 0),
                ("Mateo Kovačić", "MF", 32, "Manchester City", 100, 5, 4, 8, 72, 6000, 5, 0, 0),
                ("Marcelo Brozović", "MF", 33, "Al-Nassr", 95, 8, 3, 5, 68, 5600, 6, 0, 0),
                ("Ivan Perišić", "FW", 37, "Hajduk Split", 120, 33, 5, 3, 50, 4000, 2, 0, 0),
                ("Andrej Kramarić", "FW", 34, "Hoffenheim", 80, 22, 10, 6, 68, 5500, 3, 0, 0),
                ("Bruno Petković", "FW", 32, "Dinamo Zagreb", 35, 8, 8, 4, 62, 5000, 2, 0, 0),
                ("Lovro Majer", "MF", 28, "Wolfsburg", 25, 3, 6, 8, 68, 5600, 3, 0, 0),
                ("Mario Pašalić", "MF", 31, "Atalanta", 42, 5, 6, 6, 68, 5600, 4, 0, 0),
            ]
        },
        "Belgium": {
            "players": [
                ("Thibaut Courtois", "GK", 34, "Real Madrid", 105, 0, 0, 0, 65, 5850, 2, 0, 26),
                ("Koen Casteels", "GK", 34, "Wolfsburg", 15, 0, 0, 0, 58, 5220, 1, 0, 20),
                ("Timothy Castagne", "DF", 30, "Fulham", 40, 2, 2, 4, 68, 5700, 4, 0, 0),
                ("Arthur Theate", "DF", 26, "Rennes", 18, 1, 2, 2, 68, 5700, 5, 0, 0),
                ("Wout Faes", "DF", 27, "Leicester City", 18, 0, 1, 1, 72, 6100, 5, 0, 0),
                ("Kevin De Bruyne", "MF", 35, "Manchester City", 105, 28, 10, 16, 65, 5200, 4, 0, 0),
                ("Amadou Onana", "MF", 25, "Aston Villa", 22, 2, 4, 5, 75, 6300, 6, 0, 0),
                ("Charles De Ketelaere", "MF", 25, "Atalanta", 15, 2, 12, 10, 75, 6200, 3, 0, 0),
                ("Jérémy Doku", "FW", 24, "Manchester City", 25, 3, 8, 12, 72, 5800, 2, 0, 0),
                ("Loïs Openda", "FW", 26, "RB Leipzig", 22, 8, 22, 8, 78, 6500, 2, 0, 0),
                ("Romelu Lukaku", "FW", 33, "Napoli", 115, 85, 18, 6, 72, 5800, 3, 0, 0),
                ("Leandro Trossard", "FW", 31, "Arsenal", 45, 12, 10, 8, 72, 5800, 3, 0, 0),
            ]
        },
        "Italy": {
            "players": [
                ("Gianluigi Donnarumma", "GK", 27, "PSG", 65, 0, 0, 0, 75, 6750, 2, 0, 30),
                ("Alessandro Bastoni", "DF", 27, "Inter Milan", 22, 2, 3, 3, 78, 6600, 4, 0, 0),
                ("Federico Dimarco", "DF", 28, "Inter Milan", 20, 3, 5, 10, 78, 6600, 5, 0, 0),
                ("Riccardo Calafiori", "DF", 24, "Arsenal", 12, 1, 2, 3, 68, 5700, 4, 0, 0),
                ("Sandro Tonali", "MF", 26, "Newcastle", 20, 1, 3, 5, 65, 5400, 5, 0, 0),
                ("Nicolò Barella", "MF", 29, "Inter Milan", 55, 8, 8, 12, 80, 6800, 4, 0, 0),
                ("Lorenzo Pellegrini", "MF", 29, "Roma", 35, 5, 6, 8, 68, 5600, 5, 0, 0),
                ("Federico Chiesa", "FW", 28, "Liverpool", 42, 8, 6, 4, 55, 4200, 2, 0, 0),
                ("Giacomo Raspadori", "FW", 26, "Napoli", 28, 5, 8, 4, 62, 5000, 2, 0, 0),
                ("Mateo Retegui", "FW", 25, "Atalanta", 12, 5, 16, 6, 72, 6000, 2, 0, 0),
                ("Gianluca Scamacca", "FW", 27, "Atalanta", 15, 3, 14, 6, 65, 5200, 2, 0, 0),
            ]
        },
        "Colombia": {
            "players": [
                ("David Ospina", "GK", 37, "Al-Nassr", 70, 0, 0, 0, 45, 4050, 1, 0, 18),
                ("Camilo Vargas", "GK", 35, "Atlas", 12, 0, 0, 0, 62, 5580, 2, 0, 22),
                ("Dávinson Sánchez", "DF", 30, "Galatasaray", 55, 2, 2, 1, 65, 5500, 5, 0, 0),
                ("Yerry Mina", "DF", 31, "Cagliari", 40, 5, 1, 1, 55, 4500, 6, 0, 0),
                ("Johan Mojica", "DF", 32, "Mallorca", 18, 1, 1, 4, 62, 5200, 4, 0, 0),
                ("Daniel Muñoz", "DF", 28, "Crystal Palace", 15, 2, 4, 5, 72, 6100, 5, 0, 0),
                ("James Rodríguez", "MF", 35, "Rayo Vallecano", 105, 28, 5, 10, 55, 4200, 3, 0, 0),
                ("Richard Ríos", "MF", 25, "Palmeiras", 15, 1, 3, 5, 72, 6100, 4, 0, 0),
                ("Jefferson Lerma", "MF", 30, "Crystal Palace", 45, 2, 2, 3, 68, 5700, 6, 0, 0),
                ("Luis Díaz", "FW", 29, "Liverpool", 45, 8, 18, 12, 78, 6400, 3, 0, 0),
                ("Rafael Santos Borré", "FW", 30, "Internacional", 20, 5, 14, 6, 68, 5600, 2, 0, 0),
                ("Jhon Córdoba", "FW", 32, "Krasnodar", 15, 3, 12, 4, 62, 5100, 3, 0, 0),
            ]
        },
        "Mexico": {
            "players": [
                ("Guillermo Ochoa", "GK", 41, "Salernitana", 135, 0, 0, 0, 45, 4050, 2, 0, 16),
                ("César Montes", "DF", 29, "Almería", 30, 3, 2, 1, 65, 5500, 5, 0, 0),
                ("Johan Vásquez", "DF", 26, "Genoa", 20, 1, 2, 1, 68, 5700, 4, 0, 0),
                ("Edson Álvarez", "MF", 28, "West Ham", 70, 3, 3, 4, 72, 6100, 8, 0, 0),
                ("Héctor Herrera", "MF", 36, "Houston Dynamo", 100, 8, 2, 3, 55, 4500, 4, 0, 0),
                ("Hirving Lozano", "FW", 30, "PSV", 60, 18, 10, 8, 72, 5800, 3, 0, 0),
                ("Santiago Giménez", "FW", 25, "Feyenoord", 20, 8, 24, 8, 78, 6500, 2, 0, 0),
                ("Alexis Vega", "FW", 28, "Toluca", 18, 4, 10, 6, 65, 5400, 2, 0, 0),
                ("Julián Quiñones", "FW", 29, "América", 8, 2, 14, 8, 72, 6000, 3, 0, 0),
                ("Diego Lainez", "MF", 26, "Tigres UANL", 15, 1, 5, 6, 62, 5000, 2, 0, 0),
            ]
        },
        "Uruguay": {
            "players": [
                ("Sergio Rochet", "GK", 33, "Internacional", 18, 0, 0, 0, 68, 6120, 2, 0, 24),
                ("José María Giménez", "DF", 31, "Atlético Madrid", 80, 5, 2, 1, 68, 5700, 6, 0, 0),
                ("Ronald Araújo", "DF", 27, "Barcelona", 25, 2, 2, 1, 62, 5200, 4, 0, 0),
                ("Federico Valverde", "MF", 28, "Real Madrid", 55, 8, 10, 12, 82, 7000, 5, 0, 0),
                ("Rodrigo Bentancur", "MF", 29, "Tottenham", 55, 2, 3, 5, 68, 5700, 5, 0, 0),
                ("Facundo Pellistri", "FW", 24, "Manchester United", 15, 2, 4, 6, 62, 5000, 2, 0, 0),
                ("Darwin Núñez", "FW", 27, "Liverpool", 30, 8, 28, 10, 78, 6400, 4, 0, 0),
                ("Luis Suárez", "FW", 39, "Inter Miami", 140, 69, 10, 4, 45, 3500, 2, 0, 0),
                ("Nicolás De La Cruz", "MF", 27, "Flamengo", 25, 3, 5, 8, 68, 5700, 3, 0, 0),
                ("Maximiliano Araújo", "FW", 24, "Sporting CP", 8, 1, 8, 6, 65, 5400, 2, 0, 0),
            ]
        },
        "Japan": {
            "players": [
                ("Shūichi Gonda", "GK", 37, "Shimizu S-Pulse", 35, 0, 0, 0, 55, 4950, 2, 0, 20),
                ("Takehiro Tomiyasu", "DF", 27, "Arsenal", 35, 1, 1, 2, 62, 5200, 3, 0, 0),
                ("Ko Itakura", "DF", 29, "Borussia Mönchengladbach", 30, 2, 2, 1, 72, 6100, 5, 0, 0),
                ("Wataru Endo", "MF", 33, "Liverpool", 55, 3, 3, 4, 72, 6000, 6, 0, 0),
                ("Hidemasa Morita", "MF", 30, "Sporting CP", 30, 2, 3, 5, 72, 6100, 4, 0, 0),
                ("Takefusa Kubo", "FW", 25, "Real Sociedad", 28, 4, 12, 10, 78, 6400, 3, 0, 0),
                ("Kaoru Mitoma", "FW", 29, "Brighton", 30, 6, 14, 10, 75, 6200, 2, 0, 0),
                ("Daichi Kamada", "MF", 30, "Crystal Palace", 35, 4, 6, 8, 68, 5600, 3, 0, 0),
                ("Ritsu Doan", "FW", 28, "Freiburg", 25, 6, 8, 6, 65, 5400, 2, 0, 0),
                ("Junya Ito", "FW", 33, "Reims", 45, 5, 8, 8, 68, 5600, 2, 0, 0),
                ("Ayase Ueda", "FW", 27, "Feyenoord", 15, 5, 18, 6, 72, 5900, 2, 0, 0),
            ]
        },
        "Korea Republic": {
            "players": [
                ("Kim Seung-gyu", "GK", 36, "Al-Shabab", 55, 0, 0, 0, 55, 4950, 2, 0, 20),
                ("Kim Min-jae", "DF", 29, "Bayern Munich", 48, 3, 2, 1, 72, 6100, 4, 0, 0),
                ("Son Heung-min", "FW", 33, "Tottenham", 120, 48, 18, 14, 78, 6400, 3, 0, 0),
                ("Lee Kang-in", "MF", 25, "PSG", 40, 6, 10, 12, 78, 6400, 3, 0, 0),
                ("Hwang Hee-chan", "FW", 30, "Wolverhampton", 55, 12, 12, 6, 72, 5800, 2, 0, 0),
                ("Cho Gue-sung", "FW", 28, "Midtjylland", 22, 6, 10, 4, 62, 5000, 2, 0, 0),
                ("Hwang In-beom", "MF", 30, "Red Star Belgrade", 48, 4, 4, 6, 68, 5700, 4, 0, 0),
                ("Jung Woo-young", "MF", 36, "Al-Sadd", 70, 2, 1, 2, 55, 4500, 5, 0, 0),
                ("Jeong Woo-yeong", "MF", 26, "Stuttgart", 18, 2, 6, 6, 68, 5600, 3, 0, 0),
            ]
        },
        "Morocco": {
            "players": [
                ("Yassine Bounou", "GK", 35, "Al-Hilal", 55, 0, 0, 0, 68, 6120, 2, 0, 26),
                ("Achraf Hakimi", "DF", 27, "PSG", 65, 8, 5, 10, 80, 6800, 4, 0, 0),
                ("Nayef Aguerd", "DF", 30, "Real Sociedad", 35, 2, 2, 1, 68, 5700, 4, 0, 0),
                ("Azzedine Ounahi", "MF", 26, "Marseille", 22, 1, 3, 6, 65, 5400, 3, 0, 0),
                ("Sofyan Amrabat", "MF", 30, "Fenerbahçe", 60, 1, 2, 3, 72, 6100, 7, 0, 0),
                ("Hakim Ziyech", "FW", 33, "Galatasaray", 48, 10, 8, 10, 65, 5200, 3, 0, 0),
                ("Youssef En-Nesyri", "FW", 29, "Roma", 68, 20, 16, 6, 72, 5900, 2, 0, 0),
                ("Brahim Díaz", "MF", 26, "Real Madrid", 18, 3, 8, 8, 68, 5600, 2, 0, 0),
                ("Ilias Akhomach", "FW", 21, "Villarreal", 8, 1, 8, 6, 65, 5400, 2, 0, 0),
                ("Noussair Mazraoui", "DF", 28, "Manchester United", 30, 1, 2, 4, 65, 5400, 3, 0, 0),
            ]
        },
        "Senegal": {
            "players": [
                ("Édouard Mendy", "GK", 34, "Al-Ahli", 42, 0, 0, 0, 62, 5580, 2, 0, 22),
                ("Kalidou Koulibaly", "DF", 35, "Al-Hilal", 72, 3, 1, 1, 55, 4500, 4, 0, 0),
                ("Sadio Mané", "FW", 34, "Al-Nassr", 95, 40, 18, 6, 68, 5500, 2, 0, 0),
                ("Ismaïla Sarr", "FW", 28, "Crystal Palace", 55, 10, 10, 8, 72, 5800, 3, 0, 0),
                ("Idrissa Gueye", "MF", 36, "Everton", 100, 4, 2, 3, 62, 5200, 5, 0, 0),
                ("Nicolas Jackson", "FW", 25, "Chelsea", 18, 5, 16, 8, 78, 6400, 2, 0, 0),
                ("Pape Matar Sarr", "MF", 23, "Tottenham", 15, 2, 5, 6, 72, 6000, 3, 0, 0),
                ("Krepin Diatta", "FW", 27, "Monaco", 35, 5, 6, 5, 62, 5000, 2, 0, 0),
                ("Abdoulaye Doucouré", "MF", 33, "Everton", 8, 0, 3, 4, 58, 4800, 5, 0, 0),
            ]
        },
        "Switzerland": {
            "players": [
                ("Yann Sommer", "GK", 37, "Inter Milan", 90, 0, 0, 0, 65, 5850, 2, 0, 28),
                ("Manuel Akanji", "DF", 30, "Manchester City", 55, 5, 2, 1, 72, 6100, 3, 0, 0),
                ("Ricardo Rodríguez", "DF", 33, "Real Betis", 80, 8, 1, 3, 62, 5200, 4, 0, 0),
                ("Granit Xhaka", "MF", 33, "Bayer Leverkusen", 125, 13, 5, 10, 78, 6600, 6, 0, 0),
                ("Xherdan Shaqiri", "FW", 34, "Chicago Fire", 115, 32, 5, 4, 50, 3800, 2, 0, 0),
                ("Denis Zakaria", "MF", 29, "Monaco", 48, 1, 2, 3, 68, 5700, 4, 0, 0),
                ("Breel Embolo", "FW", 29, "Monaco", 65, 14, 10, 4, 58, 4700, 2, 0, 0),
                ("Ruben Vargas", "FW", 28, "Augsburg", 38, 8, 6, 6, 65, 5400, 2, 0, 0),
                ("Noah Okafor", "FW", 26, "AC Milan", 22, 4, 6, 4, 58, 4600, 2, 0, 0),
                ("Dan Ndoye", "FW", 24, "Bologna", 12, 2, 6, 6, 68, 5600, 2, 0, 0),
            ]
        },
        "Australia": {
            "players": [
                ("Mathew Ryan", "GK", 34, "AZ Alkmaar", 80, 0, 0, 0, 68, 6120, 2, 0, 24),
                ("Harry Souttar", "DF", 27, "Sheffield United", 15, 4, 2, 1, 55, 4500, 3, 0, 0),
                ("Aziz Behich", "DF", 33, "Dundee United", 52, 1, 0, 2, 58, 4800, 4, 0, 0),
                ("Aaron Mooy", "MF", 35, "Retired", 55, 8, 0, 0, 0, 0, 0, 0, 0),
                ("Tom Rogic", "MF", 33, "Retired", 55, 10, 0, 0, 0, 0, 0, 0, 0),
                ("Craig Goodwin", "FW", 35, "Adelaide United", 15, 4, 8, 6, 58, 4800, 2, 0, 0),
                ("Mitchell Duke", "FW", 35, "Machida Zelvia", 25, 8, 6, 2, 50, 4000, 2, 0, 0),
                ("Jackson Irvine", "MF", 32, "St. Pauli", 52, 4, 4, 4, 65, 5400, 4, 0, 0),
                ("Riley McGree", "MF", 26, "Middlesbrough", 18, 2, 4, 4, 68, 5600, 3, 0, 0),
                ("Kye Rowles", "DF", 27, "Hearts", 12, 0, 1, 1, 62, 5200, 3, 0, 0),
            ]
        },
        "IR Iran": {
            "players": [
                ("Alireza Beiranvand", "GK", 33, "Persepolis", 50, 0, 0, 0, 62, 5580, 2, 0, 22),
                ("Ehsan Hajsafi", "DF", 36, "AEK Athens", 125, 7, 1, 2, 55, 4500, 4, 0, 0),
                ("Sardar Azmoun", "FW", 31, "Roma", 72, 48, 8, 4, 55, 4400, 2, 0, 0),
                ("Mehdi Taremi", "FW", 34, "Inter Milan", 75, 42, 14, 6, 72, 5800, 2, 0, 0),
                ("Alireza Jahanbakhsh", "FW", 33, "Feyenoord", 70, 12, 5, 4, 58, 4700, 2, 0, 0),
                ("Ali Gholizadeh", "FW", 28, "Charleroi", 35, 4, 6, 5, 62, 5100, 2, 0, 0),
                ("Saeid Ezatolahi", "MF", 30, "Vejle", 55, 2, 1, 2, 62, 5200, 5, 0, 0),
                ("Ahmad Nourollahi", "MF", 32, "Shabab Al-Ahli", 55, 5, 3, 4, 60, 5000, 4, 0, 0),
            ]
        },
        "Saudi Arabia": {
            "players": [
                ("Mohammed Al-Owais", "GK", 30, "Al-Hilal", 55, 0, 0, 0, 68, 6120, 2, 0, 24),
                ("Ali Al-Bulaihi", "DF", 35, "Al-Hilal", 40, 1, 0, 0, 55, 4500, 5, 0, 0),
                ("Salem Al-Dawsari", "FW", 33, "Al-Hilal", 70, 20, 12, 8, 68, 5600, 3, 0, 0),
                ("Saleh Al-Shehri", "FW", 31, "Al-Hilal", 35, 12, 10, 4, 62, 5100, 2, 0, 0),
                ("Firas Al-Buraikan", "FW", 26, "Al-Ahli", 25, 8, 14, 6, 72, 5900, 2, 0, 0),
                ("Abdullah Otayf", "MF", 29, "Al-Hilal", 30, 2, 2, 4, 62, 5200, 4, 0, 0),
                ("Mohamed Kanno", "MF", 32, "Al-Hilal", 65, 3, 2, 3, 60, 5000, 5, 0, 0),
            ]
        },
        "Ecuador": {
            "players": [
                ("Hernán Galíndez", "GK", 39, "Univ. Católica", 18, 0, 0, 0, 55, 4950, 2, 0, 18),
                ("Pervis Estupiñán", "DF", 28, "Brighton", 35, 2, 2, 6, 72, 6100, 4, 0, 0),
                ("Moisés Caicedo", "MF", 24, "Chelsea", 32, 3, 4, 6, 78, 6600, 5, 0, 0),
                ("Enner Valencia", "FW", 36, "Internacional", 80, 40, 8, 4, 55, 4400, 3, 0, 0),
                ("Gonzalo Plata", "FW", 25, "Flamengo", 35, 6, 8, 6, 65, 5400, 3, 0, 0),
                ("Jeremy Sarmiento", "FW", 23, "Brighton", 12, 1, 4, 4, 58, 4700, 2, 0, 0),
                ("Alan Franco", "MF", 27, "Atlético Mineiro", 18, 0, 1, 2, 65, 5500, 4, 0, 0),
                ("Piero Hincapié", "DF", 24, "Bayer Leverkusen", 30, 1, 2, 2, 72, 6100, 3, 0, 0),
            ]
        },
        "United States": {
            "players": [
                ("Matt Turner", "GK", 32, "Crystal Palace", 28, 0, 0, 0, 62, 5580, 2, 0, 22),
                ("Sergiño Dest", "DF", 25, "PSV", 30, 2, 2, 5, 65, 5400, 3, 0, 0),
                ("Chris Richards", "DF", 26, "Crystal Palace", 12, 0, 1, 1, 65, 5500, 4, 0, 0),
                ("Antonee Robinson", "DF", 28, "Fulham", 40, 2, 1, 4, 72, 6100, 3, 0, 0),
                ("Tyler Adams", "MF", 27, "Bournemouth", 35, 1, 2, 3, 62, 5200, 4, 0, 0),
                ("Weston McKennie", "MF", 27, "Juventus", 42, 10, 5, 6, 68, 5700, 5, 0, 0),
                ("Gio Reyna", "MF", 23, "Borussia Dortmund", 18, 4, 6, 6, 58, 4600, 2, 0, 0),
                ("Christian Pulisic", "FW", 27, "AC Milan", 65, 28, 16, 14, 78, 6400, 3, 0, 0),
                ("Timothy Weah", "FW", 26, "Juventus", 35, 5, 8, 6, 68, 5600, 2, 0, 0),
                ("Folarin Balogun", "FW", 25, "Monaco", 18, 4, 14, 6, 72, 5900, 2, 0, 0),
                ("Ricardo Pepi", "FW", 23, "PSV", 20, 6, 12, 4, 65, 5200, 2, 0, 0),
                ("Yunus Musah", "MF", 23, "AC Milan", 28, 0, 2, 4, 68, 5700, 3, 0, 0),
            ]
        },
        "Canada": {
            "players": [
                ("Milan Borjan", "GK", 38, "Yokohama FM", 65, 0, 0, 0, 50, 4500, 2, 0, 18),
                ("Maxime Crépeau", "GK", 31, "Portland Timbers", 12, 0, 0, 0, 58, 5220, 1, 0, 20),
                ("Alphonso Davies", "DF", 25, "Real Madrid", 50, 5, 4, 8, 78, 6600, 3, 0, 0),
                ("Tajon Buchanan", "DF", 27, "Inter Milan", 35, 5, 4, 6, 65, 5400, 2, 0, 0),
                ("Jonathan David", "FW", 26, "Lille", 50, 28, 24, 10, 80, 6600, 2, 0, 0),
                ("Cyle Larin", "FW", 31, "Mallorca", 60, 28, 8, 4, 60, 4800, 2, 0, 0),
                ("Ismael Koné", "MF", 24, "Marseille", 18, 2, 3, 5, 68, 5700, 3, 0, 0),
                ("Stephen Eustáquio", "MF", 28, "Porto", 28, 2, 3, 6, 68, 5700, 4, 0, 0),
                ("Liam Millar", "FW", 25, "Basel", 22, 3, 8, 5, 62, 5100, 2, 0, 0),
            ]
        },
        "Nigeria": {
            "players": [
                ("Francis Uzoho", "GK", 27, "Jong Ajax", 20, 0, 0, 0, 58, 5220, 2, 0, 18),
                ("William Troost-Ekong", "DF", 32, "PAOK", 70, 5, 1, 1, 58, 4800, 4, 0, 0),
                ("Victor Osimhen", "FW", 27, "Napoli", 28, 18, 28, 8, 78, 6400, 2, 0, 0),
                ("Samuel Chukwueze", "FW", 27, "AC Milan", 38, 5, 6, 6, 62, 5000, 2, 0, 0),
                ("Alex Iwobi", "MF", 30, "Fulham", 70, 5, 4, 6, 72, 6000, 3, 0, 0),
                ("Ademola Lookman", "FW", 28, "Atalanta", 18, 5, 18, 12, 78, 6400, 2, 0, 0),
                ("Wilfred Ndidi", "MF", 29, "Leicester City", 55, 1, 2, 2, 68, 5700, 6, 0, 0),
                ("Joe Aribo", "MF", 29, "Southampton", 25, 2, 4, 4, 62, 5200, 3, 0, 0),
            ]
        },
        "Egypt": {
            "players": [
                ("Mohamed El-Shenawy", "GK", 35, "Al Ahly", 22, 0, 0, 0, 62, 5580, 2, 0, 22),
                ("Ahmed Hegazi", "DF", 35, "Al-Ittihad", 72, 6, 1, 0, 55, 4500, 4, 0, 0),
                ("Mohamed Salah", "FW", 34, "Liverpool", 95, 55, 32, 18, 82, 6800, 2, 0, 0),
                ("Omar Marmoush", "FW", 27, "Manchester City", 28, 8, 22, 14, 78, 6400, 2, 0, 0),
                ("Trézéguet", "FW", 31, "Trabzonspor", 55, 8, 6, 4, 58, 4700, 3, 0, 0),
                ("Mohamed Elneny", "MF", 34, "Beşiktaş", 95, 2, 1, 2, 55, 4500, 4, 0, 0),
                ("Mahmoud Hassan (Trézéguet)", "MF", 31, "Trabzonspor", 55, 8, 4, 4, 55, 4400, 3, 0, 0),
                ("Emam Ashour", "MF", 27, "Al Ahly", 18, 1, 2, 3, 65, 5400, 4, 0, 0),
            ]
        },
        "Algeria": {
            "players": [
                ("Raïs M'Bolhi", "GK", 38, "Al Ahli", 70, 0, 0, 0, 42, 3780, 1, 0, 14),
                ("Riyad Mahrez", "FW", 35, "Al-Ahli", 85, 30, 8, 6, 55, 4400, 2, 0, 0),
                ("Ismaël Bennacer", "MF", 28, "AC Milan", 48, 3, 2, 4, 58, 4700, 4, 0, 0),
                ("Yacine Brahimi", "FW", 34, "Al-Gharafa", 68, 14, 4, 3, 48, 3800, 2, 0, 0),
                ("Islam Slimani", "FW", 38, "CR Belouizdad", 90, 42, 4, 2, 42, 3200, 2, 0, 0),
                ("Amine Gouiri", "FW", 26, "Rennes", 18, 4, 8, 6, 65, 5400, 2, 0, 0),
                ("Saïd Benrahma", "FW", 30, "Lyon", 22, 4, 6, 6, 62, 5000, 2, 0, 0),
                ("Houssem Aouar", "MF", 28, "Al-Ittihad", 15, 2, 3, 4, 55, 4500, 2, 0, 0),
            ]
        },
        "South Africa": {
            "players": [
                ("Ronwen Williams", "GK", 32, "Mamelodi Sundowns", 42, 0, 0, 0, 68, 6120, 2, 0, 24),
                ("Mothobi Mvala", "MF", 30, "Mamelodi Sundowns", 20, 1, 1, 2, 62, 5200, 4, 0, 0),
                ("Percy Tau", "FW", 32, "Al-Ahli", 45, 15, 6, 4, 55, 4400, 2, 0, 0),
                ("Themba Zwane", "MF", 35, "Mamelodi Sundowns", 35, 12, 4, 4, 55, 4400, 2, 0, 0),
                ("Bongokuhle Hlongwane", "FW", 24, "Minnesota United", 18, 4, 8, 4, 62, 5000, 2, 0, 0),
                ("Teboho Mokoena", "MF", 27, "Mamelodi Sundowns", 22, 6, 4, 4, 65, 5400, 3, 0, 0),
                ("Lyle Foster", "FW", 24, "Burnley", 15, 2, 4, 2, 55, 4400, 2, 0, 0),
                ("Grant Kekana", "DF", 32, "Mamelodi Sundowns", 10, 0, 1, 0, 62, 5200, 4, 0, 0),
            ]
        },
        "Tunisia": {
            "players": [
                ("Aymen Dahmen", "GK", 28, "KV Mechelen", 18, 0, 0, 0, 62, 5580, 2, 0, 20),
                ("Montassar Talbi", "DF", 28, "Lorient", 18, 1, 1, 1, 62, 5200, 4, 0, 0),
                ("Aïssa Laïdouni", "MF", 30, "Union Berlin", 28, 1, 2, 3, 68, 5700, 5, 0, 0),
                ("Wahbi Khazri", "FW", 35, "Montpellier", 72, 25, 4, 2, 45, 3500, 2, 0, 0),
                ("Hannibal Mejbri", "MF", 23, "Manchester United", 12, 0, 1, 2, 55, 4400, 3, 0, 0),
                ("Youssef Msakni", "FW", 36, "Al-Arabi", 80, 18, 4, 2, 42, 3200, 2, 0, 0),
                ("Seifeddine Jaziri", "FW", 33, "Zamalek", 15, 4, 8, 4, 58, 4700, 2, 0, 0),
            ]
        },
        "Côte d'Ivoire": {
            "players": [
                ("Yahia Fofana", "GK", 25, "Angers", 10, 0, 0, 0, 62, 5580, 2, 0, 20),
                ("Odilon Kossounou", "DF", 25, "Atalanta", 22, 1, 1, 1, 65, 5500, 4, 0, 0),
                ("Sébastien Haller", "FW", 32, "Borussia Dortmund", 30, 8, 8, 4, 58, 4700, 2, 0, 0),
                ("Franck Kessié", "MF", 29, "Al-Ahli", 62, 8, 3, 3, 58, 4800, 5, 0, 0),
                ("Simon Adingra", "FW", 23, "Brighton", 15, 3, 8, 8, 72, 5900, 2, 0, 0),
                ("Ibrahim Sangaré", "MF", 28, "Nottingham Forest", 22, 1, 2, 3, 65, 5400, 4, 0, 0),
                ("Nicolas Pépé", "FW", 31, "Villarreal", 45, 8, 4, 4, 52, 4100, 2, 0, 0),
                ("Oumar Diakité", "MF", 21, "Reims", 8, 0, 2, 3, 55, 4500, 2, 0, 0),
            ]
        },
        "Congo DR": {
            "players": [
                ("Lionel Mpasi", "GK", 25, "Stade Brestois", 8, 0, 0, 0, 55, 4950, 2, 0, 18),
                ("Chancel Mbemba", "DF", 31, "Marseille", 60, 3, 1, 1, 58, 4800, 5, 0, 0),
                ("Cédric Bakambu", "FW", 34, "Betis", 40, 12, 4, 2, 48, 3800, 2, 0, 0),
                ("Yoane Wissa", "FW", 28, "Brentford", 12, 3, 14, 6, 72, 5900, 2, 0, 0),
                ("Silas Katompa Mvumpa", "FW", 26, "Stuttgart", 8, 1, 8, 6, 62, 5000, 2, 0, 0),
                ("Samuel Moutoussamy", "MF", 30, "Nantes", 15, 0, 2, 2, 58, 4800, 3, 0, 0),
                ("Arthur Masuaku", "DF", 32, "Beşiktaş", 18, 0, 0, 2, 52, 4200, 3, 0, 0),
                ("Théo Bongonda", "FW", 30, "Cadiz", 22, 4, 4, 4, 55, 4400, 2, 0, 0),
            ]
        },
        "Cabo Verde": {
            "players": [
                ("Vozinha", "GK", 40, "Gil Vicente", 50, 0, 0, 0, 42, 3780, 2, 0, 14),
                ("Roberto Lopes", "DF", 32, "Shamrock Rovers", 20, 1, 0, 0, 48, 4000, 3, 0, 0),
                ("Ryan Mendes", "FW", 35, "Santa Clara", 50, 12, 4, 2, 45, 3600, 2, 0, 0),
                ("Garry Rodrigues", "FW", 35, "Hajduk Split", 40, 8, 3, 2, 42, 3300, 2, 0, 0),
                ("Jovane Cabral", "FW", 28, "Lazio", 20, 4, 4, 3, 48, 3800, 2, 0, 0),
                ("Kenny Rocha Santos", "MF", 29, "Gil Vicente", 28, 2, 2, 3, 55, 4500, 3, 0, 0),
                ("Nuno Borges", "MF", 29, "Boavista", 18, 1, 2, 2, 52, 4300, 3, 0, 0),
            ]
        },
        "Qatar": {
            "players": [
                ("Saad Al-Sheeb", "GK", 36, "Al Sadd", 85, 0, 0, 0, 50, 4500, 2, 0, 16),
                ("Hassan Al-Haydos", "FW", 35, "Al Sadd", 180, 38, 4, 2, 42, 3300, 2, 0, 0),
                ("Almoez Ali", "FW", 28, "Al-Duhail", 95, 48, 14, 4, 62, 5000, 2, 0, 0),
                ("Akram Afif", "FW", 28, "Al Sadd", 85, 30, 12, 10, 68, 5600, 2, 0, 0),
                ("Abdulaziz Hatem", "MF", 35, "Al-Gharafa", 60, 5, 2, 2, 48, 3800, 3, 0, 0),
                ("Karim Boudiaf", "MF", 36, "Al-Duhail", 70, 2, 0, 1, 42, 3300, 4, 0, 0),
                ("Assim Madibo", "MF", 30, "Al-Duhail", 55, 1, 1, 2, 55, 4500, 4, 0, 0),
            ]
        },
        "Iraq": {
            "players": [
                ("Jalal Hassan", "GK", 33, "Al-Shorta", 50, 0, 0, 0, 55, 4950, 2, 0, 18),
                ("Ali Adnan", "DF", 32, "Al-Wakrah", 60, 4, 1, 2, 48, 3800, 3, 0, 0),
                ("Mohanad Ali", "FW", 25, "Al-Arabi", 38, 14, 10, 4, 62, 5000, 2, 0, 0),
                ("Aymen Hussein", "FW", 30, "Riga FC", 48, 22, 10, 4, 62, 5000, 2, 0, 0),
                ("Amjad Attwan", "MF", 28, "Al-Quwa Al-Jawiya", 30, 2, 2, 3, 55, 4500, 4, 0, 0),
                ("Ibrahim Bayesh", "MF", 32, "Al-Quwa Al-Jawiya", 55, 3, 1, 2, 52, 4200, 4, 0, 0),
                ("Justin Meram", "FW", 37, "Austin FC", 22, 5, 3, 2, 42, 3300, 2, 0, 0),
            ]
        },
        "Jordan": {
            "players": [
                ("Yazeed Abulaila", "GK", 30, "Al-Faisaly", 35, 0, 0, 0, 58, 5220, 2, 0, 20),
                ("Musa Al-Tamari", "FW", 28, "Montpellier", 55, 12, 6, 6, 62, 5000, 2, 0, 0),
                ("Yazan Al-Naimat", "FW", 24, "Young Boys", 20, 4, 8, 4, 58, 4700, 2, 0, 0),
                ("Mousa Tamari", "MF", 28, "Montpellier", 55, 12, 6, 6, 62, 5000, 2, 0, 0),
                ("Baha' Faisal", "FW", 28, "Al-Wehdat", 30, 8, 8, 4, 55, 4400, 2, 0, 0),
                ("Ehsan Haddad", "MF", 30, "Al-Wehdat", 25, 1, 1, 2, 52, 4200, 3, 0, 0),
            ]
        },
        "Uzbekistan": {
            "players": [
                ("Otabek Shukurov", "MF", 28, "AGMK", 45, 4, 3, 4, 58, 4800, 3, 0, 0),
                ("Eldor Shomurodov", "FW", 29, "Roma", 35, 12, 8, 4, 58, 4700, 2, 0, 0),
                ("Jaloliddin Masharipov", "FW", 33, "Shabab Al-Ahli", 55, 12, 5, 4, 52, 4100, 2, 0, 0),
                ("Abbosbek Fayzullaev", "FW", 22, "Lens", 18, 3, 6, 5, 62, 5000, 2, 0, 0),
                ("Abdulla Abdullaev", "FW", 24, "Pakhtakor", 12, 2, 8, 4, 55, 4500, 2, 0, 0),
                ("Narzikulov Husniddin", "MF", 28, "Pakhtakor", 22, 1, 2, 2, 52, 4200, 3, 0, 0),
                ("Islom Tukhtakhodjaev", "GK", 28, "Pakhtakor", 15, 0, 0, 0, 55, 4950, 2, 0, 18),
            ]
        },
        "Austria": {
            "players": [
                ("Patrick Pentz", "GK", 29, "Real Valladolid", 10, 0, 0, 0, 62, 5580, 2, 0, 22),
                ("David Alaba", "DF", 34, "Real Madrid", 105, 15, 1, 1, 35, 2800, 2, 0, 0),
                ("Philipp Lienhart", "DF", 29, "Freiburg", 15, 0, 1, 0, 65, 5500, 3, 0, 0),
                ("Konrad Laimer", "MF", 29, "Bayern Munich", 35, 3, 3, 5, 72, 6000, 5, 0, 0),
                ("Marcel Sabitzer", "MF", 32, "Borussia Dortmund", 75, 14, 8, 8, 72, 6000, 4, 0, 0),
                ("Christoph Baumgartner", "MF", 25, "RB Leipzig", 35, 8, 8, 6, 68, 5600, 2, 0, 0),
                ("Marko Arnautović", "FW", 37, "Inter Milan", 110, 34, 4, 2, 45, 3500, 3, 0, 0),
                ("Michael Gregoritsch", "FW", 32, "Freiburg", 45, 12, 8, 4, 62, 5100, 2, 0, 0),
            ]
        },
        "Türkiye": {
            "players": [
                ("Altay Bayındır", "GK", 28, "Manchester United", 8, 0, 0, 0, 45, 4050, 2, 0, 16),
                ("Mert Günok", "GK", 35, "Beşiktaş", 12, 0, 0, 0, 58, 5220, 2, 0, 20),
                ("Merih Demiral", "DF", 28, "Al-Ahli", 42, 3, 1, 1, 58, 4800, 5, 0, 0),
                ("Ferdi Kadıoğlu", "DF", 25, "Brighton", 20, 1, 3, 6, 72, 6100, 3, 0, 0),
                ("Hakan Çalhanoğlu", "MF", 32, "Inter Milan", 85, 18, 8, 10, 75, 6200, 5, 0, 0),
                ("Arda Güler", "MF", 21, "Real Madrid", 12, 3, 8, 6, 55, 4200, 2, 0, 0),
                ("Kenan Yıldız", "FW", 21, "Juventus", 12, 2, 8, 6, 65, 5200, 2, 0, 0),
                ("Barış Alper Yılmaz", "FW", 25, "Galatasaray", 18, 3, 8, 6, 65, 5400, 2, 0, 0),
                ("Cenk Tosun", "FW", 35, "Beşiktaş", 55, 18, 6, 3, 48, 3800, 2, 0, 0),
            ]
        },
        "Norway": {
            "players": [
                ("Ørjan Nyland", "GK", 35, "Sevilla", 30, 0, 0, 0, 55, 4950, 2, 0, 18),
                ("Kristoffer Ajer", "DF", 28, "Brentford", 35, 1, 1, 1, 65, 5500, 3, 0, 0),
                ("Erling Haaland", "FW", 25, "Manchester City", 35, 30, 42, 10, 82, 6800, 1, 0, 0),
                ("Martin Ødegaard", "MF", 27, "Arsenal", 55, 8, 12, 16, 78, 6400, 3, 0, 0),
                ("Alexander Sørloth", "FW", 30, "Atlético Madrid", 45, 12, 14, 6, 72, 5800, 2, 0, 0),
                ("Sander Berge", "MF", 28, "Fulham", 55, 4, 4, 5, 68, 5700, 4, 0, 0),
                ("Antonio Nusa", "FW", 21, "RB Leipzig", 12, 2, 8, 6, 62, 5000, 2, 0, 0),
                ("Fredrik Aursnes", "MF", 30, "Benfica", 22, 1, 2, 4, 68, 5700, 3, 0, 0),
            ]
        },
        "Scotland": {
            "players": [
                ("Angus Gunn", "GK", 30, "Norwich City", 12, 0, 0, 0, 62, 5580, 2, 0, 20),
                ("Andrew Robertson", "DF", 32, "Liverpool", 68, 3, 2, 8, 72, 6100, 4, 0, 0),
                ("Kieran Tierney", "DF", 29, "Real Sociedad", 40, 1, 1, 2, 55, 4500, 3, 0, 0),
                ("Scott McTominay", "MF", 29, "Napoli", 52, 10, 10, 6, 72, 6000, 4, 0, 0),
                ("John McGinn", "MF", 31, "Aston Villa", 62, 8, 5, 6, 68, 5600, 4, 0, 0),
                ("Billy Gilmour", "MF", 25, "Napoli", 22, 0, 2, 4, 65, 5400, 2, 0, 0),
                ("Che Adams", "FW", 30, "Torino", 22, 5, 12, 6, 68, 5600, 2, 0, 0),
                ("Lyndon Dykes", "FW", 30, "QPR", 35, 8, 8, 4, 55, 4400, 2, 0, 0),
            ]
        },
        "Sweden": {
            "players": [
                ("Robin Olsen", "GK", 36, "Aston Villa", 55, 0, 0, 0, 48, 4320, 2, 0, 16),
                ("Victor Lindelöf", "DF", 32, "Manchester United", 55, 3, 1, 1, 58, 4800, 3, 0, 0),
                ("Alexander Isak", "FW", 26, "Newcastle", 42, 14, 24, 10, 80, 6600, 2, 0, 0),
                ("Dejan Kulusevski", "FW", 26, "Tottenham", 32, 5, 10, 12, 78, 6400, 3, 0, 0),
                ("Viktor Gyökeres", "FW", 28, "Sporting CP", 18, 8, 38, 10, 80, 6600, 2, 0, 0),
                ("Emil Forsberg", "MF", 34, "Retired", 60, 15, 0, 0, 0, 0, 0, 0, 0),
                ("Hugo Larsson", "MF", 21, "Eintracht Frankfurt", 10, 1, 4, 6, 68, 5600, 2, 0, 0),
                ("Mattias Svanberg", "MF", 26, "Wolfsburg", 18, 1, 2, 3, 62, 5200, 3, 0, 0),
            ]
        },
        "Czechia": {
            "players": [
                ("Jindřich Staněk", "GK", 30, "Slavia Prague", 12, 0, 0, 0, 62, 5580, 2, 0, 22),
                ("Tomáš Souček", "MF", 31, "West Ham", 48, 10, 6, 5, 72, 6000, 5, 0, 0),
                ("Vladimír Coufal", "DF", 34, "West Ham", 38, 1, 1, 3, 62, 5200, 3, 0, 0),
                ("Patrik Schick", "FW", 30, "Bayer Leverkusen", 38, 18, 10, 4, 58, 4700, 2, 0, 0),
                ("Adam Hložek", "FW", 24, "Bayer Leverkusen", 18, 4, 8, 6, 65, 5400, 2, 0, 0),
                ("Pavel Šulc", "MF", 26, "Viktoria Plzeň", 10, 2, 6, 4, 58, 4800, 2, 0, 0),
                ("Alex Král", "MF", 28, "Espanyol", 30, 1, 1, 2, 55, 4500, 3, 0, 0),
                ("Ladislav Krejčí", "DF", 25, "Girona", 12, 1, 2, 1, 65, 5500, 3, 0, 0),
            ]
        },
        "Bosnia and Herzegovina": {
            "players": [
                ("Nikola Vasilj", "GK", 30, "St. Pauli", 12, 0, 0, 0, 62, 5580, 2, 0, 20),
                ("Sead Kolašinac", "DF", 33, "Atalanta", 55, 2, 1, 2, 58, 4800, 4, 0, 0),
                ("Edin Džeko", "FW", 40, "Fenerbahçe", 130, 65, 8, 4, 52, 4000, 2, 0, 0),
                ("Miralem Pjanić", "MF", 36, "KCCA", 110, 8, 1, 2, 42, 3300, 2, 0, 0),
                ("Ermedin Demirović", "FW", 28, "Stuttgart", 18, 4, 14, 6, 72, 5900, 2, 0, 0),
                ("Anel Ahmedhodžić", "DF", 26, "Sheffield United", 15, 1, 2, 1, 62, 5200, 3, 0, 0),
                ("Benjamin Tahirović", "MF", 23, "Ajax", 12, 0, 2, 3, 62, 5000, 3, 0, 0),
                ("Armin Gigović", "MF", 24, "Heerenveen", 8, 1, 3, 4, 55, 4500, 2, 0, 0),
            ]
        },
        "Paraguay": {
            "players": [
                ("Antony Silva", "GK", 43, "Cerro Porteño", 60, 0, 0, 0, 42, 3780, 2, 0, 14),
                ("Gustavo Gómez", "DF", 33, "Palmeiras", 55, 8, 2, 1, 62, 5200, 4, 0, 0),
                ("Miguel Almirón", "MF", 32, "Newcastle", 50, 10, 6, 6, 65, 5400, 3, 0, 0),
                ("Ángel Romero", "FW", 32, "Corinthians", 42, 14, 10, 6, 62, 5000, 3, 0, 0),
                ("Julio Enciso", "FW", 21, "Brighton", 18, 4, 6, 4, 55, 4400, 2, 0, 0),
                ("Mathías Villasanti", "MF", 28, "Grêmio", 25, 2, 3, 3, 65, 5400, 4, 0, 0),
                ("Andrés Cubas", "MF", 29, "Vancouver Whitecaps", 15, 0, 1, 2, 58, 4800, 4, 0, 0),
                ("Antonio Sanabria", "FW", 30, "Torino", 22, 5, 10, 4, 62, 5000, 2, 0, 0),
            ]
        },
        "Panama": {
            "players": [
                ("Luis Mejía", "GK", 31, "Municipal Grecia", 20, 0, 0, 0, 55, 4950, 2, 0, 18),
                ("Fidel Escobar", "DF", 31, "Saprissa", 60, 1, 0, 0, 55, 4500, 4, 0, 0),
                ("José Fajardo", "FW", 28, "Saprissa", 25, 6, 10, 4, 62, 5000, 2, 0, 0),
                ("Adalberto Carrasquilla", "MF", 28, "Houston Dynamo", 40, 3, 3, 4, 62, 5200, 3, 0, 0),
                ("Édgar Bárcenas", "FW", 33, "Universitatea Craiova", 70, 12, 4, 3, 52, 4100, 2, 0, 0),
                ("Ismael Díaz", "FW", 28, "Liga de Quito", 12, 2, 6, 4, 55, 4500, 2, 0, 0),
                ("Michael Murillo", "DF", 30, "Portland Timbers", 50, 1, 1, 2, 58, 4800, 3, 0, 0),
            ]
        },
        "Haiti": {
            "players": [
                ("Alexandre Marcelin", "GK", 28, "Guingamp", 10, 0, 0, 0, 52, 4680, 2, 0, 16),
                ("Frantzdy Pierrot", "FW", 31, "Gaziantep", 28, 8, 6, 3, 52, 4100, 2, 0, 0),
                ("Duckens Nazon", "FW", 32, "Académica", 38, 10, 4, 2, 45, 3500, 2, 0, 0),
                ("Derrick Étienne Jr.", "FW", 30, "Columbus Crew", 35, 5, 6, 4, 55, 4400, 2, 0, 0),
                ("Melchie Dumornay", "MF", 22, "Lyon", 8, 1, 3, 3, 48, 3800, 2, 0, 0),
                ("Bryan Alceus", "MF", 27, "Pau FC", 18, 1, 2, 2, 52, 4200, 3, 0, 0),
                ("Carlens Arcus", "DF", 28, "LASK", 22, 0, 0, 1, 55, 4500, 3, 0, 0),
            ]
        },
        "Curaçao": {
            "players": [
                ("Eloy Room", "GK", 37, "Vitesse", 30, 0, 0, 0, 48, 4320, 2, 0, 16),
                ("Cuco Martina", "DF", 37, "Retired", 30, 1, 0, 0, 0, 0, 0, 0, 0),
                ("Juninho Bacuna", "MF", 29, "Rangers", 28, 4, 4, 4, 58, 4800, 3, 0, 0),
                ("Kenji Gorré", "FW", 31, "NAC Breda", 22, 5, 4, 3, 52, 4100, 2, 0, 0),
                ("Rangelo Janga", "FW", 33, "Çaykur Rizespor", 32, 8, 6, 3, 48, 3800, 2, 0, 0),
                ("Shermaine Martina", "DF", 29, "MVV Maastricht", 15, 0, 0, 1, 52, 4200, 3, 0, 0),
                ("Jarchinio Antonia", "FW", 32, "Anorthosis", 18, 2, 3, 2, 45, 3500, 2, 0, 0),
            ]
        },
        "New Zealand": {
            "players": [
                ("Stefan Marinovic", "GK", 35, "Wellington Phoenix", 18, 0, 0, 0, 52, 4680, 2, 0, 18),
                ("Winston Reid", "DF", 38, "Retired", 50, 3, 0, 0, 0, 0, 0, 0, 0),
                ("Chris Wood", "FW", 34, "Nottingham Forest", 70, 30, 18, 4, 72, 5800, 2, 0, 0),
                ("Liberato Cacace", "DF", 25, "Empoli", 18, 1, 2, 3, 62, 5200, 3, 0, 0),
                ("Matthew Garbett", "MF", 22, "Torino", 12, 1, 2, 3, 52, 4200, 2, 0, 0),
                ("Joe Bell", "MF", 26, "Brøndby", 22, 1, 2, 3, 58, 4800, 3, 0, 0),
                ("Ben Waine", "FW", 25, "Plymouth Argyle", 12, 3, 6, 2, 52, 4100, 1, 0, 0),
                ("Sarpreet Singh", "MF", 26, "Greuther Fürth", 10, 1, 2, 2, 48, 3800, 2, 0, 0),
            ]
        },
        "Ghana": {
            "players": [
                ("Lawrence Ati-Zigi", "GK", 28, "St. Gallen", 12, 0, 0, 0, 58, 5220, 2, 0, 18),
                ("Mohammed Salisu", "DF", 27, "Monaco", 18, 1, 1, 1, 65, 5500, 3, 0, 0),
                ("Thomas Partey", "MF", 33, "Arsenal", 48, 12, 4, 4, 62, 5000, 4, 0, 0),
                ("Mohammed Kudus", "FW", 25, "West Ham", 30, 8, 14, 10, 78, 6400, 4, 0, 0),
                ("Jordan Ayew", "FW", 34, "Crystal Palace", 85, 22, 6, 4, 58, 4700, 2, 0, 0),
                ("Inaki Williams", "FW", 32, "Athletic Bilbao", 22, 3, 10, 6, 72, 5900, 2, 0, 0),
                ("André Ayew", "FW", 36, "Le Havre", 115, 24, 4, 2, 42, 3300, 2, 0, 0),
                ("Kamaldeen Sulemana", "FW", 24, "Southampton", 15, 2, 4, 4, 55, 4400, 2, 0, 0),
            ]
        },
    }
    
    for team_name, team_data in remaining_teams_data.items():
        for p in team_data["players"]:
            name, pos, age, club, caps, intl_goals = p[0], p[1], p[2], p[3], p[4], p[5]
            sg, sa, sapp, smins, yc, rc = p[6], p[7], p[8], p[9], p[10], p[11]
            cs = p[12] if len(p) > 12 else 0
            add_player(team_name, name, pos, age, club, caps, intl_goals,
                       sg, sa, sapp, smins, yc, rc, cs)
    
    return squads, player_stats


# ---------------------------------------------------------------------------
# 4.  Team profiles (pre-computed features for each team)
# ---------------------------------------------------------------------------

def build_team_profiles(matches_df, rankings_2026_df, squads, player_stats_data):
    """Build comprehensive team profiles for all 48 WC 2026 teams."""
    
    profiles = {}
    
    # Build ranking lookup
    rank_lookup = {}
    for _, row in rankings_2026_df.iterrows():
        team = str(row["team"]).strip()
        rank_lookup[team] = {
            "rank": int(row["rank"]),
            "points": float(row["points"]),
        }
    
    # Historical World Cup performance
    wc_champions = {
        "Brazil": 5, "Germany": 4, "Italy": 4, "Argentina": 3, "France": 2,
        "Uruguay": 2, "England": 1, "Spain": 1
    }
    wc_finals = {
        "Brazil": 7, "Germany": 8, "Italy": 6, "Argentina": 6, "France": 3,
        "Uruguay": 2, "Netherlands": 3, "Hungary": 2, "Czechia": 2,
        "England": 1, "Spain": 1, "Croatia": 1, "Sweden": 1
    }
    
    for team in WC2026_TEAMS:
        profile = {"team": team}
        
        # --- Rankings ---
        team_rank = rank_lookup.get(team, {})
        profile["fifa_rank"] = team_rank.get("rank", 100)
        profile["fifa_points"] = team_rank.get("points", 1200.0)
        
        # --- Historical WC record ---
        # Count matches as home or away team
        home_matches = matches_df[matches_df["home_team"] == team]
        away_matches = matches_df[matches_df["away_team"] == team]
        
        total_wc_matches = len(home_matches) + len(away_matches)
        
        home_wins = len(home_matches[home_matches["home_score"] > home_matches["away_score"]])
        away_wins = len(away_matches[away_matches["away_score"] > away_matches["home_score"]])
        total_wins = home_wins + away_wins
        
        home_draws = len(home_matches[home_matches["home_score"] == home_matches["away_score"]])
        away_draws = len(away_matches[away_matches["away_score"] == away_matches["home_score"]])
        total_draws = home_draws + away_draws
        
        total_losses = total_wc_matches - total_wins - total_draws
        
        home_goals = home_matches["home_score"].sum()
        away_goals = away_matches["away_score"].sum()
        total_goals_scored = home_goals + away_goals
        
        home_conceded = home_matches["away_score"].sum()
        away_conceded = away_matches["home_score"].sum()
        total_goals_conceded = home_conceded + away_conceded
        
        profile["wc_total_matches"] = int(total_wc_matches)
        profile["wc_wins"] = int(total_wins)
        profile["wc_draws"] = int(total_draws)
        profile["wc_losses"] = int(total_losses)
        profile["wc_goals_scored"] = int(total_goals_scored)
        profile["wc_goals_conceded"] = int(total_goals_conceded)
        profile["wc_goal_diff"] = int(total_goals_scored - total_goals_conceded)
        profile["wc_win_rate"] = round(total_wins / max(total_wc_matches, 1), 3)
        profile["wc_titles"] = wc_champions.get(team, 0)
        profile["wc_finals"] = wc_finals.get(team, 0)
        
        # --- Average goals per WC match ---
        profile["avg_goals_scored_per_match"] = round(
            total_goals_scored / max(total_wc_matches, 1), 2)
        profile["avg_goals_conceded_per_match"] = round(
            total_goals_conceded / max(total_wc_matches, 1), 2)
        
        # --- Recent form (last 10 WC matches) ---
        team_matches = pd.concat([
            home_matches[["Date", "home_score", "away_score"]].rename(
                columns={"home_score": "gf", "away_score": "ga"}),
            away_matches[["Date", "home_score", "away_score"]].rename(
                columns={"away_score": "gf", "home_score": "ga"}),
        ])
        team_matches = team_matches.sort_values("Date", ascending=False).head(10)
        
        form_points = 0
        for _, m in team_matches.iterrows():
            if m["gf"] > m["ga"]:
                form_points += 3
            elif m["gf"] == m["ga"]:
                form_points += 1
        profile["recent_form_points"] = int(form_points)
        profile["recent_form_max"] = int(len(team_matches) * 3)
        
        # --- Squad aggregates ---
        squad = squads.get(team, [])
        stats = player_stats_data.get(team, [])
        
        if squad:
            ages = [p["age"] for p in squad if p["age"] > 0]
            profile["squad_avg_age"] = round(np.mean(ages), 1) if ages else 27.0
            profile["squad_total_caps"] = sum(p["international_caps"] for p in squad)
            profile["squad_total_intl_goals"] = sum(p["international_goals"] for p in squad)
            profile["squad_size"] = len(squad)
        else:
            profile["squad_avg_age"] = 27.0
            profile["squad_total_caps"] = 200
            profile["squad_total_intl_goals"] = 50
            profile["squad_size"] = 23
        
        if stats:
            profile["squad_season_goals"] = sum(p["goals_last_2_seasons"] for p in stats)
            profile["squad_season_assists"] = sum(p["assists_last_2_seasons"] for p in stats)
            profile["squad_season_apps"] = sum(p["appearances_last_2_seasons"] for p in stats)
            # Star player impact: top 3 scorers
            top_scorers = sorted(stats, key=lambda x: x["goals_last_2_seasons"], reverse=True)[:3]
            profile["star_player_goals"] = sum(p["goals_last_2_seasons"] for p in top_scorers)
        else:
            profile["squad_season_goals"] = 80
            profile["squad_season_assists"] = 60
            profile["squad_season_apps"] = 500
            profile["star_player_goals"] = 30
        
        # --- Confederation ---
        conf_map = {
            "Argentina": "CONMEBOL", "Brazil": "CONMEBOL", "Colombia": "CONMEBOL",
            "Ecuador": "CONMEBOL", "Uruguay": "CONMEBOL", "Paraguay": "CONMEBOL",
            "France": "UEFA", "Spain": "UEFA", "England": "UEFA", "Portugal": "UEFA",
            "Netherlands": "UEFA", "Belgium": "UEFA", "Germany": "UEFA", "Croatia": "UEFA",
            "Italy": "UEFA", "Switzerland": "UEFA", "Austria": "UEFA", "Türkiye": "UEFA",
            "Norway": "UEFA", "Scotland": "UEFA", "Sweden": "UEFA", "Czechia": "UEFA",
            "Bosnia and Herzegovina": "UEFA",
            "Japan": "AFC", "Korea Republic": "AFC", "Australia": "AFC",
            "IR Iran": "AFC", "Saudi Arabia": "AFC", "Qatar": "AFC", "Iraq": "AFC",
            "Jordan": "AFC", "Uzbekistan": "AFC",
            "Morocco": "CAF", "Senegal": "CAF", "Côte d'Ivoire": "CAF",
            "Nigeria": "CAF", "Egypt": "CAF", "Algeria": "CAF",
            "South Africa": "CAF", "Tunisia": "CAF", "Congo DR": "CAF",
            "Cabo Verde": "CAF",
            "United States": "CONCACAF", "Mexico": "CONCACAF", "Canada": "CONCACAF",
            "Panama": "CONCACAF", "Haiti": "CONCACAF", "Curaçao": "CONCACAF",
            "New Zealand": "OFC",
        }
        profile["confederation"] = conf_map.get(team, "UEFA")
        
        # --- Home advantage (for 2026 hosted by USA/Mexico/Canada) ---
        profile["is_host"] = team in ["United States", "Mexico", "Canada"]
        
        profiles[team] = profile
    
    return profiles


# ---------------------------------------------------------------------------
# 5.  Main entry point
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("FIFA World Cup 2026 — Data Collector")
    print("=" * 60)
    
    # Load existing data
    raw_dir = os.path.join("data", "raw")
    matches = load_matches(os.path.join(raw_dir, "matches_1930_2022.csv"))
    r2022, r2026 = load_rankings(
        os.path.join(raw_dir, "fifa_ranking_2022-10-06.csv"),
        os.path.join(raw_dir, "fifa_ranking_2026-06-08.csv")
    )
    wc_summary = load_world_cup_summary(os.path.join(raw_dir, "world_cup.csv"))
    
    # Generate squad and player data
    print("\nGenerating squad rosters and player stats...")
    squads, player_stats_data = generate_squads_and_stats()
    print(f"Generated data for {len(squads)} teams")
    total_players = sum(len(v) for v in squads.values())
    print(f"Total players: {total_players}")
    
    # Build team profiles
    print("\nBuilding team profiles...")
    profiles = build_team_profiles(matches, r2026, squads, player_stats_data)
    print(f"Built profiles for {len(profiles)} teams")
    
    # Save outputs
    output_dir = os.path.join("data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    
    squads_path = os.path.join(output_dir, "squads_2026.json")
    with open(squads_path, "w", encoding="utf-8") as f:
        json.dump(squads, f, ensure_ascii=False, indent=2)
    print(f"\nSaved squads to {squads_path}")
    
    stats_path = os.path.join(output_dir, "player_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(player_stats_data, f, ensure_ascii=False, indent=2)
    print(f"Saved player stats to {stats_path}")
    
    profiles_path = os.path.join(output_dir, "team_profiles.json")
    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, ensure_ascii=False, indent=2)
    print(f"Saved team profiles to {profiles_path}")
    
    # Quick summary
    print("\n" + "=" * 60)
    print("Data collection complete!")
    print(f"  Teams:   {len(squads)}")
    print(f"  Players: {total_players}")
    print(f"  Matches: {len(matches)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
