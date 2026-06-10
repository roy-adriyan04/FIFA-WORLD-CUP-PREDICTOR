"""
Feature Engineering for FIFA World Cup 2026 Match Predictor
===========================================================
Transforms raw match/team data into ML-ready feature vectors.
"""

import pandas as pd
import numpy as np
import json
import os

# ---------------------------------------------------------------------------
# 1.  Stage encoding
# ---------------------------------------------------------------------------
STAGE_MAP = {
    "Group stage": 0,
    "Round of 32": 1,
    "Round of 16": 2,
    "Quarter-finals": 3,
    "Semi-finals": 4,
    "Third-place match": 4,
    "Final": 5,
}

CONFEDERATION_MAP = {
    "CONMEBOL": 0, "UEFA": 1, "AFC": 2, "CAF": 3,
    "CONCACAF": 4, "OFC": 5,
}

# ---------------------------------------------------------------------------
# 2.  Build features from historical matches
# ---------------------------------------------------------------------------

def build_historical_features(matches_df, rankings_2026, team_profiles):
    """
    Build feature matrix from historical WC matches for model training.
    Each row = one historical match with features + outcome label.
    """
    
    features_list = []
    labels = []
    
    # Pre-compute cumulative team stats up to each match date
    # For simplicity, we use overall stats (not time-windowed) for training
    # The model learns relative strength patterns regardless
    
    for idx, row in matches_df.iterrows():
        home = str(row["home_team"]).strip()
        away = str(row["away_team"]).strip()
        
        home_score = row["home_score"]
        away_score = row["away_score"]
        
        if pd.isna(home_score) or pd.isna(away_score):
            continue
        
        home_score = int(home_score)
        away_score = int(away_score)
        
        # Outcome label: 0=home win, 1=draw, 2=away win
        if home_score > away_score:
            label = 0
        elif home_score == away_score:
            label = 1
        else:
            label = 2
        
        # Get team profiles (fall back to defaults)
        hp = team_profiles.get(home, _default_profile(home))
        ap = team_profiles.get(away, _default_profile(away))
        
        # Stage
        stage_raw = str(row.get("Round", "Group stage")).strip()
        stage = STAGE_MAP.get(stage_raw, 0)
        
        # Year
        year = int(row.get("Year", 2000))
        
        # Build feature vector
        feat = _build_match_features(hp, ap, stage, year)
        features_list.append(feat)
        labels.append(label)
    
    feature_names = _get_feature_names()
    X = pd.DataFrame(features_list, columns=feature_names)
    y = np.array(labels)
    
    return X, y, feature_names


def _default_profile(team_name):
    """Default profile for teams not in our 48-team dataset."""
    return {
        "team": team_name,
        "fifa_rank": 80,
        "fifa_points": 1300.0,
        "wc_total_matches": 5,
        "wc_wins": 1,
        "wc_draws": 1,
        "wc_losses": 3,
        "wc_goals_scored": 4,
        "wc_goals_conceded": 8,
        "wc_goal_diff": -4,
        "wc_win_rate": 0.2,
        "wc_titles": 0,
        "wc_finals": 0,
        "avg_goals_scored_per_match": 0.8,
        "avg_goals_conceded_per_match": 1.6,
        "recent_form_points": 5,
        "recent_form_max": 30,
        "squad_avg_age": 27.0,
        "squad_total_caps": 200,
        "squad_total_intl_goals": 30,
        "squad_size": 23,
        "squad_season_goals": 60,
        "squad_season_assists": 40,
        "squad_season_apps": 400,
        "star_player_goals": 20,
        "confederation": "UEFA",
        "is_host": False,
    }


def _build_match_features(home_profile, away_profile, stage, year):
    """
    Build a single feature vector for a match between two teams.
    Returns a list of numeric features.
    """
    hp = home_profile
    ap = away_profile
    
    # Ranking differential (negative = home is better)
    rank_diff = hp["fifa_rank"] - ap["fifa_rank"]
    points_diff = hp["fifa_points"] - ap["fifa_points"]
    
    # Historical WC record differentials
    wc_matches_diff = hp["wc_total_matches"] - ap["wc_total_matches"]
    wc_wins_diff = hp["wc_wins"] - ap["wc_wins"]
    wc_win_rate_diff = hp["wc_win_rate"] - ap["wc_win_rate"]
    wc_goals_diff = hp["wc_goal_diff"] - ap["wc_goal_diff"]
    wc_titles_diff = hp["wc_titles"] - ap["wc_titles"]
    wc_finals_diff = hp["wc_finals"] - ap["wc_finals"]
    
    # Attacking/defensive differentials
    avg_scored_diff = hp["avg_goals_scored_per_match"] - ap["avg_goals_scored_per_match"]
    avg_conceded_diff = hp["avg_goals_conceded_per_match"] - ap["avg_goals_conceded_per_match"]
    
    # Form differential
    form_diff = hp["recent_form_points"] - ap["recent_form_points"]
    
    # Squad quality differentials
    age_diff = hp["squad_avg_age"] - ap["squad_avg_age"]
    caps_diff = hp["squad_total_caps"] - ap["squad_total_caps"]
    intl_goals_diff = hp["squad_total_intl_goals"] - ap["squad_total_intl_goals"]
    season_goals_diff = hp["squad_season_goals"] - ap["squad_season_goals"]
    star_goals_diff = hp["star_player_goals"] - ap["star_player_goals"]
    
    # Absolute values (team strengths)
    home_rank = hp["fifa_rank"]
    away_rank = ap["fifa_rank"]
    home_points = hp["fifa_points"]
    away_points = ap["fifa_points"]
    home_wc_exp = hp["wc_total_matches"]
    away_wc_exp = ap["wc_total_matches"]
    
    # Confederation encoding
    home_conf = CONFEDERATION_MAP.get(hp.get("confederation", "UEFA"), 1)
    away_conf = CONFEDERATION_MAP.get(ap.get("confederation", "UEFA"), 1)
    same_conf = int(home_conf == away_conf)
    
    # Host advantage
    home_is_host = int(hp.get("is_host", False))
    away_is_host = int(ap.get("is_host", False))
    
    # Match context
    stage_encoded = stage
    year_normalized = (year - 1930) / (2026 - 1930)  # Normalize to 0-1
    
    return [
        rank_diff, points_diff,
        wc_matches_diff, wc_wins_diff, wc_win_rate_diff, wc_goals_diff,
        wc_titles_diff, wc_finals_diff,
        avg_scored_diff, avg_conceded_diff,
        form_diff,
        age_diff, caps_diff, intl_goals_diff,
        season_goals_diff, star_goals_diff,
        home_rank, away_rank, home_points, away_points,
        home_wc_exp, away_wc_exp,
        home_conf, away_conf, same_conf,
        home_is_host, away_is_host,
        stage_encoded, year_normalized,
    ]


def _get_feature_names():
    return [
        "rank_diff", "points_diff",
        "wc_matches_diff", "wc_wins_diff", "wc_win_rate_diff", "wc_goals_diff",
        "wc_titles_diff", "wc_finals_diff",
        "avg_scored_diff", "avg_conceded_diff",
        "form_diff",
        "age_diff", "caps_diff", "intl_goals_diff",
        "season_goals_diff", "star_goals_diff",
        "home_rank", "away_rank", "home_points", "away_points",
        "home_wc_exp", "away_wc_exp",
        "home_conf", "away_conf", "same_conf",
        "home_is_host", "away_is_host",
        "stage_encoded", "year_normalized",
    ]


# ---------------------------------------------------------------------------
# 3.  Build prediction features for a new match
# ---------------------------------------------------------------------------

def build_prediction_features(team1_name, team2_name, stage_name, team_profiles):
    """
    Build feature vector for predicting a new match.
    team1 = home/team A, team2 = away/team B
    """
    hp = team_profiles.get(team1_name, _default_profile(team1_name))
    ap = team_profiles.get(team2_name, _default_profile(team2_name))
    stage = STAGE_MAP.get(stage_name, 0)
    year = 2026
    
    feat = _build_match_features(hp, ap, stage, year)
    feature_names = _get_feature_names()
    
    return pd.DataFrame([feat], columns=feature_names)


# ---------------------------------------------------------------------------
# 4.  Score prediction features
# ---------------------------------------------------------------------------

def build_score_features(team_profile, opponent_profile, stage, is_home=True):
    """
    Build features for Poisson score prediction (predicting goals for one team).
    """
    tp = team_profile
    op = opponent_profile
    
    # Team's attacking strength
    attack_strength = tp["avg_goals_scored_per_match"]
    # Opponent's defensive weakness  
    defense_weakness = op["avg_goals_conceded_per_match"]
    
    # Ranking-based adjustment
    rank_advantage = (op["fifa_rank"] - tp["fifa_rank"]) / 100.0
    points_advantage = (tp["fifa_points"] - op["fifa_points"]) / 500.0
    
    # Historical form
    form_score = tp["recent_form_points"] / max(tp["recent_form_max"], 1)
    
    # Squad quality
    star_factor = tp["star_player_goals"] / 30.0  # Normalize
    season_goals_factor = tp["squad_season_goals"] / 200.0
    
    # Stage factor (fewer goals in later rounds typically)
    stage_factor = 1.0 - (stage * 0.05)
    
    # WC experience
    wc_exp = min(tp["wc_total_matches"] / 50.0, 1.0)
    
    return {
        "attack_strength": attack_strength,
        "defense_weakness": defense_weakness,
        "rank_advantage": rank_advantage,
        "points_advantage": points_advantage,
        "form_score": form_score,
        "star_factor": star_factor,
        "season_goals_factor": season_goals_factor,
        "stage_factor": stage_factor,
        "wc_experience": wc_exp,
        "is_host": int(tp.get("is_host", False)),
    }


if __name__ == "__main__":
    # Quick test
    import os
    profiles_path = os.path.join("data", "processed", "team_profiles.json")
    with open(profiles_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)
    
    feat = build_prediction_features("Argentina", "France", "Final", profiles)
    print("Sample features for Argentina vs France (Final):")
    print(feat.T)
