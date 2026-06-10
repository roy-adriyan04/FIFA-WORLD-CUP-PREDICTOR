"""
ML Model for FIFA World Cup 2026 Match Predictor
=================================================
XGBoost classifier for W/D/L, Poisson-based score prediction,
MOTM heuristic, and match stats generator.
"""

import numpy as np
import pandas as pd
import json
import os
import joblib
from xgboost import XGBClassifier
from sklearn.linear_model import PoissonRegressor
from feature_engineering import (
    build_prediction_features, build_score_features,
    STAGE_MAP, _default_profile
)


# ---------------------------------------------------------------------------
# 1.  Match Outcome Predictor (XGBoost)
# ---------------------------------------------------------------------------

class MatchPredictor:
    """Predicts match outcome (win/draw/loss) and probabilities."""
    
    def __init__(self, model_path=None):
        self.model = None
        self.feature_names = None
        if model_path and os.path.exists(model_path):
            self.load(model_path)
    
    def train(self, X, y, feature_names=None):
        """Train the XGBoost classifier."""
        self.feature_names = feature_names or list(X.columns)
        
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        )
        
        self.model.fit(X, y)
        return self
    
    def predict_proba(self, X):
        """Return probabilities for [home_win, draw, away_win]."""
        if self.model is None:
            raise ValueError("Model not trained/loaded")
        return self.model.predict_proba(X)
    
    def predict(self, X):
        """Return predicted class (0=home, 1=draw, 2=away)."""
        if self.model is None:
            raise ValueError("Model not trained/loaded")
        return self.model.predict(X)
    
    def save(self, path):
        """Save model and metadata."""
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
        }, path)
    
    def load(self, path):
        """Load model and metadata."""
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data["feature_names"]
    
    def get_feature_importance(self):
        """Return feature importance dict."""
        if self.model is None:
            return {}
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances.tolist()))


# ---------------------------------------------------------------------------
# 2.  Score Predictor (Poisson-based heuristic)
# ---------------------------------------------------------------------------

class ScorePredictor:
    """Predicts match score using expected goals approach."""
    
    # Historical World Cup average goals per team per match
    WC_AVG_GOALS = 1.35
    
    def predict_score(self, team1_profile, team2_profile, stage_name,
                      win_probs=None):
        """
        Predict score for team1 vs team2.
        win_probs: [team1_win, draw, team2_win] from MatchPredictor
        Returns (team1_goals, team2_goals, team1_xg, team2_xg)
        """
        stage = STAGE_MAP.get(stage_name, 0)
        
        # Calculate expected goals for each team
        t1_xg = self._calc_expected_goals(team1_profile, team2_profile, stage)
        t2_xg = self._calc_expected_goals(team2_profile, team1_profile, stage)
        
        # Adjust based on win probabilities if available
        if win_probs is not None:
            p1_win, p_draw, p2_win = win_probs
            
            # Nudge xG based on predicted outcome
            if p1_win > p2_win:
                t1_xg *= (1 + 0.15 * (p1_win - p2_win))
                t2_xg *= (1 - 0.1 * (p1_win - p2_win))
            else:
                t2_xg *= (1 + 0.15 * (p2_win - p1_win))
                t1_xg *= (1 - 0.1 * (p2_win - p1_win))
        
        # Clamp xG to reasonable range
        t1_xg = max(0.3, min(4.0, t1_xg))
        t2_xg = max(0.2, min(3.5, t2_xg))
        
        # Generate actual score from xG using Poisson sampling
        # But make it deterministic based on xG for consistency
        t1_goals = self._xg_to_goals(t1_xg)
        t2_goals = self._xg_to_goals(t2_xg)
        
        # Ensure score aligns with predicted outcome
        if win_probs is not None:
            p1_win, p_draw, p2_win = win_probs
            predicted_outcome = np.argmax(win_probs)
            
            if predicted_outcome == 0 and t1_goals <= t2_goals:
                t1_goals = t2_goals + 1
            elif predicted_outcome == 2 and t2_goals <= t1_goals:
                t2_goals = t1_goals + 1
            elif predicted_outcome == 1 and t1_goals != t2_goals:
                avg = round((t1_goals + t2_goals) / 2)
                t1_goals = max(1, avg)
                t2_goals = t1_goals
        
        return int(t1_goals), int(t2_goals), round(t1_xg, 2), round(t2_xg, 2)
    
    def _calc_expected_goals(self, team_profile, opp_profile, stage):
        """Calculate expected goals for a team against an opponent."""
        tp = team_profile
        op = opp_profile
        
        # Base: team's historical WC scoring rate
        base_rate = tp.get("avg_goals_scored_per_match", self.WC_AVG_GOALS)
        if base_rate == 0:
            base_rate = self.WC_AVG_GOALS
        
        # Adjust for opponent's defensive record
        opp_concede_rate = op.get("avg_goals_conceded_per_match", self.WC_AVG_GOALS)
        if opp_concede_rate == 0:
            opp_concede_rate = self.WC_AVG_GOALS
        
        # Weight: team attack vs opponent defense
        xg = base_rate * 0.5 + opp_concede_rate * 0.3 + self.WC_AVG_GOALS * 0.2
        
        # Ranking adjustment
        rank_diff = op["fifa_rank"] - tp["fifa_rank"]
        xg += rank_diff * 0.005
        
        # Points adjustment
        pts_diff = tp["fifa_points"] - op["fifa_points"]
        xg += pts_diff * 0.0003
        
        # Form adjustment
        form_pct = tp["recent_form_points"] / max(tp["recent_form_max"], 1)
        xg *= (0.8 + 0.4 * form_pct)
        
        # Stage adjustment (fewer goals in later stages)
        stage_multiplier = 1.0 - (stage * 0.04)
        xg *= stage_multiplier
        
        # Squad quality
        star_goals = tp.get("star_player_goals", 20)
        xg *= (0.9 + 0.1 * min(star_goals / 30.0, 1.0))
        
        # Host advantage
        if tp.get("is_host", False):
            xg *= 1.1
        
        return max(0.3, xg)
    
    def _xg_to_goals(self, xg):
        """Convert xG to a discrete goal count."""
        # Use a deterministic mapping based on xG
        if xg < 0.5:
            return 0
        elif xg < 1.0:
            return 1 if xg > 0.7 else 0
        elif xg < 1.8:
            return 1
        elif xg < 2.5:
            return 2
        elif xg < 3.3:
            return 3
        elif xg < 4.0:
            return 4
        else:
            return 5


# ---------------------------------------------------------------------------
# 3.  Man of the Match Predictor
# ---------------------------------------------------------------------------

class MOTMPredictor:
    """Predicts Man of the Match based on player stats and match outcome."""
    
    def predict_motm(self, team1_name, team2_name, team1_goals, team2_goals,
                     player_stats, win_probs):
        """
        Predict Man of the Match.
        Returns dict with player name, team, position, and reasoning.
        """
        # Winning team's players are more likely to be MOTM
        if team1_goals > team2_goals:
            primary_team = team1_name
            secondary_team = team2_name
            margin = team1_goals - team2_goals
        elif team2_goals > team1_goals:
            primary_team = team2_name
            secondary_team = team1_name
            margin = team2_goals - team1_goals
        else:
            # Draw — pick from the team with higher win probability
            p1, _, p2 = win_probs
            primary_team = team1_name if p1 >= p2 else team2_name
            secondary_team = team2_name if p1 >= p2 else team1_name
            margin = 0
        
        # Score each player
        primary_players = player_stats.get(primary_team, [])
        secondary_players = player_stats.get(secondary_team, [])
        
        candidates = []
        
        for p in primary_players:
            score = self._score_player(p, is_winning_team=True, goal_margin=margin)
            candidates.append((p, primary_team, score))
        
        for p in secondary_players:
            score = self._score_player(p, is_winning_team=False, goal_margin=margin)
            candidates.append((p, secondary_team, score))
        
        if not candidates:
            return {
                "name": "Unknown",
                "team": primary_team,
                "position": "FW",
                "reason": "Key contribution to team performance"
            }
        
        # Sort by score, pick top
        candidates.sort(key=lambda x: x[2], reverse=True)
        best = candidates[0]
        player, team, score = best
        
        # Generate reason
        reason = self._generate_reason(player, team == primary_team, margin)
        
        return {
            "name": player["name"],
            "team": team,
            "position": player["position"],
            "club": player.get("club", ""),
            "reason": reason,
            "rating": round(min(10.0, 6.5 + score * 0.5), 1),
        }
    
    def _score_player(self, player, is_winning_team, goal_margin):
        """Score a player for MOTM candidacy."""
        score = 0
        
        # Position weight (attackers favored for MOTM)
        pos_weight = {"FW": 1.3, "MF": 1.1, "DF": 0.8, "GK": 0.6}
        score *= pos_weight.get(player["position"], 1.0)
        
        # Goals scored (club + intl, last 2 seasons)
        goals = player.get("goals_last_2_seasons", 0)
        score += goals * 0.15
        
        # Assists
        assists = player.get("assists_last_2_seasons", 0)
        score += assists * 0.1
        
        # International experience
        caps = player.get("international_caps", 0)
        score += min(caps / 50.0, 2.0)
        
        # International goals
        intl_goals = player.get("international_goals", 0)
        score += intl_goals * 0.08
        
        # Appearances (fitness/form)
        apps = player.get("appearances_last_2_seasons", 0)
        score += min(apps / 40.0, 1.5)
        
        # Winning team bonus
        if is_winning_team:
            score *= 1.4
        
        # Big margin bonus for attackers
        if is_winning_team and goal_margin >= 2 and player["position"] == "FW":
            score *= 1.3
        
        # GK bonus if winning with clean sheet or narrow win
        if is_winning_team and player["position"] == "GK":
            cs = player.get("clean_sheets_last_2_seasons", 0)
            score += cs * 0.1
            if goal_margin <= 1:
                score *= 1.5
        
        return score
    
    def _generate_reason(self, player, is_winning_team, margin):
        """Generate a human-readable MOTM reason."""
        pos = player["position"]
        name = player["name"]
        
        if pos == "FW":
            if margin >= 2:
                return f"Devastating attacking display with multiple goal contributions"
            elif margin == 1:
                return f"Decisive forward play, clinical finishing in a tight contest"
            else:
                return f"Constant attacking threat, created multiple chances"
        elif pos == "MF":
            if is_winning_team:
                return f"Controlled the midfield, dictating tempo and creating opportunities"
            else:
                return f"Outstanding midfield performance despite the result"
        elif pos == "DF":
            if is_winning_team and margin <= 1:
                return f"Rock-solid defensive display, crucial tackles and interceptions"
            else:
                return f"Commanding defensive performance, led from the back"
        elif pos == "GK":
            return f"Exceptional shot-stopping, key saves at crucial moments"
        
        return f"Outstanding all-round performance"


# ---------------------------------------------------------------------------
# 4.  Match Stats Generator
# ---------------------------------------------------------------------------

class MatchStatsGenerator:
    """Generates realistic match statistics based on team profiles."""
    
    def generate_stats(self, team1_profile, team2_profile, team1_goals,
                       team2_goals, stage_name, team1_xg, team2_xg):
        """Generate comprehensive match statistics."""
        stage = STAGE_MAP.get(stage_name, 0)
        
        # Possession based on ranking + style
        t1_rank = team1_profile["fifa_rank"]
        t2_rank = team2_profile["fifa_rank"]
        t1_pts = team1_profile["fifa_points"]
        t2_pts = team2_profile["fifa_points"]
        
        # Higher-ranked teams tend to have more possession
        pts_total = t1_pts + t2_pts
        t1_poss_base = (t1_pts / pts_total) * 100
        # Add some noise for realism
        np.random.seed(hash(f"{team1_profile['team']}{team2_profile['team']}{stage}") % 2**31)
        noise = np.random.normal(0, 3)
        t1_poss = max(35, min(65, t1_poss_base + noise))
        t2_poss = 100 - t1_poss
        
        # Shots (correlated with xG, ~6-8 shots per xG)
        shots_per_xg = np.random.uniform(5.5, 8.0)
        t1_shots = max(3, round(team1_xg * shots_per_xg + np.random.normal(0, 2)))
        t2_shots = max(2, round(team2_xg * shots_per_xg + np.random.normal(0, 2)))
        
        # Shots on target (~35-45% of total shots)
        sot_pct = np.random.uniform(0.33, 0.48)
        t1_sot = max(1, min(t1_shots, round(t1_shots * sot_pct)))
        t2_sot = max(1, min(t2_shots, round(t2_shots * sot_pct)))
        
        # Ensure shots on target >= goals
        t1_sot = max(t1_sot, team1_goals)
        t2_sot = max(t2_sot, team2_goals)
        t1_shots = max(t1_shots, t1_sot)
        t2_shots = max(t2_shots, t2_sot)
        
        # Corners (correlated with possession and attack)
        t1_corners = max(1, round(t1_poss / 10 + np.random.normal(0, 1.5)))
        t2_corners = max(1, round(t2_poss / 10 + np.random.normal(0, 1.5)))
        
        # Pass accuracy (higher-ranked teams generally better)
        t1_pass_acc = max(70, min(93, 78 + (100 - t1_rank) * 0.12 + np.random.normal(0, 3)))
        t2_pass_acc = max(70, min(93, 78 + (100 - t2_rank) * 0.12 + np.random.normal(0, 3)))
        
        # Cards (more in knockout stages)
        card_base = 1.5 + stage * 0.3
        t1_yellows = max(0, round(card_base + np.random.normal(0, 0.8)))
        t2_yellows = max(0, round(card_base + np.random.normal(0, 0.8)))
        t1_reds = 1 if np.random.random() < 0.05 else 0
        t2_reds = 1 if np.random.random() < 0.05 else 0
        
        # Fouls
        t1_fouls = max(5, round(12 + np.random.normal(0, 3)))
        t2_fouls = max(5, round(12 + np.random.normal(0, 3)))
        
        # Offsides
        t1_offsides = max(0, round(2 + np.random.normal(0, 1.2)))
        t2_offsides = max(0, round(2 + np.random.normal(0, 1.2)))
        
        return {
            "team1": {
                "possession": round(t1_poss, 1),
                "shots": int(t1_shots),
                "shots_on_target": int(t1_sot),
                "xg": float(team1_xg),
                "corners": int(t1_corners),
                "pass_accuracy": round(t1_pass_acc, 1),
                "yellow_cards": int(t1_yellows),
                "red_cards": int(t1_reds),
                "fouls": int(t1_fouls),
                "offsides": int(t1_offsides),
            },
            "team2": {
                "possession": round(t2_poss, 1),
                "shots": int(t2_shots),
                "shots_on_target": int(t2_sot),
                "xg": float(team2_xg),
                "corners": int(t2_corners),
                "pass_accuracy": round(t2_pass_acc, 1),
                "yellow_cards": int(t2_yellows),
                "red_cards": int(t2_reds),
                "fouls": int(t2_fouls),
                "offsides": int(t2_offsides),
            }
        }


# ---------------------------------------------------------------------------
# 5.  Full Prediction Pipeline
# ---------------------------------------------------------------------------

class WorldCupPredictor:
    """
    Complete prediction pipeline: combines all sub-models to generate
    a full match prediction.
    """
    
    def __init__(self, model_path="model.pkl",
                 profiles_path="team_profiles.json",
                 player_stats_path="player_stats.json"):
        
        self.match_predictor = MatchPredictor(model_path)
        self.score_predictor = ScorePredictor()
        self.motm_predictor = MOTMPredictor()
        self.stats_generator = MatchStatsGenerator()
        
        # Load team profiles
        with open(profiles_path, "r", encoding="utf-8") as f:
            self.team_profiles = json.load(f)
        
        # Load player stats
        with open(player_stats_path, "r", encoding="utf-8") as f:
            self.player_stats = json.load(f)
    
    def predict(self, team1, team2, stage):
        """
        Generate full match prediction.
        Returns a comprehensive result dict.
        """
        # 1. Build features
        features = build_prediction_features(
            team1, team2, stage, self.team_profiles)
        
        # 2. Predict outcome probabilities
        probs = self.match_predictor.predict_proba(features)[0]
        # probs = [home_win, draw, away_win]
        
        predicted_outcome = int(np.argmax(probs))
        outcome_labels = {0: team1, 1: "Draw", 2: team2}
        
        # 3. Predict score
        t1_profile = self.team_profiles.get(team1, _default_profile(team1))
        t2_profile = self.team_profiles.get(team2, _default_profile(team2))
        
        t1_goals, t2_goals, t1_xg, t2_xg = self.score_predictor.predict_score(
            t1_profile, t2_profile, stage, win_probs=probs.tolist())
        
        # 4. Predict MOTM
        motm = self.motm_predictor.predict_motm(
            team1, team2, t1_goals, t2_goals,
            self.player_stats, probs.tolist())
        
        # 5. Generate match stats
        stats = self.stats_generator.generate_stats(
            t1_profile, t2_profile, t1_goals, t2_goals,
            stage, t1_xg, t2_xg)
        
        # 6. Determine winner and confidence
        if predicted_outcome == 0:
            winner = team1
            confidence = float(probs[0])
        elif predicted_outcome == 2:
            winner = team2
            confidence = float(probs[2])
        else:
            winner = "Draw"
            confidence = float(probs[1])
        
        return {
            "team1": team1,
            "team2": team2,
            "stage": stage,
            "winner": winner,
            "confidence": round(confidence * 100, 1),
            "probabilities": {
                team1: round(float(probs[0]) * 100, 1),
                "Draw": round(float(probs[1]) * 100, 1),
                team2: round(float(probs[2]) * 100, 1),
            },
            "score": {
                team1: t1_goals,
                team2: t2_goals,
            },
            "xg": {
                team1: t1_xg,
                team2: t2_xg,
            },
            "motm": motm,
            "stats": stats,
            "team1_rank": t1_profile.get("fifa_rank", "N/A"),
            "team2_rank": t2_profile.get("fifa_rank", "N/A"),
        }
    
    def get_teams(self):
        """Return list of available teams."""
        return sorted(self.team_profiles.keys())
    
    def get_team_info(self, team_name):
        """Return team profile and squad info."""
        profile = self.team_profiles.get(team_name, {})
        squad = self.player_stats.get(team_name, [])
        return {
            "profile": profile,
            "squad": squad,
        }
