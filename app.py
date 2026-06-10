"""
Flask API Backend for FIFA World Cup 2026 Match Predictor
=========================================================
Serves the prediction API and static frontend files.
"""

from flask import Flask, jsonify, request, send_from_directory
import os
import json
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model import WorldCupPredictor

app = Flask(__name__, static_folder="static", static_url_path="/static")

# ---------------------------------------------------------------------------
# Initialize predictor
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
PROFILES_PATH = os.path.join(BASE_DIR, "data", "processed", "team_profiles.json")
PLAYER_STATS_PATH = os.path.join(BASE_DIR, "data", "processed", "player_stats.json")
SQUADS_PATH = os.path.join(BASE_DIR, "data", "processed", "squads_2026.json")

predictor = None


def get_predictor():
    global predictor
    if predictor is None:
        predictor = WorldCupPredictor(MODEL_PATH, PROFILES_PATH, PLAYER_STATS_PATH)
    return predictor


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the main page."""
    return send_from_directory("static", "index.html")


@app.route("/api/teams", methods=["GET"])
def get_teams():
    """Return list of all available teams with their rankings."""
    wp = get_predictor()
    teams = wp.get_teams()
    
    # Include basic info for each team
    teams_data = []
    for team in teams:
        profile = wp.team_profiles.get(team, {})
        teams_data.append({
            "name": team,
            "rank": profile.get("fifa_rank", 999),
            "confederation": profile.get("confederation", ""),
            "points": profile.get("fifa_points", 0),
        })
    
    # Sort by rank
    teams_data.sort(key=lambda x: x["rank"])
    
    return jsonify({"teams": teams_data})


@app.route("/api/predict", methods=["POST"])
def predict_match():
    """Predict a match outcome."""
    data = request.get_json()
    
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400
    
    team1 = data.get("team1", "").strip()
    team2 = data.get("team2", "").strip()
    stage = data.get("stage", "Group stage").strip()
    
    if not team1 or not team2:
        return jsonify({"error": "Both team1 and team2 are required"}), 400
    
    if team1 == team2:
        return jsonify({"error": "Teams must be different"}), 400
    
    wp = get_predictor()
    
    # Validate teams
    available = wp.get_teams()
    if team1 not in available:
        return jsonify({"error": f"Team '{team1}' not found"}), 404
    if team2 not in available:
        return jsonify({"error": f"Team '{team2}' not found"}), 404
    
    # Valid stages
    valid_stages = [
        "Group stage", "Round of 32", "Round of 16",
        "Quarter-finals", "Semi-finals", "Final"
    ]
    if stage not in valid_stages:
        return jsonify({"error": f"Invalid stage. Must be one of: {valid_stages}"}), 400
    
    # Run prediction
    try:
        result = wp.predict(team1, team2, stage)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/team/<name>", methods=["GET"])
def get_team_info(name):
    """Return team profile and squad info."""
    wp = get_predictor()
    
    if name not in wp.team_profiles:
        return jsonify({"error": f"Team '{name}' not found"}), 404
    
    info = wp.get_team_info(name)
    
    # Also include squad roster
    with open(SQUADS_PATH, "r", encoding="utf-8") as f:
        squads = json.load(f)
    
    info["roster"] = squads.get(name, [])
    
    return jsonify(info)


@app.route("/api/stages", methods=["GET"])
def get_stages():
    """Return available tournament stages."""
    return jsonify({
        "stages": [
            "Group stage",
            "Round of 32",
            "Round of 16",
            "Quarter-finals",
            "Semi-finals",
            "Final",
        ]
    })


# ---------------------------------------------------------------------------
# Run server
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 50)
    print("FIFA World Cup 2026 Match Predictor — API Server")
    print("=" * 50)
    print(f"Loading model from {MODEL_PATH}...")
    get_predictor()
    print("Model loaded successfully!")
    print("\nStarting server on http://localhost:5000")
    print("Press Ctrl+C to stop.\n")
    
    app.run(host="0.0.0.0", port=5000, debug=False)
