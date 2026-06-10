"""
Train the FIFA World Cup 2026 Match Predictor
==============================================
Trains XGBoost model on historical WC data, evaluates with 75/25 split.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, log_loss
)

from data_collector import load_matches, load_rankings
from feature_engineering import build_historical_features
from model import MatchPredictor


def main():
    print("=" * 60)
    print("Training FIFA World Cup 2026 Match Predictor")
    print("=" * 60)
    
    # -----------------------------------------------------------------------
    # 1. Load data
    # -----------------------------------------------------------------------
    print("\n[1/6] Loading data...")
    matches = load_matches(os.path.join("data", "raw", "matches_1930_2022.csv"))
    _, r2026 = load_rankings(
        os.path.join("data", "raw", "fifa_ranking_2022-10-06.csv"),
        os.path.join("data", "raw", "fifa_ranking_2026-06-08.csv")
    )
    
    profiles_path = os.path.join("data", "processed", "team_profiles.json")
    with open(profiles_path, "r", encoding="utf-8") as f:
        team_profiles = json.load(f)
    
    # -----------------------------------------------------------------------
    # 2. Build features
    # -----------------------------------------------------------------------
    print("\n[2/6] Building feature matrix...")
    X, y, feature_names = build_historical_features(matches, r2026, team_profiles)
    
    print(f"  Feature matrix shape: {X.shape}")
    print(f"  Label distribution:")
    unique, counts = np.unique(y, return_counts=True)
    labels_map = {0: "Home Win", 1: "Draw", 2: "Away Win"}
    for u, c in zip(unique, counts):
        print(f"    {labels_map[u]}: {c} ({c/len(y)*100:.1f}%)")
    
    # -----------------------------------------------------------------------
    # 3. Train/test split (75/25, stratified)
    # -----------------------------------------------------------------------
    print("\n[3/6] Splitting data (75/25)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train)}, Test: {len(X_test)}")
    
    # -----------------------------------------------------------------------
    # 4. Train model
    # -----------------------------------------------------------------------
    print("\n[4/6] Training XGBoost classifier...")
    predictor = MatchPredictor()
    predictor.train(X_train, y_train, feature_names)
    print("  Training complete.")
    
    # -----------------------------------------------------------------------
    # 5. Evaluate
    # -----------------------------------------------------------------------
    print("\n[5/6] Evaluating model...")
    
    # --- Training accuracy ---
    train_pred = predictor.predict(X_train)
    train_acc = accuracy_score(y_train, train_pred)
    print(f"\n  Training accuracy: {train_acc:.4f} ({train_acc*100:.1f}%)")
    
    # --- Test accuracy ---
    test_pred = predictor.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    print(f"  Test accuracy (3-way W/D/L): {test_acc:.4f} ({test_acc*100:.1f}%)")
    
    # --- Binary accuracy (correct winner, excluding draws) ---
    # For matches that weren't draws, did we predict the right winner?
    non_draw_mask = y_test != 1
    if non_draw_mask.sum() > 0:
        # Map: 0 -> team1 wins, 2 -> team2 wins
        # A prediction is "binary correct" if:
        #   actual=0 and predicted=0 (correct home win)
        #   actual=2 and predicted=2 (correct away win)
        #   We also count predicted=1 but actual!=1 as wrong
        binary_correct = (test_pred[non_draw_mask] == y_test[non_draw_mask]).sum()
        binary_total = non_draw_mask.sum()
        binary_acc = binary_correct / binary_total
        print(f"  Binary accuracy (winner only, no draws): {binary_acc:.4f} ({binary_acc*100:.1f}%)")
    
    # --- Log loss ---
    test_proba = predictor.predict_proba(X_test)
    ll = log_loss(y_test, test_proba)
    print(f"  Log loss: {ll:.4f}")
    
    # --- Classification report ---
    print(f"\n  Classification Report:")
    print(classification_report(y_test, test_pred,
                                target_names=["Home Win", "Draw", "Away Win"]))
    
    # --- Confusion matrix ---
    cm = confusion_matrix(y_test, test_pred)
    print(f"  Confusion Matrix:")
    print(f"                Pred:Home  Pred:Draw  Pred:Away")
    for i, label in enumerate(["Actual:Home", "Actual:Draw", "Actual:Away"]):
        print(f"  {label:14s}  {cm[i][0]:9d}  {cm[i][1]:9d}  {cm[i][2]:9d}")
    
    # --- Cross-validation ---
    print(f"\n  5-Fold Stratified Cross-Validation...")
    from sklearn.model_selection import StratifiedKFold
    from xgboost import XGBClassifier
    cv_model = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.08,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        objective="multi:softprob", num_class=3,
        eval_metric="mlogloss", random_state=42, n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(cv_model, X, y, cv=cv, scoring="accuracy")
    print(f"  CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    print(f"  CV Scores: {[f'{s:.4f}' for s in cv_scores]}")
    
    # --- Feature importance (top 10) ---
    importance = predictor.get_feature_importance()
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  Top 10 Feature Importances:")
    for name, imp in sorted_imp[:10]:
        bar = "#" * int(imp * 100)
        print(f"    {name:25s} {imp:.4f} {bar}")
    
    # -----------------------------------------------------------------------
    # 6. Save model
    # -----------------------------------------------------------------------
    print("\n[6/6] Saving model...")
    model_path = os.path.join("models", "model.pkl")
    predictor.save(model_path)
    print(f"  Model saved to {model_path}")
    
    # -----------------------------------------------------------------------
    # Quick prediction test
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Quick Prediction Test")
    print("=" * 60)
    
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model import WorldCupPredictor
    wp = WorldCupPredictor(
        model_path,
        os.path.join("data", "processed", "team_profiles.json"),
        os.path.join("data", "processed", "player_stats.json")
    )
    
    test_matches = [
        ("Argentina", "France", "Final"),
        ("Brazil", "Germany", "Semi-finals"),
        ("England", "Spain", "Quarter-finals"),
        ("United States", "Mexico", "Group stage"),
        ("Morocco", "Portugal", "Round of 16"),
    ]
    
    for t1, t2, stage in test_matches:
        result = wp.predict(t1, t2, stage)
        print(f"\n  {t1} vs {t2} ({stage})")
        print(f"    Winner: {result['winner']} ({result['confidence']:.1f}% confidence)")
        print(f"    Score:  {result['score'][t1]}-{result['score'][t2]}")
        print(f"    xG:     {result['xg'][t1]}-{result['xg'][t2]}")
        print(f"    MOTM:   {result['motm']['name']} ({result['motm']['team']})")
        probs = result['probabilities']
        print(f"    Probs:  {t1}: {probs[t1]}% | Draw: {probs['Draw']}% | {t2}: {probs[t2]}%")
    
    print("\n" + "=" * 60)
    print("Training complete! Model ready for predictions.")
    print("=" * 60)


if __name__ == "__main__":
    main()
