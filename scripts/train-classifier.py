"""
Kindling Meta-Confidence Classifier — Training Script

Reads shadow training data (logs/shadow/training.jsonl), trains a
logistic regression or gradient-boosted tree classifier, and exports
the model weights as JSON for the TypeScript inference engine.

GPU KICKSTART: This script uses sklearn/xgboost which can leverage
GPU for training (especially XGBoost with tree_method='gpu_hist').
The exported model runs on CPU as pure math in TypeScript.

Usage:
    python scripts/train-classifier.py                    # logistic regression (default)
    python scripts/train-classifier.py --model xgboost    # gradient-boosted tree
    python scripts/train-classifier.py --model xgboost --gpu  # XGBoost on GPU

Requirements:
    pip install scikit-learn numpy
    pip install xgboost  # optional, for --model xgboost
"""

import json
import sys
import os
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# ─── Config ──────────────────────────────────────────────────────

TRAINING_FILE = Path(__file__).parent.parent / "logs" / "shadow" / "training.jsonl"
MODEL_OUTPUT = Path(__file__).parent.parent / "models" / "meta-classifier.json"

FEATURE_NAMES = [
    "tokenProbabilitySpread",
    "semanticVelocity",
    "surpriseScore",
    "attentionAnomalyScore",
    "valenceUrgency",
    "valenceComplexity",
    "valenceStakes",
    "valenceComposite",
    # Interaction features
    "spreadXcomplexity",
    "surpriseXstakes",
]

MIN_EXAMPLES = 20  # minimum training examples before we'll train


# ─── Feature extraction (mirrors TypeScript extractFeatures) ─────

def extract_features(example: dict) -> list[float]:
    """Extract feature vector from a training example — must match
    TypeScript extractFeatures() exactly."""
    signals = example["signals"]
    valence = example["valence"]
    return [
        signals["tokenProbabilitySpread"],
        signals["semanticVelocity"],
        signals["surpriseScore"],
        signals["attentionAnomalyScore"],
        valence["urgency"],
        valence["complexity"],
        valence["stakes"],
        valence["composite"],
        # Interaction features
        signals["tokenProbabilitySpread"] * valence["complexity"],
        signals["surpriseScore"] * valence["stakes"],
    ]


# ─── Data loading ────────────────────────────────────────────────

def load_training_data() -> tuple[np.ndarray, np.ndarray]:
    """Load training examples from JSONL file."""
    if not TRAINING_FILE.exists():
        print(f"ERROR: Training file not found: {TRAINING_FILE}")
        print("Run Kindling with shadow evaluation enabled to generate training data.")
        sys.exit(1)

    examples = []
    with open(TRAINING_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if len(examples) < MIN_EXAMPLES:
        print(f"ERROR: Only {len(examples)} training examples found (need {MIN_EXAMPLES}+)")
        print("Run more queries with shadow evaluation to build up training data.")
        sys.exit(1)

    X = np.array([extract_features(ex) for ex in examples])
    y = np.array([1 if ex["escalationNeeded"] else 0 for ex in examples])

    print(f"Loaded {len(examples)} training examples")
    print(f"  Escalation needed: {y.sum()} ({y.mean()*100:.1f}%)")
    print(f"  Not needed: {len(y) - y.sum()} ({(1-y.mean())*100:.1f}%)")
    print(f"  Features: {len(FEATURE_NAMES)}")

    return X, y


# ─── Logistic Regression ────────────────────────────────────────

def train_logistic_regression(X: np.ndarray, y: np.ndarray) -> dict:
    """Train logistic regression and export weights."""
    print("\n--- Training Logistic Regression ---")

    # Standardize features for better convergence
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )

    # Cross-validation
    if len(X) >= 10:
        k = min(5, len(X) // 2)
        scores = cross_val_score(model, X_scaled, y, cv=k, scoring="accuracy")
        print(f"  Cross-validation accuracy ({k}-fold): {scores.mean():.3f} +/- {scores.std():.3f}")
    else:
        scores = np.array([0.0])
        print("  Too few examples for cross-validation")

    # Train on full dataset
    model.fit(X_scaled, y)

    # The exported weights must account for the scaler:
    # z = (x - mean) / std  =>  w_orig = w_scaled / std,  b_orig = b_scaled - sum(w_scaled * mean / std)
    weights_scaled = model.coef_[0]
    intercept_scaled = model.intercept_[0]

    weights_orig = weights_scaled / scaler.scale_
    intercept_orig = intercept_scaled - np.sum(weights_scaled * scaler.mean_ / scaler.scale_)

    print(f"\n  Weights (original scale):")
    for name, w in zip(FEATURE_NAMES, weights_orig):
        print(f"    {name:30s} {w:+.4f}")
    print(f"    {'intercept':30s} {intercept_orig:+.4f}")

    train_accuracy = model.score(X_scaled, y)
    print(f"\n  Training accuracy: {train_accuracy:.3f}")

    return {
        "type": "logistic_regression",
        "version": 1,
        "features": FEATURE_NAMES,
        "weights": weights_orig.tolist(),
        "intercept": float(intercept_orig),
        "threshold": 0.5,
        "trainedAt": datetime.utcnow().isoformat() + "Z",
        "trainingExamples": len(X),
        "accuracy": float(scores.mean()) if scores.mean() > 0 else float(train_accuracy),
    }


# ─── XGBoost (GPU kickstart) ────────────────────────────────────

def train_xgboost(X: np.ndarray, y: np.ndarray, use_gpu: bool = False) -> dict:
    """Train gradient-boosted tree and export as traversable tree structure."""
    try:
        import xgboost as xgb
    except ImportError:
        print("ERROR: xgboost not installed. Run: pip install xgboost")
        sys.exit(1)

    print("\n--- Training XGBoost ---")

    tree_method = "hist"
    device = "cpu"
    if use_gpu:
        device = "cuda"
        print("  Using GPU for training (cuda)")
    else:
        print("  Using CPU for training")

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "max_depth": 4,         # shallow trees for small dataset
        "n_estimators": 50,     # few trees — we want fast inference
        "learning_rate": 0.1,
        "tree_method": tree_method,
        "device": device,
        "random_state": 42,
    }

    model = xgb.XGBClassifier(**params)

    # Cross-validation
    if len(X) >= 10:
        k = min(5, len(X) // 2)
        scores = cross_val_score(model, X, y, cv=k, scoring="accuracy")
        print(f"  Cross-validation accuracy ({k}-fold): {scores.mean():.3f} +/- {scores.std():.3f}")
    else:
        scores = np.array([0.0])

    # Train on full dataset
    model.fit(X, y)
    train_accuracy = model.score(X, y)
    print(f"  Training accuracy: {train_accuracy:.3f}")

    # Export tree structure as JSON-traversable nodes
    booster = model.get_booster()
    trees_json = booster.get_dump(dump_format="json")
    trees = [_parse_xgb_tree(json.loads(t)) for t in trees_json]

    # Feature importance
    importance = model.feature_importances_
    print(f"\n  Feature importance:")
    for name, imp in sorted(zip(FEATURE_NAMES, importance), key=lambda x: -x[1]):
        bar = "#" * int(imp * 40)
        print(f"    {name:30s} {imp:.3f} {bar}")

    return {
        "type": "gradient_boosted_tree",
        "version": 1,
        "features": FEATURE_NAMES,
        "trees": trees,
        "learningRate": 0.1,
        "baseScore": 0.5,
        "threshold": 0.5,
        "trainedAt": datetime.utcnow().isoformat() + "Z",
        "trainingExamples": len(X),
        "accuracy": float(scores.mean()) if scores.mean() > 0 else float(train_accuracy),
    }


def _parse_xgb_tree(node: dict) -> dict | float:
    """Recursively parse XGBoost tree dump into our JSON format."""
    if "leaf" in node:
        return node["leaf"]

    # XGBoost uses "split" (feature name like "f0"), "split_condition" (threshold)
    feature_idx = int(node["split"].replace("f", ""))
    threshold = node["split_condition"]

    # "yes" = left child (feature <= threshold), "no" = right child
    children = node.get("children", [])
    left = _parse_xgb_tree(children[0]) if len(children) > 0 else 0.0
    right = _parse_xgb_tree(children[1]) if len(children) > 1 else 0.0

    return {
        "feature": feature_idx,
        "threshold": threshold,
        "left": left,
        "right": right,
    }


# ─── Main ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train Kindling meta-confidence classifier")
    parser.add_argument("--model", choices=["logreg", "xgboost"], default="logreg",
                        help="Model type (default: logreg)")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU for XGBoost training")
    args = parser.parse_args()

    X, y = load_training_data()

    if args.model == "xgboost":
        model_json = train_xgboost(X, y, use_gpu=args.gpu)
    else:
        model_json = train_logistic_regression(X, y)

    # Write model
    MODEL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUTPUT, "w") as f:
        json.dump(model_json, f, indent=2)

    print(f"\nModel exported to: {MODEL_OUTPUT}")
    print(f"Type: {model_json['type']}")
    print(f"Training examples: {model_json['trainingExamples']}")
    print(f"Accuracy: {model_json['accuracy']:.3f}")
    print(f"\nKindling will auto-load this model on next startup.")
    print("To force reload during runtime: meta.reloadMLClassifier()")


if __name__ == "__main__":
    main()
