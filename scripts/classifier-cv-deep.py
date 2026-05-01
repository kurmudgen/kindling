"""
Deep cross-validation analysis of the meta-confidence classifier.

Goes beyond the basic CV in train-classifier.py:
- Stratified k-fold (k=5 and k=10) with per-fold detail
- Per-class precision/recall/F1 (the escalation-needed class is what matters)
- Confusion matrix
- Stability across random seeds (does this depend on luck?)
- Permutation feature importance (which features actually contribute?)
- Train-vs-test gap (is it overfitting?)

No LLM calls, no PC strain — pure analysis of the 90 examples we have.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).parent.parent
TRAINING_FILE = ROOT / "logs" / "shadow" / "training.jsonl"

FEATURE_NAMES = [
    "tokenProbabilitySpread",
    "semanticVelocity",
    "surpriseScore",
    "attentionAnomalyScore",
    "valenceUrgency",
    "valenceComplexity",
    "valenceStakes",
    "valenceComposite",
    "spreadXcomplexity",
    "surpriseXstakes",
]


def extract_features(ex):
    s, v = ex["signals"], ex["valence"]
    return [
        s["tokenProbabilitySpread"],
        s["semanticVelocity"],
        s["surpriseScore"],
        s["attentionAnomalyScore"],
        v["urgency"],
        v["complexity"],
        v["stakes"],
        v["composite"],
        s["tokenProbabilitySpread"] * v["complexity"],
        s["surpriseScore"] * v["stakes"],
    ]


def load_data():
    examples = []
    with open(TRAINING_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    X = np.array([extract_features(ex) for ex in examples])
    y = np.array([1 if ex["escalationNeeded"] else 0 for ex in examples])
    return X, y, examples


def make_model():
    return LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )


def main():
    print("=" * 70)
    print("Meta-Confidence Classifier — Deep Cross-Validation")
    print("=" * 70)

    X, y, examples = load_data()
    n = len(y)
    pos = int(y.sum())
    neg = n - pos
    print(f"\nDataset: {n} examples")
    print(f"  Escalation-needed (positive): {pos} ({pos/n*100:.1f}%)")
    print(f"  No escalation (negative):     {neg} ({neg/n*100:.1f}%)")
    print(f"  Class balance: {'balanced' if 0.4 <= pos/n <= 0.6 else 'imbalanced — F1 matters more than accuracy'}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Stratified k-fold at k=5 and k=10 ---
    print("\n" + "=" * 70)
    print("STRATIFIED CROSS-VALIDATION")
    print("=" * 70)
    for k in [5, 10]:
        print(f"\n--- k={k} ---")
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        accs = cross_val_score(make_model(), X_scaled, y, cv=skf, scoring="accuracy")
        f1s = cross_val_score(make_model(), X_scaled, y, cv=skf, scoring="f1")
        precs = cross_val_score(make_model(), X_scaled, y, cv=skf, scoring="precision")
        recs = cross_val_score(make_model(), X_scaled, y, cv=skf, scoring="recall")
        print(f"  Accuracy:  {accs.mean():.3f} +/- {accs.std():.3f}   per-fold: {[f'{x:.2f}' for x in accs]}")
        print(f"  F1:        {f1s.mean():.3f} +/- {f1s.std():.3f}   per-fold: {[f'{x:.2f}' for x in f1s]}")
        print(f"  Precision: {precs.mean():.3f} +/- {precs.std():.3f}")
        print(f"  Recall:    {recs.mean():.3f} +/- {recs.std():.3f}")

    # --- Confusion matrix from cross-val predictions ---
    print("\n" + "=" * 70)
    print("CROSS-VAL CONFUSION MATRIX (k=5)")
    print("=" * 70)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(make_model(), X_scaled, y, cv=skf)
    cm = confusion_matrix(y, y_pred)
    print(f"\n                pred no   pred yes")
    print(f"  actual no    {cm[0,0]:6d}    {cm[0,1]:6d}")
    print(f"  actual yes   {cm[1,0]:6d}    {cm[1,1]:6d}")
    print()
    print(classification_report(y, y_pred, target_names=["no_escalation", "escalation"], digits=3))

    # --- Stability across random seeds ---
    print("=" * 70)
    print("STABILITY ACROSS RANDOM SEEDS (k=5, 10 seeds)")
    print("=" * 70)
    seed_f1s = []
    seed_accs = []
    for seed in range(10):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        f1 = cross_val_score(make_model(), X_scaled, y, cv=skf, scoring="f1").mean()
        acc = cross_val_score(make_model(), X_scaled, y, cv=skf, scoring="accuracy").mean()
        seed_f1s.append(f1)
        seed_accs.append(acc)
    seed_f1s = np.array(seed_f1s)
    seed_accs = np.array(seed_accs)
    print(f"\n  Accuracy across seeds: mean={seed_accs.mean():.3f}  std={seed_accs.std():.3f}  range=[{seed_accs.min():.3f}, {seed_accs.max():.3f}]")
    print(f"  F1 across seeds:       mean={seed_f1s.mean():.3f}  std={seed_f1s.std():.3f}  range=[{seed_f1s.min():.3f}, {seed_f1s.max():.3f}]")
    if seed_f1s.std() > 0.05:
        print("  WARNING: F1 varies by >0.05 across seeds — model is sensitive to fold split (small dataset effect)")

    # --- Train-vs-test gap (overfit check) ---
    print("\n" + "=" * 70)
    print("TRAIN-VS-TEST GAP (overfit check)")
    print("=" * 70)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_accs = []
    test_accs = []
    for train_idx, test_idx in skf.split(X_scaled, y):
        m = make_model().fit(X_scaled[train_idx], y[train_idx])
        train_accs.append(m.score(X_scaled[train_idx], y[train_idx]))
        test_accs.append(m.score(X_scaled[test_idx], y[test_idx]))
    train_accs = np.array(train_accs)
    test_accs = np.array(test_accs)
    print(f"\n  Train accuracy avg: {train_accs.mean():.3f}")
    print(f"  Test accuracy avg:  {test_accs.mean():.3f}")
    gap = train_accs.mean() - test_accs.mean()
    print(f"  Gap:                {gap:.3f}")
    if gap > 0.10:
        print("  WARNING: train-test gap > 0.10 — likely overfitting (small dataset, complex features)")
    elif gap > 0.05:
        print("  Gap is moderate — some overfit but classifier generalizes")
    else:
        print("  Gap looks healthy")

    # --- Permutation importance ---
    print("\n" + "=" * 70)
    print("PERMUTATION FEATURE IMPORTANCE (which features actually matter)")
    print("=" * 70)
    full_model = make_model().fit(X_scaled, y)
    perm = permutation_importance(full_model, X_scaled, y, n_repeats=20, random_state=42, scoring="f1")
    importance_pairs = sorted(zip(FEATURE_NAMES, perm.importances_mean, perm.importances_std), key=lambda p: -p[1])
    print(f"\n  {'Feature':<28} {'mean drop in F1':>16} {'std':>8}")
    for name, mean, std in importance_pairs:
        flag = "  <-- noise" if abs(mean) < 2 * std and mean > 0 else ""
        print(f"  {name:<28} {mean:>16.4f} {std:>8.4f}{flag}")
    print("\n  (Higher = more important. Features with |mean| < 2*std are statistically indistinguishable from noise.)")

    print("\n" + "=" * 70)
    print("Summary: classifier is " + (
        "production-ready" if seed_f1s.mean() > 0.75 and gap < 0.10 else
        "useful but small-data — F1 stability and overfit gap are the limits"
    ))
    print("=" * 70)


if __name__ == "__main__":
    main()
