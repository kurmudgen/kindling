"""
Streaming-signals vs valence-only ablation.

The Phase 7 CV deep-dive showed that of 10 features, only 3 are
statistically meaningful — and they're all "complexity"-flavored.
The runtime token-level signals (probability spread, surprise score,
attention anomaly) — the supposedly novel part — appeared at noise level.

This test answers: does the streaming-signal architecture justify itself?

Three feature sets compared via stratified 5-fold CV (10 seeds each):
  A. ALL:       all 10 features (baseline, what we ship)
  B. VALENCE:   only the pre-generation features (urgency, complexity,
                stakes, composite) — 4 features
  C. STREAMING: only the runtime token signals (probabilitySpread,
                semanticVelocity, surpriseScore, attentionAnomalyScore)
                — 4 features

If A ≈ B, the streaming signals don't justify their architectural cost.
If A > B, they pull weight even if individually noisy.
If A < B, the streaming signals are actively hurting the classifier.
"""
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).parent.parent
TRAINING_FILE = ROOT / "logs" / "shadow" / "training.jsonl"


def load_examples():
    examples = []
    with open(TRAINING_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    examples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return examples


def features_all(ex):
    s, v = ex["signals"], ex["valence"]
    return [
        s["tokenProbabilitySpread"], s["semanticVelocity"],
        s["surpriseScore"], s["attentionAnomalyScore"],
        v["urgency"], v["complexity"], v["stakes"], v["composite"],
        s["tokenProbabilitySpread"] * v["complexity"],
        s["surpriseScore"] * v["stakes"],
    ]


def features_valence(ex):
    v = ex["valence"]
    return [v["urgency"], v["complexity"], v["stakes"], v["composite"]]


def features_streaming(ex):
    s = ex["signals"]
    return [
        s["tokenProbabilitySpread"], s["semanticVelocity"],
        s["surpriseScore"], s["attentionAnomalyScore"],
    ]


def make_model():
    return LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                              class_weight="balanced", random_state=42)


def cv_score(X, y, n_seeds=10):
    f1s, accs = [], []
    for seed in range(n_seeds):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        f1s.append(cross_val_score(make_model(), Xs, y, cv=skf, scoring="f1").mean())
        accs.append(cross_val_score(make_model(), Xs, y, cv=skf, scoring="accuracy").mean())
    return np.array(f1s), np.array(accs)


def main():
    examples = load_examples()
    y = np.array([1 if ex["escalationNeeded"] else 0 for ex in examples])
    print(f"Dataset: {len(examples)} examples, {y.sum()} positive ({y.mean()*100:.1f}%)\n")

    sets = [
        ("ALL (10 features — current ship)", features_all),
        ("VALENCE-ONLY (4 features — pre-gen complexity scoring)", features_valence),
        ("STREAMING-ONLY (4 features — runtime token signals)", features_streaming),
    ]

    results = {}
    print(f"{'Feature set':<55} {'CV F1':>15} {'CV Accuracy':>15}")
    print("-" * 90)
    for name, fn in sets:
        X = np.array([fn(ex) for ex in examples])
        f1s, accs = cv_score(X, y)
        results[name] = (f1s, accs)
        print(f"{name:<55} {f1s.mean():.3f} ± {f1s.std():.3f}    {accs.mean():.3f} ± {accs.std():.3f}")

    print()
    all_f1, _ = results["ALL (10 features — current ship)"]
    val_f1, _ = results["VALENCE-ONLY (4 features — pre-gen complexity scoring)"]
    str_f1, _ = results["STREAMING-ONLY (4 features — runtime token signals)"]

    delta_all_val = all_f1.mean() - val_f1.mean()
    delta_all_str = all_f1.mean() - str_f1.mean()
    delta_val_str = val_f1.mean() - str_f1.mean()

    print(f"ALL vs VALENCE-only:    delta F1 = {delta_all_val:+.3f}")
    print(f"ALL vs STREAMING-only:  delta F1 = {delta_all_str:+.3f}")
    print(f"VALENCE vs STREAMING:   delta F1 = {delta_val_str:+.3f}")
    print()

    if abs(delta_all_val) < 0.02:
        print("=> Streaming signals are NOT pulling meaningful weight.")
        print("  Pre-generation valence scoring achieves nearly the same F1 with 60%")
        print("  fewer features. The streaming-signal architecture has implementation")
        print("  cost (per-token probability tracking, semantic velocity calc, etc.)")
        print("  but doesn't translate to better escalation decisions on this data.")
    elif delta_all_val > 0.05:
        print("=> Streaming signals DO contribute meaningfully (>0.05 F1 lift).")
        print("  The architecture justifies its cost.")
    else:
        print("=> Marginal benefit (0.02–0.05 F1). Whether streaming signals are worth")
        print("  the complexity is a judgment call — depends on whether you value")
        print("  every percent of F1 or simpler code.")


if __name__ == "__main__":
    main()
