"""Reproducible pre-call classification benchmark for UCI Bank Marketing."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score,
                             RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]


def make_pipeline(X, model):
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = X.select_dtypes(exclude="number").columns.tolist()
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                  ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), categorical),
    ])
    return Pipeline([("preprocess", preprocessing), ("model", model)])


def psi(reference, observed, bins=10):
    """PSI with full-tail coverage, unique bins and normalized smoothing."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        center = float(edges[0])
        edges = np.array([-np.inf, center - 1e-8, center + 1e-8, np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(reference, edges)[0].astype(float) + 0.5
    q = np.histogram(observed, edges)[0].astype(float) + 0.5
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum((q - p) * np.log(q / p)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/bank-additional-full.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/current")
    parser.add_argument("--include-duration", action="store_true", help="Post-call research benchmark only")
    parser.add_argument("--sample", type=int, help="Optional stratified subset for a quick check")
    args = parser.parse_args()
    df = pd.read_csv(args.data, sep=";")
    if not set(df["y"].unique()).issubset({"yes", "no"}):
        raise ValueError("Expected target y with yes/no values")
    if args.sample:
        if not 10 <= args.sample < len(df):
            parser.error("--sample must be >= 10 and smaller than the dataset")
        df, _ = train_test_split(df, train_size=args.sample, stratify=df.y, random_state=42)
    X = df.drop(columns=["y"] + ([] if args.include_duration else ["duration"]))
    y = df.y.eq("yes").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, stratify=y, random_state=42)
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, model in [
        ("Logistic regression", LogisticRegression(max_iter=2000)),
        ("Gradient boosting", GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ]:
        pipeline = make_pipeline(X_train, model).fit(X_train, y_train)
        probabilities = pipeline.predict_proba(X_test)[:, 1]
        rows.append({"model": name, "roc_auc": roc_auc_score(y_test, probabilities),
                     "average_precision": average_precision_score(y_test, probabilities),
                     "f1_at_0.5": f1_score(y_test, probabilities >= .5)})
        RocCurveDisplay.from_predictions(y_test, probabilities, name=name, ax=axes[0])
        PrecisionRecallDisplay.from_predictions(y_test, probabilities, name=name, ax=axes[1])
    axes[0].set_title("Held-out ROC curves")
    axes[1].set_title("Held-out precision / recall")
    fig.tight_layout()
    fig.savefig(args.output / "evaluation.png", dpi=160)
    plt.close(fig)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(args.output / "metrics.csv", index=False)
    # Synthetic feature shift: distribution diagnostics only, no invented outcomes.
    shifted = X_test.copy()
    shifted["euribor3m"] *= 1.5
    shifted["cons.conf.idx"] -= 5
    drift = [{"feature": col, "psi": psi(X_test[col], shifted[col]),
              "ks_statistic": float(ks_2samp(X_test[col], shifted[col]).statistic),
              "wasserstein": float(wasserstein_distance(X_test[col], shifted[col]))}
             for col in ["euribor3m", "cons.conf.idx", "age"]]
    pd.DataFrame(drift).to_csv(args.output / "synthetic_drift.csv", index=False)
    manifest = {"seed": 42, "train_rows": len(X_train), "test_rows": len(X_test),
                "duration_included": args.include_duration, "sample": args.sample,
                "feature_names": list(X.columns), "positive_rate": float(y.mean()),
                "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest()}
    (args.output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
