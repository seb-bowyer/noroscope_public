#!/usr/bin/env python3
"""Train and evaluate a group-held-out candidate-ranking model.

This generalised example demonstrates a modelling pattern used in biological
sequence analysis: candidates associated with the same biological position are
kept together during train/test splitting, so the model is evaluated on groups
it has not seen during fitting.

The script is intentionally independent of any unpublished research dataset,
label definition or publication feature catalogue.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight

FEATURES = [
    "language_model_score",
    "site_entropy",
    "candidate_frequency",
    "structure_score",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a classifier with position-grouped holdout and rank candidates."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_input(df: pd.DataFrame) -> None:
    required = {"position", "candidate", "target", *FEATURES}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")
    if df.duplicated(["position", "candidate"]).any():
        raise ValueError("Each position-candidate pair must be unique")
    if not set(df["target"].dropna().unique()).issubset({0, 1}):
        raise ValueError("target must be binary (0/1)")
    if df["position"].nunique() < 4:
        raise ValueError("At least four positions are required for grouped evaluation")


def safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, scores))


def rank_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    positive_ranks: list[int] = []
    positions_with_positive = 0
    best_positive_ranks: list[int] = []

    ranked_frames = []
    for position, group in predictions.groupby("position", sort=True):
        ranked = group.sort_values(
            ["score", "candidate"], ascending=[False, True], kind="mergesort"
        ).copy()
        ranked["within_position_rank"] = np.arange(1, len(ranked) + 1)
        ranked_frames.append(ranked)

        positives = ranked.loc[ranked["target"] == 1, "within_position_rank"]
        if not positives.empty:
            positions_with_positive += 1
            positive_ranks.extend(positives.astype(int).tolist())
            best_positive_ranks.append(int(positives.min()))

    if not positive_ranks:
        return {
            "positions_with_positive": 0,
            "mean_positive_rank": float("nan"),
            "median_best_positive_rank": float("nan"),
            "positive_hit_at_3": float("nan"),
        }

    ranks = np.asarray(positive_ranks)
    return {
        "positions_with_positive": int(positions_with_positive),
        "mean_positive_rank": float(ranks.mean()),
        "median_best_positive_rank": float(np.median(best_positive_ranks)),
        "positive_hit_at_3": float(np.mean(ranks <= 3)),
    }


def main() -> int:
    args = parse_args()
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1")

    df = pd.read_csv(args.input)
    validate_input(df)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=args.test_size,
        random_state=args.seed,
    )
    train_idx, test_idx = next(
        splitter.split(df, df["target"], groups=df["position"])
    )

    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()

    overlap = set(train["position"]) & set(test["position"])
    if overlap:
        raise RuntimeError(f"Group leakage detected: {sorted(overlap)[:5]}")
    if train["target"].nunique() < 2 or test["target"].nunique() < 2:
        raise RuntimeError(
            "This split contains only one class in train or test. Try a different seed."
        )

    X_train = train[FEATURES].apply(pd.to_numeric, errors="coerce")
    X_test = test[FEATURES].apply(pd.to_numeric, errors="coerce")
    y_train = train["target"].to_numpy(dtype=np.int8)
    y_test = test["target"].to_numpy(dtype=np.int8)

    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=250,
        max_depth=4,
        min_samples_leaf=15,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=args.seed,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)

    test["score"] = model.predict_proba(X_test)[:, 1]
    test["within_position_rank"] = test.groupby("position")["score"].rank(
        method="first", ascending=False
    )

    metrics = {
        "seed": args.seed,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_positions": int(train["position"].nunique()),
        "test_positions": int(test["position"].nunique()),
        "average_precision": float(average_precision_score(y_test, test["score"])),
        "roc_auc": safe_auc(y_test, test["score"].to_numpy()),
        **rank_metrics(test),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    test.sort_values(["position", "within_position_rank"]).to_csv(
        args.output_dir / "test_predictions.csv", index=False
    )
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
