#!/usr/bin/env python3
"""Create a deterministic synthetic dataset for the grouped ranking demo."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

AMINO_ACIDS = np.array(list("ACDEFGHIKLMNPQRSTVWY"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("example_data/candidates.csv"))
    parser.add_argument("--positions", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    rows = []

    for position in range(1, args.positions + 1):
        site_entropy = float(rng.beta(2.0, 4.0))
        position_shift = float(rng.normal(0.0, 0.4))
        candidates = rng.choice(AMINO_ACIDS, size=12, replace=False)

        latent_scores = []
        candidate_rows = []
        for candidate in candidates:
            language_model_score = float(rng.normal(position_shift, 1.0))
            candidate_frequency = float(rng.beta(1.4, 6.0))
            structure_score = float(rng.normal(0.0, 1.0))
            latent = (
                1.2 * language_model_score
                + 1.8 * candidate_frequency
                - 0.45 * abs(structure_score)
                + 0.35 * site_entropy
                + rng.normal(0.0, 0.7)
            )
            latent_scores.append(latent)
            candidate_rows.append(
                {
                    "position": position,
                    "candidate": candidate,
                    "language_model_score": language_model_score,
                    "site_entropy": site_entropy,
                    "candidate_frequency": candidate_frequency,
                    "structure_score": structure_score,
                }
            )

        # Mark the strongest synthetic candidate as a positive, plus an
        # occasional second positive. This is purely demonstration data.
        order = np.argsort(latent_scores)[::-1]
        positives = {int(order[0])}
        if rng.random() < 0.35:
            positives.add(int(order[1]))
        for idx, row in enumerate(candidate_rows):
            row["target"] = int(idx in positives)
            rows.append(row)

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df):,} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
