#!/usr/bin/env python3
"""Build and audit a compact DuckDB feature database for protein mutation data.

This is a generalised public code sample adapted from a larger research
pipeline. It intentionally excludes unpublished project-specific labels,
structural datasets and publication logic.

Inputs
------
sequences.csv:
    sequence_id, sequence
llr_scores.csv:
    sequence_id, position, wt_residue, mut_residue, llr
annotations.csv:
    position, region_label

The resulting database separates raw inputs, derived features and audit checks
into distinct schemas. Derived features include masked-language-model residue
probabilities, site entropy, within-position LLR rank and distance from the
best-scoring candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
AA_SET = set(AMINO_ACIDS)


class ValidationError(RuntimeError):
    """Raised when a required data invariant is violated."""


@dataclass(frozen=True)
class Paths:
    sequences: Path
    llr_scores: Path
    annotations: Path
    database: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a small audited DuckDB feature database."
    )
    parser.add_argument("--sequences", required=True, type=Path)
    parser.add_argument("--llr-scores", required=True, type=Path)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--replace", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_sequences(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"sequence_id", "sequence"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValidationError(f"Sequence file missing columns: {missing}")

    df = df.loc[:, ["sequence_id", "sequence"]].copy()
    df["sequence_id"] = df["sequence_id"].astype("string").str.strip()
    df["sequence"] = df["sequence"].astype("string").str.strip().str.upper()

    if df["sequence_id"].isna().any() or df["sequence_id"].eq("").any():
        raise ValidationError("Every sequence requires a non-empty sequence_id")
    if df["sequence_id"].duplicated().any():
        duplicated = df.loc[df["sequence_id"].duplicated(), "sequence_id"].tolist()
        raise ValidationError(f"Duplicate sequence_id values: {duplicated[:5]}")
    if df["sequence"].isna().any() or df["sequence"].eq("").any():
        raise ValidationError("Every sequence requires an amino-acid sequence")

    lengths = df["sequence"].str.len()
    if lengths.nunique() != 1:
        raise ValidationError(
            "This demonstration expects aligned sequences of equal length; "
            f"observed lengths={sorted(lengths.unique().tolist())}"
        )

    invalid = {
        sequence_id: sorted(set(sequence) - AA_SET)
        for sequence_id, sequence in zip(df["sequence_id"], df["sequence"])
        if set(sequence) - AA_SET
    }
    if invalid:
        first_id = next(iter(invalid))
        raise ValidationError(
            f"Non-canonical residues in {first_id}: {invalid[first_id]}"
        )

    df["sequence_hash"] = df["sequence"].map(
        lambda seq: hashlib.sha256(seq.encode("ascii")).hexdigest()
    )
    return df


def load_llrs(path: Path, sequences: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"sequence_id", "position", "wt_residue", "mut_residue", "llr"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValidationError(f"LLR file missing columns: {missing}")

    df = df.loc[:, sorted(required)].copy()
    df["sequence_id"] = df["sequence_id"].astype("string").str.strip()
    df["position"] = pd.to_numeric(df["position"], errors="raise").astype(int)
    df["llr"] = pd.to_numeric(df["llr"], errors="raise").astype(float)
    for column in ["wt_residue", "mut_residue"]:
        df[column] = df[column].astype("string").str.strip().str.upper()

    known_ids = set(sequences["sequence_id"])
    unknown_ids = sorted(set(df["sequence_id"]) - known_ids)
    if unknown_ids:
        raise ValidationError(f"LLR rows reference unknown sequence IDs: {unknown_ids[:5]}")

    if (~df["wt_residue"].isin(AMINO_ACIDS)).any() or (~df["mut_residue"].isin(AMINO_ACIDS)).any():
        raise ValidationError("LLR table contains non-canonical residue keys")

    duplicated = df.duplicated(["sequence_id", "position", "mut_residue"], keep=False)
    if duplicated.any():
        raise ValidationError("Duplicate sequence-position-candidate keys in LLR input")

    site_sizes = df.groupby(["sequence_id", "position"]).size()
    bad_sizes = site_sizes[site_sizes != 20]
    if not bad_sizes.empty:
        raise ValidationError(
            "Each scored sequence-position must contain exactly 20 candidate residues; "
            f"bad examples={bad_sizes.head().to_dict()}"
        )

    sequence_lookup = sequences.set_index("sequence_id")["sequence"].to_dict()
    expected_wt = []
    for row in df.itertuples(index=False):
        sequence = sequence_lookup[str(row.sequence_id)]
        if not 1 <= int(row.position) <= len(sequence):
            raise ValidationError(
                f"Position {row.position} outside sequence {row.sequence_id}"
            )
        expected_wt.append(sequence[int(row.position) - 1])

    mismatch = df["wt_residue"].to_numpy() != np.asarray(expected_wt)
    if mismatch.any():
        example = df.loc[mismatch].iloc[0]
        raise ValidationError(
            "Wild-type residue does not match aligned sequence at "
            f"{example.sequence_id}:{example.position}"
        )

    return df


def load_annotations(path: Path, alignment_length: int) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"position", "region_label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValidationError(f"Annotation file missing columns: {missing}")

    df = df.loc[:, ["position", "region_label"]].copy()
    df["position"] = pd.to_numeric(df["position"], errors="raise").astype(int)
    df["region_label"] = df["region_label"].astype("string").fillna("")
    if df["position"].duplicated().any():
        raise ValidationError("Annotation positions must be unique")

    expected = set(range(1, alignment_length + 1))
    observed = set(df["position"])
    if observed != expected:
        raise ValidationError(
            "Annotations must cover the complete aligned coordinate system"
        )
    return df.sort_values("position")


def build_database(paths: Paths, replace: bool) -> dict[str, object]:
    for input_path in [paths.sequences, paths.llr_scores, paths.annotations]:
        if not input_path.exists():
            raise FileNotFoundError(input_path)

    if paths.database.exists():
        if not replace:
            raise FileExistsError(
                f"Database already exists: {paths.database}. Use --replace to rebuild."
            )
        paths.database.unlink()
    paths.database.parent.mkdir(parents=True, exist_ok=True)

    sequences = load_sequences(paths.sequences)
    alignment_length = int(sequences["sequence"].str.len().iloc[0])
    llrs = load_llrs(paths.llr_scores, sequences)
    annotations = load_annotations(paths.annotations, alignment_length)

    con = duckdb.connect(str(paths.database))
    try:
        for schema in ["source", "features", "audit"]:
            con.execute(f'CREATE SCHEMA "{schema}"')

        con.register("sequences_df", sequences)
        con.execute(
            """
            CREATE TABLE source.sequences AS
            SELECT sequence_id::VARCHAR AS sequence_id,
                   sequence::VARCHAR AS sequence,
                   sequence_hash::VARCHAR AS sequence_hash
            FROM sequences_df
            """
        )
        con.unregister("sequences_df")

        con.register("llr_df", llrs)
        con.execute(
            """
            CREATE TABLE source.llr_scores AS
            SELECT sequence_id::VARCHAR AS sequence_id,
                   position::INTEGER AS position,
                   wt_residue::VARCHAR AS wt_residue,
                   mut_residue::VARCHAR AS mut_residue,
                   llr::DOUBLE AS llr
            FROM llr_df
            """
        )
        con.unregister("llr_df")

        con.register("annotations_df", annotations)
        con.execute(
            """
            CREATE TABLE source.annotations AS
            SELECT position::INTEGER AS position,
                   region_label::VARCHAR AS region_label
            FROM annotations_df
            """
        )
        con.unregister("annotations_df")

        # LLRs are relative to the WT residue. Re-normalising exp(LLR) across
        # the 20 amino acids therefore recovers the model's residue distribution
        # up to the same site-specific constant.
        con.execute(
            """
            CREATE VIEW features.mutation_features AS
            WITH shifted AS (
                SELECT *,
                       MAX(llr) OVER (PARTITION BY sequence_id, position) AS site_max_llr
                FROM source.llr_scores
            ), probabilities AS (
                SELECT *,
                       EXP(llr - site_max_llr)
                       / SUM(EXP(llr - site_max_llr)) OVER (
                           PARTITION BY sequence_id, position
                       ) AS residue_probability
                FROM shifted
            ), enriched AS (
                SELECT *,
                       -SUM(
                           CASE WHEN residue_probability > 0
                                THEN residue_probability * LN(residue_probability)
                                ELSE 0 END
                       ) OVER (PARTITION BY sequence_id, position) AS site_entropy,
                       RANK() OVER (
                           PARTITION BY sequence_id, position
                           ORDER BY llr DESC
                       )::INTEGER AS llr_rank,
                       MAX(llr) OVER (PARTITION BY sequence_id, position) - llr
                           AS distance_from_best
                FROM probabilities
            )
            SELECT e.sequence_id,
                   e.position,
                   e.wt_residue,
                   e.mut_residue,
                   e.llr,
                   e.residue_probability,
                   e.site_entropy,
                   e.llr_rank,
                   e.distance_from_best,
                   a.region_label
            FROM enriched e
            LEFT JOIN source.annotations a USING (position)
            WHERE e.wt_residue <> e.mut_residue
            """
        )

        checks = [
            (
                "unique_sequence_ids",
                con.execute(
                    "SELECT COUNT(*) = COUNT(DISTINCT sequence_id) FROM source.sequences"
                ).fetchone()[0],
            ),
            (
                "twenty_scores_per_scored_site",
                con.execute(
                    """
                    SELECT COUNT(*) = 0 FROM (
                        SELECT sequence_id, position, COUNT(*) AS n
                        FROM source.llr_scores
                        GROUP BY 1, 2
                        HAVING n <> 20
                    )
                    """
                ).fetchone()[0],
            ),
            (
                "one_feature_row_per_true_substitution",
                con.execute(
                    """
                    SELECT COUNT(*) = COUNT(DISTINCT
                        concat_ws(':', sequence_id, CAST(position AS VARCHAR), mut_residue))
                    FROM features.mutation_features
                    """
                ).fetchone()[0],
            ),
        ]

        audit_df = pd.DataFrame(checks, columns=["check_id", "passed"])
        con.register("audit_df", audit_df)
        con.execute("CREATE TABLE audit.checks AS SELECT * FROM audit_df")
        con.unregister("audit_df")

        manifest = pd.DataFrame(
            [
                ("sequences", paths.sequences.name, sha256_file(paths.sequences)),
                ("llr_scores", paths.llr_scores.name, sha256_file(paths.llr_scores)),
                ("annotations", paths.annotations.name, sha256_file(paths.annotations)),
            ],
            columns=["input_role", "file_name", "sha256"],
        )
        con.register("manifest_df", manifest)
        con.execute("CREATE TABLE audit.input_manifest AS SELECT * FROM manifest_df")
        con.unregister("manifest_df")

        summary = {
            "alignment_length": alignment_length,
            "sequences": int(len(sequences)),
            "llr_rows": int(len(llrs)),
            "mutation_feature_rows": int(
                con.execute("SELECT COUNT(*) FROM features.mutation_features").fetchone()[0]
            ),
            "audit_passed": bool(audit_df["passed"].all()),
        }
        con.execute("CHECKPOINT")
        return summary
    finally:
        con.close()


def main() -> int:
    args = parse_args()
    paths = Paths(
        sequences=args.sequences,
        llr_scores=args.llr_scores,
        annotations=args.annotations,
        database=args.database,
    )
    summary = build_database(paths, replace=args.replace)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
