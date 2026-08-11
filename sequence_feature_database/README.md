# Audited protein feature database

A compact DuckDB example showing how I structure reproducible biological-data integration workflows.

The script validates aligned protein sequences and masked-language-model mutation scores, records SHA-256 input identities, keeps source data separate from derived features, and builds a queryable mutation-feature view using SQL window functions. It also runs simple full-table invariants before the database is considered usable.

This is **not** the database implementation used in an unpublished manuscript. It is a deliberately reduced example derived from the same engineering principles, using synthetic inputs only.

## Run

```bash
python -m pip install -r requirements.txt
python build_feature_database.py \
  --sequences example_data/sequences.csv \
  --llr-scores example_data/llr_scores.csv \
  --annotations example_data/annotations.csv \
  --database example_data/demo.duckdb \
  --replace
```

Inspect the output with DuckDB, for example:

```sql
SELECT * FROM features.mutation_features LIMIT 10;
SELECT * FROM audit.checks;
SELECT * FROM audit.input_manifest;
```
