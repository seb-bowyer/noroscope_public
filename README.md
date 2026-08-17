# Computational biology code samples

A small collection of representative Python code from my computational biology / machine-learning work.

These examples are derived from research workflows developed during my PhD and have been prepared specifically as **standalone code samples**. They are not a release of the underlying unpublished research project: no unpublished biological datasets, results, publication labels, frozen analysis parameters or complete manuscript pipelines are included here.

The full research codebase will be released separately with the associated preprint.

## Included examples

### 1. `esm2_llr/` — protein language-model inference

A command-line ESM-2 workflow that performs batched masked-token inference and calculates zero-shot amino-acid log-likelihood ratios. It demonstrates PyTorch/Hugging Face inference, FASTA handling, GPU-aware execution, logging, restart-safe output handling and optional residue embeddings.

### 2. `sequence_feature_database/` — reproducible biological data integration

A reduced DuckDB pipeline that validates aligned protein sequences and mutation-score tables, records input checksums, builds derived mutation features with SQL window functions and runs full-table integrity checks. Synthetic inputs are supplied.

### 3. `grouped_model_demo/` — leakage-aware ML evaluation

A generalised candidate-ranking example using position-grouped train/test splitting, class-balanced gradient boosting and both classification and within-position ranking metrics. All data and labels in this example are synthetic.

## Environment

Each example has its own `requirements.txt` and README because the ESM-2 example requires a substantially heavier ML environment than the DuckDB and scikit-learn demonstrations.
