# Group-aware candidate ranking

A small modelling example showing how I avoid group leakage when evaluating candidate-level biological models.

Candidates from the same biological position are kept together with `GroupShuffleSplit`. The held-out set contains positions that were not present during model fitting. A `HistGradientBoostingClassifier` is trained with balanced sample weights, then evaluated with average precision, AUROC and within-position candidate ranks.

## Run

```bash
python -m pip install -r requirements.txt
python make_synthetic_data.py --output example_data/candidates.csv
python train_grouped_candidate_ranker.py \
  --input example_data/candidates.csv \
  --output-dir outputs \
  --seed 42
```
