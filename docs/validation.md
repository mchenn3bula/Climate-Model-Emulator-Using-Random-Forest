# Verification record — 21 September 2026

Local verification used Python 3.12.14 on Windows, scikit-learn 1.9.1, pandas 3.0.6, NumPy 2.5.3 and Matplotlib 3.11.2. Exact dependencies are pinned in `requirements-test.txt`.

- **17 tests passed** in 4.93 seconds of pytest execution.
- Ruff lint passed; package and tests were formatted with Ruff.
- `pip check` reported no broken requirements.
- The installed CLI completed the synthetic demonstration with three seeds, four model families and three evaluation regimes.
- Both generated plots were visually inspected.

Tests cover deterministic grouped/row splits, disjoint partition membership, strict schema and coordinate validation, target exclusion, longitude aliases, changed-file detection, spatial/scenario holdout membership, train-only scaler fitting, undefined/negative R², aggregate seed statistics, saved-model evaluation and overwrite protection. Both split modes run end to end while the holdout file is absent during training.

Because the account's default pytest temporary directory is inaccessible, the local test command used a fresh dedicated directory after creating `runs/`:

```bash
python -m pytest -q --tb=short --basetemp=runs/pytest-local-01
```

In a normal environment, `python -m pytest -q` is sufficient. Pytest manages the contents of an explicitly supplied temporary directory.

## Synthetic demonstration

The demo generates 240 development rows at 80 locations across three scenarios, 75 spatial rows at 25 new locations, and 80 rows for one new scenario. The default grouped split retains 48 training locations, so scenario-only evaluation retains 48 of the 80 scenario rows. Forests use 40 trees and seeds 42, 43 and 44.

The [saved report](demo/report.md) and plots are examples of the pipeline's output. They are labeled synthetic and do not establish climate performance. Demo metrics follow from an artificial generating function, not observations or NorESM2 simulations.

## Still unverified

The three original course CSVs are absent. No full climate-data experiment, historical-score reproduction, area-weighted evaluation or improved real-data accuracy is claimed. GitHub Actions is configured to rerun lint and tests on Ubuntu; check the Actions page for its current status.
