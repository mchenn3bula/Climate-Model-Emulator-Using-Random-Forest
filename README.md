# Climate model emulation under distribution shift

How much does a model's apparent accuracy change when it encounters new locations or a new emissions scenario? This project studies that question with **Random Forest regression**, a **Ridge baseline**, and a **training-mean baseline** using a course-prepared tabular dataset derived from ClimateBench.

The original notebook is now accompanied by a runnable Python package with fixed splits, validation-only model selection, repeated-seed evaluation, geographic checks and error plots.

**Status:** the full pipeline runs locally on a clearly labeled synthetic demonstration. The original course CSVs are absent, so the notebook's historical climate scores have not been reproduced by the new code.

## What the project demonstrates

- **Evaluation design:** separate internal holdout, out-of-region and unseen-scenario evaluations.
- **Leakage prevention:** group coordinates across scenarios; fit scalers and models only on training data; exclude target and scenario labels from model inputs.
- **Baselines:** compare mean, Ridge, Random Forest and scaled Random Forest under the same prepared split.
- **Reproducibility:** fixed seeds, saved split indices, data hashes, dependency pins, validation histories and model artifacts.
- **Error analysis:** R², RMSE, MAE and signed bias, with breakdowns by scenario and 15-degree latitude band.

## Run it without external data

Python 3.12 is used for verification.

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux instead: source .venv/bin/activate

python -m pip install -r requirements-test.txt
python -m pip install -e . --no-deps
python -m pytest -q
climate-study demo --output runs/synthetic-demo
```

Open `runs/synthetic-demo/experiment/report.md` to see the generated comparison and residual-location plots. The demo uses artificial data, three training seeds and 40 trees per forest. It verifies the workflow; its scores are **not ClimateBench results**.

![Synthetic pipeline demonstration, not climate performance](docs/demo/rmse_comparison.png)

[View the complete synthetic report](docs/demo/report.md).

## Run the course-data experiment

Obtain the matching three course CSVs and place them in `data/`:

```text
climatebench_train_val.csv
climatebench_spatial_test.csv
climatebench_scenario_test.csv
```

The expected schema has 152 columns: `scenario`, `lat`, `lon`, four baseline variables (`tas_2015`, `pr_2015`, `pr90_2015`, `dtr_2015`), annual `CO2`, `SO2`, `CH4`, and `BC` features for 2015–2050, and the target `tas_FINAL`. Only the 150 named numeric predictors enter a model. Missing/extra columns, missing values, nonfinite values and duplicate scenario/location rows are rejected.

### 1. Freeze a split

```bash
climate-study prepare --csv data/climatebench_train_val.csv --output data/prepared --seed 42
```

The default assigns approximately **60% / 20% / 20% of locations** to training, validation and internal holdout, keeping all scenarios at a coordinate together. The manifest records actual row counts and membership. This measures unseen locations within the development region; it is not a claim of geographically distant generalization.

`--split row` provides a stratified row-split comparison. Rows from the same location can then appear in multiple partitions; interpret it as an easier, familiar-region task. It is still a new 60/20/20 protocol, not an exact reproduction of the old 80/20 notebook split. Longitudes are canonicalized to [-180, 180), and grouping uses coordinates rounded to six decimals.

### 2. Fit baselines and select on validation

```bash
climate-study train --data data/prepared --output runs/course --seeds 42 43 44
```

The default forest size is 100 trees. Ridge compares alpha values 0.1, 1 and 10. Forests compare maximum depths unrestricted/12 and minimum leaf sizes 1/3. Each model family and seed selects its candidate by **validation RMSE**. StandardScaler is fitted inside training pipelines. The internal holdout and external CSVs are not opened by the training command.

Ridge and the mean baseline are deterministic; their repeated-seed variation is zero. Raw and scaled forests share seeds and candidate grids, but their selected candidates can differ. Scaling is a comparison condition, not an assumed requirement for trees.

### 3. Evaluate the three regimes

```bash
climate-study evaluate --data data/prepared --run runs/course --spatial-csv data/climatebench_spatial_test.csv --scenario-csv data/climatebench_scenario_test.csv
```

| Regime | Enforced condition |
| --- | --- |
| Internal holdout | Frozen location-grouped or row-based partition from preparation |
| Spatial holdout | No development coordinates in the external CSV; scenarios must be known |
| Scenario holdout | No development scenario in the external CSV; evaluate only rows at training coordinates |

Scenario rows at untrained development coordinates are excluded and counted explicitly, avoiding a mixed spatial-and-scenario shift. Rows outside the development locations cause an error. The external checks establish membership conditions, not a minimum geographic distance.

The run folder receives `report.md`, `evaluation.json`, `summary.csv`, per-row `predictions.csv`, and two PNG plots. Per-seed results and subgroup metrics remain available alongside aggregate means and standard deviations. Standard deviations describe training randomness on **one fixed split**, not confidence intervals. Metrics weight rows equally, not by grid-cell area; they are not global ClimateBench leaderboard metrics.

Run folders and final reports cannot be overwritten. Freeze choices before evaluation and do not tune on holdout reports. Only load Joblib artifacts from your own trusted runs.

## Historical notebook results

These are saved outputs from the [original notebook](Climate%20Model%20Emulator%20Using%20Random%20Forest.ipynb), **not newly reproduced results**:

| Original evaluation | R² |
| --- | ---: |
| Random 20% holdout, unscaled forest | 0.9286 |
| Random 20% holdout, scaled forest | 0.9275 |
| External spatial holdout, scaled forest | 0.5206 |
| External SSP245 holdout, scaled forest | 0.3787 |

These outputs show a generalization gap. They do not establish that scaling improves a forest or that a positive R² is worse than guessing. The new pipeline includes an explicit training-mean baseline and reports absolute errors as well as R². R² is recorded as undefined for constant targets or singleton groups.

## Data provenance and remaining work

The [upstream ClimateBench project](https://github.com/duncanwp/ClimateBench) provides NorESM2 data and links to its [Zenodo archive](https://doi.org/10.5281/zenodo.5196512). Its published benchmark concerns a different task and evaluation period. The course's regional, year-2050 CSV extraction is not included here, so downloading upstream data alone does not recover the exact experiment.

The remaining scientific work is to obtain those three CSVs or document a new extraction, run the real-data protocol, and inspect the resulting subgroup errors. No real climate accuracy improvement is claimed by this refactor.

## Code and verification

```text
src/climate_study/data.py        Schema validation, coordinate grouping, split hashes
src/climate_study/experiment.py  Model selection, metrics and holdout reports
src/climate_study/plots.py       RMSE and residual-location diagnostics
src/climate_study/demo.py        Explicitly artificial demonstration inputs
src/climate_study/cli.py         Prepare / train / evaluate / demo commands
tests/                         Regression and end-to-end tests
docs/                          Protocol, validation record and synthetic artifacts
```

[Evaluation protocol](docs/evaluation.md) · [Verification record](docs/validation.md) · [GitHub Actions](https://github.com/mchenn3bula/Climate-Model-Emulator-Using-Random-Forest/actions)

The original notebook and figures are preserved. Course CSVs, model weights, virtual environments and local run outputs are ignored by Git.
