# Evaluation decisions

## Task and input contract

The historical assignment predicts `tas_FINAL` for 2050 from coordinates, four 2015 climate variables and 36 annual values for each of four forcing/emissions variables. The assignment describes temperature relative to a pre-industrial reference. The code preserves this target name without inferring physical units from a CSV. Synthetic targets are artificial values.

The strict predictor allowlist prevents accidental inclusion of the target, scenario labels, a saved DataFrame index or future output columns. Every input CSV must have the same schema. Longitude aliases are canonicalized; coordinate comparisons use six decimal places. Duplicate scenario/location rows are rejected rather than silently assigning ensemble-member rows to different splits. A dataset with multiple members needs an explicit member-aware extension.

## Split interpretation

The default location split keeps all scenarios at a coordinate together. It tests interpolation to unseen coordinates within the development region; nearby points may still be spatially correlated. It is not a spatial-block or leave-region-out protocol. The optional row split is easier because a location can be represented under another scenario during training.

Training, validation and holdout nominal fractions are 60/20/20. All partitions must contain every development scenario. The split is frozen in a manifest independently of training seeds. Models use the same split and feature order.

The external spatial CSV must contain known scenarios and coordinates absent from the complete development CSV. The external scenario CSV must contain new scenarios and development coordinates. Scenario scoring is restricted to coordinates in the training partition; other rows are counted as excluded. This prevents reporting a combination of unseen location and unseen scenario as a pure scenario shift.

## Fitting and selection

Mean, Ridge, raw forest and scaled forest are prespecified comparison families. Candidate fits use training data only; pipelines keep StandardScaler fitting within that boundary. Validation RMSE selects one candidate per family/seed, with first-best tie breaking. Selected models are not refitted on validation. Test files are unavailable to the training command, which is verified by a regression test that temporarily removes the internal holdout file.

Random Forest seeds vary model training, not partition membership. Mean and Ridge do not change with those seeds. Raw and scaled forest results compare selected pipelines with matched candidate grids, not one fixed set of selected hyperparameters. Finite precision and very small feature magnitudes can also affect tree splits; scaling results must be measured rather than assumed.

## Metrics and reports

R² measures performance relative to the evaluation target's mean. A negative score is possible; a constant target or singleton has undefined R² and is serialized as null. The explicit DummyRegressor baseline predicts the training mean, which is a separate comparison under distribution shift.

RMSE, MAE and signed bias are in target units. Reports weight rows equally, with subgroup metrics by scenario and 15-degree latitude band. These are not area-weighted global climate diagnostics or the upstream benchmark's NRMSE. A location plot averages residuals across scenarios at that location and shows the first prespecified forest seed; the tabular report retains every seed.

Aggregated standard deviations use sample standard deviation across training seeds. They quantify training variability on one split, not sampling uncertainty or confidence intervals. Do not tune using the final reports. Artifact overwrite protection cannot prevent deliberate holdout reuse in a new run.

## Data limits

The course CSVs and their extraction script are missing. Upstream NetCDF data cannot be substituted silently because region boundaries, aggregation choices, ensemble selection and anomaly preprocessing affect the task. The executable demo uses an explicitly artificial generating function and is not a physics simulator or replacement climate dataset.

## References

- [ClimateBench source and data links](https://github.com/duncanwp/ClimateBench)
- [GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html)
- [R² definition and constant-target behavior](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html)
