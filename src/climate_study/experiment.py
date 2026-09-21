"""Validation-only model selection and independent distribution-shift evaluation."""

import importlib.metadata
import platform
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import (
    FEATURES,
    TARGET,
    check_external,
    file_hash,
    load_csv,
    load_split,
    location_keys,
    read_json,
    write_json,
)


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if not np.isfinite(predicted).all():
        raise ValueError("Nonfinite predictions")
    # R² is undefined for constant targets or a singleton; never silently convert it to 0 or 1.
    r2 = None if len(actual) < 2 or np.ptp(actual) == 0 else float(r2_score(actual, predicted))
    return {
        "n": len(actual),
        "r2": r2,
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "bias": float(np.mean(predicted - actual)),
    }


def candidates(seed, trees):
    forest = lambda depth, leaf: RandomForestRegressor(
        n_estimators=trees, max_depth=depth, min_samples_leaf=leaf, random_state=seed, n_jobs=1
    )
    grid = [(None, 1), (None, 3), (12, 1), (12, 3)]
    return {
        "mean": [({}, DummyRegressor(strategy="mean"))],
        "ridge": [
            ({"alpha": a}, make_pipeline(StandardScaler(), Ridge(alpha=a)))
            for a in [0.1, 1.0, 10.0]
        ],
        "forest": [({"max_depth": d, "min_samples_leaf": l}, forest(d, l)) for d, l in grid],
        "scaled_forest": [
            ({"max_depth": d, "min_samples_leaf": l}, make_pipeline(StandardScaler(), forest(d, l)))
            for d, l in grid
        ],
    }


def select(candidates_list, train, validation):
    history, best, best_score, best_params = [], None, float("inf"), None
    for params, model in candidates_list:
        model.fit(train[FEATURES], train[TARGET])
        score = metrics(validation[TARGET], model.predict(validation[FEATURES]))
        history.append({"parameters": params, "validation": score})
        if score["rmse"] < best_score:
            best, best_score, best_params = model, score["rmse"], params
    return best, best_params, history


def runtime_info():
    checkout = Path(__file__).resolve().parents[2]
    revision, dirty = None, None
    if (checkout / ".git").exists():
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=checkout, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=checkout,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": revision,
        "git_dirty": dirty,
        "packages": {
            p: importlib.metadata.version(p)
            for p in ["numpy", "pandas", "scikit-learn", "matplotlib", "joblib"]
        },
    }


def train(data, output, seeds=(42,), trees=100):
    if trees < 1 or not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide positive tree count and distinct seeds")
    train_frame, manifest = load_split(data, "train")
    val_frame, _ = load_split(data, "validation")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "seeds": list(seeds),
        "trees": trees,
        "evidence": manifest["evidence"],
        "manifest_sha256": file_hash(Path(data) / "manifest.json"),
        "runtime": runtime_info(),
    }
    write_json(output / "config.json", config)
    write_json(output / "data_manifest.json", manifest)
    history, models = {}, {}
    for seed in seeds:
        for family, grid in candidates(seed, trees).items():
            key = f"{family}_seed{seed}"
            model, parameters, trials = select(grid, train_frame, val_frame)
            models[key] = model
            history[key] = {
                "family": family,
                "seed": seed,
                "selected": parameters,
                "trials": trials,
            }
            print(f"Selected {key} on validation RMSE", flush=True)
    joblib.dump(models, output / "models.joblib")
    write_json(output / "validation_history.json", history)
    write_json(output / "complete.json", {"status": "training_complete"})
    return history


def grouped_errors(frame, predictions):
    report = {}
    groups = {
        "scenario": frame.scenario,
        "latitude_band": pd.Series(
            np.floor(frame.lat.to_numpy() / 15).astype(int) * 15, index=frame.index
        ),
    }
    for kind, labels in groups.items():
        report[kind] = {}
        for label in sorted(labels.unique()):
            mask = np.asarray(labels == label)
            report[kind][str(label)] = metrics(frame[TARGET].to_numpy()[mask], predictions[mask])
    return report


def aggregate(records):
    result = []
    for (family, regime), part in pd.DataFrame(records).groupby(["family", "regime"], sort=True):
        entry = {"family": family, "regime": regime, "runs": len(part)}
        for name in ["r2", "mae", "rmse", "bias"]:
            values = pd.to_numeric(part[name], errors="coerce").dropna()
            entry[name + "_mean"] = float(values.mean()) if len(values) else None
            entry[name + "_std"] = float(values.std(ddof=1)) if len(values) > 1 else None
        result.append(entry)
    return result


def evaluate(data, run, spatial_csv, scenario_csv):
    run = Path(run)
    if not (run / "complete.json").exists():
        raise ValueError("Training did not complete")
    if (run / "evaluation.json").exists():
        raise ValueError("Evaluation already exists; preserve the held-out result")
    config = read_json(run / "config.json")
    if file_hash(Path(data) / "manifest.json") != config["manifest_sha256"]:
        raise ValueError("Prepared data manifest differs from training")
    internal, manifest = load_split(data, "holdout")
    spatial, scenario = load_csv(spatial_csv), load_csv(scenario_csv)
    check_external(spatial, manifest, "spatial")
    check_external(scenario, manifest, "scenario")
    training, _ = load_split(data, "train")
    training_locations = set(location_keys(training))
    scenario_total = len(scenario)
    # Isolate the scenario shift: new-scenario rows at untrained locations combine two shifts.
    scenario = scenario.loc[[loc in training_locations for loc in location_keys(scenario)]].copy()
    if len(scenario) < 2:
        raise ValueError("Scenario-only evaluation needs at least two rows at training locations")
    internal_name = "location_holdout" if manifest["split"] == "location" else "row_holdout"
    frames = {internal_name: internal, "spatial": spatial, "scenario": scenario}
    models = joblib.load(run / "models.joblib")
    history = read_json(run / "validation_history.json")
    records, subgroups, predictions = [], {}, []
    for key, model in models.items():
        subgroups[key] = {}
        for regime, frame in frames.items():
            pred = np.asarray(model.predict(frame[FEATURES]))
            records.append(
                {
                    "model": key,
                    "family": history[key]["family"],
                    "seed": history[key]["seed"],
                    "regime": regime,
                    **metrics(frame[TARGET], pred),
                }
            )
            subgroups[key][regime] = grouped_errors(frame, pred)
            part = frame[["scenario", "lat", "lon", TARGET]].copy()
            part["prediction"], part["model"], part["regime"] = pred, key, regime
            predictions.append(part)
    predictions = pd.concat(predictions, ignore_index=True)
    summary = aggregate(records)
    result = {
        "evidence": config["evidence"],
        "split": manifest["split"],
        "records": records,
        "summary": summary,
        "subgroups": subgroups,
        "runtime": runtime_info(),
        "scenario_rows": {
            "input": scenario_total,
            "evaluated_at_training_locations": len(scenario),
            "excluded_untrained_locations": scenario_total - len(scenario),
        },
        "external_sha256": {"spatial": file_hash(spatial_csv), "scenario": file_hash(scenario_csv)},
    }
    predictions.to_csv(run / "predictions.csv", index=False)
    pd.DataFrame(summary).to_csv(run / "summary.csv", index=False)
    from .plots import plot_results

    plot_results(predictions, summary, run, config["evidence"], config["seeds"][0])
    write_report(result, run / "report.md")
    write_json(run / "evaluation.json", result)
    return result


def write_report(result, path):
    label = (
        "SYNTHETIC DEMO — NOT CLIMATE PERFORMANCE"
        if result["evidence"] == "synthetic"
        else "Course-data evaluation"
    )
    lines = [
        f"# {label}",
        "",
        f"Internal split: `{result['split']}`. Model selection used validation only.",
        "",
        "| Model | Evaluation | Runs | R² (mean) | RMSE (mean) | RMSE (std) | MAE (mean) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    fmt = lambda value: "undefined" if value is None else f"{value:.4f}"
    for row in result["summary"]:
        lines.append(
            f"| {row['family']} | {row['regime']} | {row['runs']} | "
            + " | ".join(fmt(row[x]) for x in ["r2_mean", "rmse_mean", "rmse_std", "mae_mean"])
            + " |"
        )
    lines += [
        "",
        "Standard deviations describe training-seed variation on one fixed split; they are not confidence intervals.",
        "Rows are weighted equally. These are not area-weighted global climate metrics.",
        "",
        f"Scenario-only evaluation retained {result['scenario_rows']['evaluated_at_training_locations']} of {result['scenario_rows']['input']} rows at training locations to avoid mixing spatial and scenario shifts.",
        "",
        "![RMSE comparison](rmse_comparison.png)",
        "",
        "![Residual locations](residual_locations.png)",
        "",
        "Subgroup metrics by scenario and 15-degree latitude band are recorded in evaluation.json.",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
