"""Course CSV schema, canonical coordinates and reproducible data partitions."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

TARGET = "tas_FINAL"
FEATURES = ["lat", "lon", "tas_2015", "pr_2015", "pr90_2015", "dtr_2015"] + [
    f"{gas}_{year}" for year in range(2015, 2051) for gas in ["CO2", "SO2", "CH4", "BC"]
]
COLUMNS = ["scenario", *FEATURES, TARGET]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(frame):
    if set(frame.columns) != set(COLUMNS) or len(frame.columns) != len(COLUMNS):
        missing, extra = set(COLUMNS) - set(frame.columns), set(frame.columns) - set(COLUMNS)
        raise ValueError(f"CSV schema mismatch: missing={sorted(missing)}, extra={sorted(extra)}")
    frame = frame[COLUMNS].copy()
    if len(frame) < 2 or frame.isna().any().any():
        raise ValueError("Need at least two rows with no missing values")
    if not frame.scenario.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise ValueError("Scenario names must be nonempty strings")
    frame["scenario"] = frame.scenario.str.strip().str.lower()
    try:
        frame[FEATURES + [TARGET]] = frame[FEATURES + [TARGET]].apply(pd.to_numeric)
    except (ValueError, TypeError) as exc:
        raise ValueError("Features and target must be numeric") from exc
    if not np.isfinite(frame[FEATURES + [TARGET]].to_numpy(dtype=float)).all():
        raise ValueError("Features and target must be finite")
    if not frame.lat.between(-90, 90).all() or not frame.lon.between(-180, 360).all():
        raise ValueError("Coordinates outside latitude [-90,90] or longitude [-180,360]")
    # Accept the legacy 0..360 convention while grouping equivalent longitudes together.
    frame["lon"] = (frame.lon + 180) % 360 - 180
    keys = pd.DataFrame({"scenario": frame.scenario, "location": location_keys(frame)})
    if keys.duplicated().any():
        raise ValueError("Duplicate scenario/location rows are ambiguous; resolve them upstream")
    return frame


def load_csv(path):
    return validate(pd.read_csv(path))


def location_keys(frame):
    return list(zip(frame.lat.round(6), (((frame.lon + 180) % 360) - 180).round(6)))


def prepare(csv, output, seed=42, split="location", evidence="course"):
    if split not in {"location", "row"} or evidence not in {"course", "synthetic"}:
        raise ValueError("Invalid split or evidence type")
    frame = load_csv(csv)
    groups = pd.factorize(pd.Series(location_keys(frame)))[0]
    ids = np.arange(len(frame))
    if split == "location":
        if len(np.unique(groups)) < 10:
            raise ValueError("Need at least ten unique locations for grouped 60/20/20 splits")
        development, holdout = next(
            GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed).split(
                frame, groups=groups
            )
        )
        train_rel, val_rel = next(
            GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=seed).split(
                development, groups=groups[development]
            )
        )
        train, validation = development[train_rel], development[val_rel]
    else:
        development, holdout = train_test_split(
            ids, test_size=0.2, random_state=seed, stratify=frame.scenario
        )
        train, validation = train_test_split(
            development,
            test_size=0.25,
            random_state=seed,
            stratify=frame.scenario.iloc[development],
        )
    parts = {"train": train, "validation": validation, "holdout": holdout}
    for name, rows in parts.items():
        if len(rows) < 2 or set(frame.scenario.iloc[rows]) != set(frame.scenario):
            raise ValueError(
                f"{name} must have at least two rows and include every development scenario"
            )
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "source_sha256": file_hash(csv),
        "seed": seed,
        "split": split,
        "evidence": evidence,
        "features": FEATURES,
        "target": TARGET,
        "locations": [list(x) for x in sorted(set(location_keys(frame)))],
        "scenarios": sorted(frame.scenario.unique()),
        "splits": {},
    }
    for name, rows in parts.items():
        path = out / f"{name}.csv"
        frame.iloc[rows].to_csv(path, index=False)
        manifest["splits"][name] = {
            "rows": len(rows),
            "source_row_indices": rows.tolist(),
            "locations": len(set(location_keys(frame.iloc[rows]))),
            "scenario_counts": {
                str(k): int(v) for k, v in frame.scenario.iloc[rows].value_counts().items()
            },
            "sha256": file_hash(path),
        }
    write_json(out / "manifest.json", manifest)
    return manifest


def load_split(directory, name):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    path = directory / f"{name}.csv"
    if file_hash(path) != manifest["splits"][name]["sha256"]:
        raise ValueError(f"{name} CSV has changed since preparation")
    return load_csv(path), manifest


def check_external(frame, manifest, regime):
    locations = set(location_keys(frame))
    development = {tuple(x) for x in manifest["locations"]}
    scenarios = set(frame.scenario)
    known = set(manifest["scenarios"])
    if regime == "spatial":
        if locations & development:
            raise ValueError("Spatial holdout overlaps development locations")
        if not scenarios <= known:
            raise ValueError("Spatial holdout must use known scenarios")
    elif regime == "scenario":
        if scenarios & known:
            raise ValueError("Scenario holdout contains a development scenario")
        if not locations <= development:
            raise ValueError("Scenario holdout includes new locations; this would mix two shifts")
    else:
        raise ValueError("Unknown external holdout regime")
