import numpy as np
import pandas as pd
import pytest

from climate_study.data import (
    FEATURES,
    TARGET,
    check_external,
    load_split,
    location_keys,
    prepare,
    validate,
)
from climate_study.demo import make_frame


@pytest.mark.parametrize("split", ["location", "row"])
def test_splits_deterministic_and_disjoint(source, tmp_path, split):
    path, frame, _ = source
    one = prepare(path, tmp_path / "one", split=split)
    two = prepare(path, tmp_path / "two", split=split)
    assert one == two
    assert TARGET not in FEATURES and "scenario" not in FEATURES and len(FEATURES) == 150
    parts = [load_split(tmp_path / "one", name)[0] for name in ["train", "validation", "holdout"]]
    assert sum(map(len, parts)) == len(frame)
    ids = [
        set(one["splits"][name]["source_row_indices"])
        for name in ["train", "validation", "holdout"]
    ]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    if split == "location":
        locations = [set(location_keys(part)) for part in parts]
        assert not (
            locations[0] & locations[1]
            or locations[0] & locations[2]
            or locations[1] & locations[2]
        )


@pytest.mark.parametrize(
    "change", ["nan", "infinity", "missing", "extra", "duplicate", "bad_lat", "blank_scenario"]
)
def test_invalid_schema_rejected(source, change):
    _, frame, _ = source
    if change == "nan":
        frame.loc[0, "tas_2015"] = np.nan
    elif change == "infinity":
        frame.loc[0, TARGET] = np.inf
    elif change == "missing":
        frame = frame.drop(columns="CO2_2050")
    elif change == "extra":
        frame["future_target_copy"] = frame[TARGET]
    elif change == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif change == "bad_lat":
        frame.loc[0, "lat"] = 100
    else:
        frame.loc[0, "scenario"] = " "
    with pytest.raises(ValueError):
        validate(frame)


def test_longitude_aliases_rejected_as_duplicate(source):
    _, frame, _ = source
    alias = frame.iloc[:1].copy()
    alias["lon"] = 360
    with pytest.raises(ValueError, match="Duplicate"):
        validate(pd.concat([frame, alias]))


def test_changed_prepared_data_rejected(source, tmp_path):
    path, _, _ = source
    prepare(path, tmp_path / "data")
    csv = tmp_path / "data/train.csv"
    csv.write_text(csv.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        load_split(tmp_path / "data", "train")


def test_external_holdout_boundaries(source, tmp_path):
    path, _, locations = source
    manifest = prepare(path, tmp_path / "data")
    spatial = make_frame([(40.0, 50.0), (42.0, 51.0)], ["ssp126"], 2)
    scenario = make_frame(locations, ["ssp245"], 3)
    check_external(spatial, manifest, "spatial")
    check_external(scenario, manifest, "scenario")
    with pytest.raises(ValueError, match="overlaps"):
        check_external(make_frame(locations, ["ssp126"], 1), manifest, "spatial")
    with pytest.raises(ValueError, match="development scenario"):
        check_external(make_frame(locations, ["ssp126"], 1), manifest, "scenario")
    with pytest.raises(ValueError, match="mix two shifts"):
        check_external(
            make_frame([(40.0, 50.0), (42.0, 51.0)], ["ssp245"], 1), manifest, "scenario"
        )
