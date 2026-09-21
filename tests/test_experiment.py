import json

import numpy as np
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from climate_study.data import FEATURES, load_split, prepare
from climate_study.demo import make_frame
from climate_study.experiment import aggregate, evaluate, metrics, select, train


def test_r2_undefined_not_reported_as_zero():
    assert metrics([1, 1], [1, 1])["r2"] is None
    assert metrics([1], [2])["r2"] is None
    assert metrics([1, 2, 3], [1, 2, 3])["r2"] == 1
    assert metrics([1, 2, 3], [0, 0, 0])["r2"] < 0


def test_scaler_fitted_only_on_training(source, tmp_path):
    path, _, _ = source
    prepare(path, tmp_path / "data")
    training, _ = load_split(tmp_path / "data", "train")
    validation, _ = load_split(tmp_path / "data", "validation")
    validation[FEATURES] += 1000
    model, _, _ = select([({}, make_pipeline(StandardScaler(), Ridge()))], training, validation)
    np.testing.assert_allclose(model[0].mean_, training[FEATURES].mean())


@pytest.mark.parametrize("split", ["location", "row"])
def test_end_to_end_without_test_access_during_training(source, tmp_path, split):
    path, _, locations = source
    data, run = tmp_path / "data", tmp_path / "run"
    prepare(path, data, split=split, evidence="synthetic")
    holdout = data / "holdout.csv"
    saved = holdout.read_bytes()
    holdout.unlink()
    train(data, run, seeds=[42, 43], trees=3)
    assert not (run / "evaluation.json").exists()
    holdout.write_bytes(saved)
    spatial_csv, scenario_csv = tmp_path / "spatial.csv", tmp_path / "scenario.csv"
    make_frame([(40.0, 50.0), (42.0, 51.0)], ["ssp126", "ssp370", "ssp585"], 2).to_csv(
        spatial_csv, index=False
    )
    make_frame(locations, ["ssp245"], 3).to_csv(scenario_csv, index=False)
    result = evaluate(data, run, spatial_csv, scenario_csv)
    assert len(result["records"]) == 24 and len(result["summary"]) == 12
    assert result["evidence"] == "synthetic"
    assert result["scenario_rows"]["evaluated_at_training_locations"] > 0
    if split == "location":
        assert result["scenario_rows"]["excluded_untrained_locations"] > 0
    for name in ["rmse_comparison.png", "residual_locations.png"]:
        assert (run / name).read_bytes().startswith(b"\x89PNG")
    assert "SYNTHETIC" in (run / "report.md").read_text(encoding="utf-8")
    assert json.loads((run / "evaluation.json").read_text())["records"] == result["records"]
    with pytest.raises(ValueError, match="already exists"):
        evaluate(data, run, spatial_csv, scenario_csv)
    with pytest.raises(FileExistsError):
        train(data, run, trees=3)


def test_seed_variability_not_confidence_interval():
    rows = [
        {"family": "forest", "regime": "spatial", "r2": None, "mae": x, "rmse": x, "bias": 0.0}
        for x in [1.0, 3.0]
    ]
    result = aggregate(rows)[0]
    assert result["rmse_mean"] == 2
    assert result["rmse_std"] == pytest.approx(np.sqrt(2))
    assert result["r2_mean"] is None
