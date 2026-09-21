"""Artificial inputs for exercising the code; never a ClimateBench substitute."""

from pathlib import Path

import numpy as np
import pandas as pd

from .data import COLUMNS, prepare
from .experiment import evaluate, train


def make_frame(locations, scenarios, seed):
    rng = np.random.default_rng(seed)
    rows = []
    forcing = {"ssp126": 1.0, "ssp370": 3.0, "ssp585": 5.0, "ssp245": 2.0}
    for scenario in scenarios:
        for lat, lon in locations:
            strength = forcing[scenario]
            baseline = 0.8 + 0.008 * lat + 0.003 * lon
            row = {
                "scenario": scenario,
                "lat": lat,
                "lon": lon,
                "tas_2015": baseline,
                "pr_2015": lat * 1e-8,
                "pr90_2015": lon * 1e-8,
                "dtr_2015": 0.1 * baseline,
            }
            for year in range(2015, 2051):
                for gas, scale in [("CO2", 1000), ("SO2", 1e-8), ("CH4", 0.1), ("BC", 1e-9)]:
                    row[f"{gas}_{year}"] = scale * (1 + strength * (year - 2015) / 35)
            row["tas_FINAL"] = baseline + 0.2 * strength + 0.0002 * lon**2 + rng.normal(0, 0.03)
            rows.append(row)
    return pd.DataFrame(rows)[COLUMNS]


def run_demo(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    locations = [
        (float(lat), float(lon))
        for lat in np.linspace(-20, 40, 10)
        for lon in np.linspace(0, 30, 8)
    ]
    spatial = [
        (float(lat), float(lon))
        for lat in np.linspace(-10, 30, 5)
        for lon in np.linspace(40, 60, 5)
    ]
    known = ["ssp126", "ssp370", "ssp585"]
    for name, coords, scenarios, seed in [
        ("development", locations, known, 1),
        ("spatial", spatial, known, 2),
        ("scenario", locations, ["ssp245"], 3),
    ]:
        make_frame(coords, scenarios, seed).to_csv(output / f"{name}.csv", index=False)
    prepare(output / "development.csv", output / "prepared", evidence="synthetic")
    train(output / "prepared", output / "experiment", seeds=[42, 43, 44], trees=40)
    return evaluate(
        output / "prepared", output / "experiment", output / "spatial.csv", output / "scenario.csv"
    )
