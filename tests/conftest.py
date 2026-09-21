import pytest

from climate_study.demo import make_frame


@pytest.fixture
def source(tmp_path):
    locations = [(float(lat), float(lon)) for lat in range(0, 20, 4) for lon in range(0, 20, 4)]
    frame = make_frame(locations, ["ssp126", "ssp370", "ssp585"], 1)
    path = tmp_path / "source.csv"
    frame.to_csv(path, index=False)
    return path, frame, locations
