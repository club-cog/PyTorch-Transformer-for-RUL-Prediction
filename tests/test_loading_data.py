import numpy as np
import pandas as pd
import pytest

from loading_data import loading_FD001

INDEX_NAMES = ["unit_nr", "time_cycles"]
SETTING_NAMES = ["setting_1", "setting_2", "setting_3"]
SENSOR_NAMES = ["s_{}".format(i) for i in range(1, 22)]
DROP_SENSORS = ["s_1", "s_5", "s_6", "s_10", "s_16", "s_18", "s_19"]
KEPT_SENSORS = [s for s in SENSOR_NAMES if s not in DROP_SENSORS]


def _make_rows(rng, unit_nr, n_cycles):
    rows = []
    for cycle in range(1, n_cycles + 1):
        row = [unit_nr, cycle]
        row += list(rng.uniform(-1.0, 1.0, size=len(SETTING_NAMES)))
        row += list(rng.uniform(0.0, 100.0, size=len(SENSOR_NAMES)))
        rows.append(row)
    return rows


@pytest.fixture
def synthetic_cmapss(tmp_path, monkeypatch):
    """Create small synthetic FD001 fixture files and chdir into them."""
    rng = np.random.default_rng(42)
    data_dir = tmp_path / "CMAPSSData"
    data_dir.mkdir()

    train_units = {1: 10, 2: 8, 3: 12}
    test_units = {1: 5, 2: 7}

    def write_file(path, units):
        rows = []
        for unit_nr, n_cycles in units.items():
            rows.extend(_make_rows(rng, unit_nr, n_cycles))
        pd.DataFrame(rows).to_csv(path, sep=" ", header=False, index=False)

    write_file(data_dir / "train_FD001.txt", train_units)
    write_file(data_dir / "test_FD001.txt", test_units)
    (data_dir / "RUL_FD001.txt").write_text("20\n30\n")

    monkeypatch.chdir(tmp_path)
    return train_units, test_units


def test_returns_groupby_objects_and_dataframe(synthetic_cmapss):
    group, y_test, group_test = loading_FD001()
    assert isinstance(group, pd.core.groupby.generic.DataFrameGroupBy)
    assert isinstance(group_test, pd.core.groupby.generic.DataFrameGroupBy)
    assert isinstance(y_test, pd.DataFrame)


def test_group_counts_match_units(synthetic_cmapss):
    train_units, test_units = synthetic_cmapss
    group, _, group_test = loading_FD001()
    assert len(group) == len(train_units)
    assert len(group_test) == len(test_units)
    for unit_nr, n_cycles in train_units.items():
        assert len(group.get_group(unit_nr)) == n_cycles
    for unit_nr, n_cycles in test_units.items():
        assert len(group_test.get_group(unit_nr)) == n_cycles


def test_dropped_sensors_and_settings_absent(synthetic_cmapss):
    group, _, group_test = loading_FD001()
    train_df = group.get_group(1)
    test_df = group_test.get_group(1)
    for col in SETTING_NAMES + DROP_SENSORS:
        assert col not in train_df.columns
        assert col not in test_df.columns
    for col in KEPT_SENSORS:
        assert col in train_df.columns
        assert col in test_df.columns


def test_sensor_data_min_max_normalized(synthetic_cmapss):
    group, _, group_test = loading_FD001()
    train_df = pd.concat([group.get_group(u) for u in group.groups])
    test_df = pd.concat([group_test.get_group(u) for u in group_test.groups])
    for df in (train_df, test_df):
        sensors = df[KEPT_SENSORS]
        assert sensors.min().min() >= 0.0
        assert sensors.max().max() <= 1.0
        assert np.allclose(sensors.min(), 0.0)
        assert np.allclose(sensors.max(), 1.0)


def test_train_rul_clipped_and_decreasing(synthetic_cmapss):
    train_units, _ = synthetic_cmapss
    group, _, _ = loading_FD001()
    for unit_nr, n_cycles in train_units.items():
        unit_df = group.get_group(unit_nr).sort_values("time_cycles")
        rul = unit_df["RUL"].tolist()
        assert rul[-1] == 0
        assert rul == sorted(rul, reverse=True)
        assert max(rul) <= 125
        assert rul == [min(n_cycles - c, 125) for c in unit_df["time_cycles"]]


def test_y_test_values(synthetic_cmapss):
    _, y_test, _ = loading_FD001()
    assert list(y_test.columns) == ["RUL"]
    assert y_test["RUL"].tolist() == [20, 30]


def test_test_set_has_no_rul_column(synthetic_cmapss):
    _, _, group_test = loading_FD001()
    assert "RUL" not in group_test.get_group(1).columns


def test_loading_real_cmapss_data(repo_root, monkeypatch):
    """Smoke test against the real CMAPSSData files shipped in the repo."""
    monkeypatch.chdir(repo_root)
    group, y_test, group_test = loading_FD001()
    assert len(group) == 100
    assert len(group_test) == 100
    assert len(y_test) == 100
    train_df = group.get_group(1)
    assert train_df["RUL"].max() <= 125
    assert train_df["RUL"].iloc[-1] == 0
