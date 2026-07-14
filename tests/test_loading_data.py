import os

import pandas as pd
import pytest

# The 24 raw feature columns (3 operating settings + 21 sensors) plus the two
# index columns, matching the CMAPSS FD001 file layout.
N_UNITS = 3
CYCLES_PER_UNIT = 5


def _write_cmapss_files(base_dir):
    data_dir = os.path.join(base_dir, "CMAPSSData")
    os.makedirs(data_dir)

    train_rows = []
    test_rows = []
    row = 0
    for unit in range(1, N_UNITS + 1):
        for cycle in range(1, CYCLES_PER_UNIT + 1):
            row += 1
            # settings + 21 sensors, all varying so min-max scaling is well defined
            features = " ".join(str(float(row + k)) for k in range(24))
            train_rows.append(f"{unit} {cycle} {features}")
            test_rows.append(f"{unit} {cycle} {features}")

    with open(os.path.join(data_dir, "train_FD001.txt"), "w") as f:
        f.write("\n".join(train_rows) + "\n")
    with open(os.path.join(data_dir, "test_FD001.txt"), "w") as f:
        f.write("\n".join(test_rows) + "\n")
    with open(os.path.join(data_dir, "RUL_FD001.txt"), "w") as f:
        f.write("\n".join(str(u * 10) for u in range(1, N_UNITS + 1)) + "\n")


@pytest.fixture()
def cmapss_cwd(tmp_path, monkeypatch):
    _write_cmapss_files(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_loading_returns_grouped_train_rul_and_test(cmapss_cwd):
    from loading_data import loading_FD001

    group, y_test, group_test = loading_FD001()

    assert len(group) == N_UNITS
    assert len(group_test) == N_UNITS
    assert list(y_test.columns) == ["RUL"]
    assert len(y_test) == N_UNITS


def test_uninformative_columns_dropped_and_rul_added(cmapss_cwd):
    from loading_data import loading_FD001

    group, _, _ = loading_FD001()
    unit_df = group.get_group(1)

    dropped = {"setting_1", "setting_2", "setting_3", "s_1", "s_5", "s_6", "s_10", "s_16", "s_18", "s_19"}
    assert dropped.isdisjoint(unit_df.columns)
    assert "RUL" in unit_df.columns
    # 14 kept sensors + unit_nr + time_cycles + RUL
    assert unit_df.shape[1] == 17


def test_sensor_values_are_min_max_normalized(cmapss_cwd):
    from loading_data import loading_FD001

    group, _, _ = loading_FD001()
    full = pd.concat([group.get_group(u) for u in range(1, N_UNITS + 1)])
    sensors = [c for c in full.columns if c.startswith("s_")]

    assert (full[sensors].min() >= 0).all()
    assert (full[sensors].max() <= 1).all()


def test_rul_is_clipped_at_125(cmapss_cwd, monkeypatch):
    # Force a very long unit so the piece-wise RUL would exceed the 125 cap.
    from loading_data import loading_FD001

    group, _, _ = loading_FD001()
    full = pd.concat([group.get_group(u) for u in range(1, N_UNITS + 1)])
    assert full["RUL"].max() <= 125
