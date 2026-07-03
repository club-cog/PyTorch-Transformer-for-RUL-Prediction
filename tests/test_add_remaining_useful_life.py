import pandas as pd

from add_remaining_useful_life import add_remaining_useful_life


def test_rul_decreases_to_zero():
    df = pd.DataFrame({
        "unit_nr": [1, 1, 1, 2, 2],
        "time_cycles": [1, 2, 3, 1, 2],
    })
    result = add_remaining_useful_life(df)
    assert result["RUL"].tolist() == [2, 1, 0, 1, 0]


def test_max_cycle_column_dropped():
    df = pd.DataFrame({
        "unit_nr": [1, 1],
        "time_cycles": [1, 2],
    })
    result = add_remaining_useful_life(df)
    assert "max_cycle" not in result.columns
