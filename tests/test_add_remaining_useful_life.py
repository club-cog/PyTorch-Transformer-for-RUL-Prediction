import pandas as pd


def test_rul_is_max_cycle_minus_current_cycle():
    from add_remaining_useful_life import add_remaining_useful_life

    df = pd.DataFrame(
        {
            "unit_nr": [1, 1, 1, 2, 2],
            "time_cycles": [1, 2, 3, 1, 2],
        }
    )

    result = add_remaining_useful_life(df)

    # unit 1 max cycle is 3 -> RUL 2,1,0 ; unit 2 max cycle is 2 -> RUL 1,0
    assert result["RUL"].tolist() == [2, 1, 0, 1, 0]


def test_last_cycle_of_each_unit_has_zero_rul():
    from add_remaining_useful_life import add_remaining_useful_life

    df = pd.DataFrame(
        {
            "unit_nr": [1, 1, 2, 2, 2],
            "time_cycles": [10, 11, 5, 6, 7],
        }
    )

    result = add_remaining_useful_life(df)

    last_rows = result.groupby("unit_nr")["time_cycles"].idxmax()
    assert (result.loc[last_rows, "RUL"] == 0).all()


def test_helper_column_is_dropped_and_rows_preserved():
    from add_remaining_useful_life import add_remaining_useful_life

    df = pd.DataFrame({"unit_nr": [1, 1], "time_cycles": [1, 2]})

    result = add_remaining_useful_life(df)

    assert "max_cycle" not in result.columns
    assert "RUL" in result.columns
    assert len(result) == len(df)
