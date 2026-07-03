import pandas as pd


def add_remaining_useful_life(df: pd.DataFrame) -> pd.DataFrame:
    """Append a piece-wise linear RUL column derived from each unit's max cycle.

    Args:
        df: Frame with ``unit_nr`` and ``time_cycles`` columns.

    Returns:
        Copy of ``df`` with an added ``RUL`` column.
    """
    # Get the total number of cycles for each unit
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()

    # Merge the max cycle back into the original frame
    result_frame = df.merge(max_cycle.to_frame(name='max_cycle'), left_on='unit_nr', right_index=True)

    # Calculate remaining useful life for each row (piece-wise Linear)
    remaining_useful_life = result_frame["max_cycle"] - result_frame["time_cycles"]
    result_frame["RUL"] = remaining_useful_life

    # drop max_cycle as it's no longer needed
    result_frame = result_frame.drop("max_cycle", axis=1)

    return result_frame
