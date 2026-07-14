import numpy as np
import pandas as pd
import torch

N_FEATURES = 14
N_UNITS = 100  # testing() is hard-coded to iterate units 1..100


class ConstantModel:
    """Minimal stand-in for the trained Transformer used to exercise the
    evaluation loop without any training. Returns a fixed prediction."""

    def __init__(self, value):
        self.value = value

    def forward(self, x, t):
        return torch.Tensor([self.value])


def _build_grouped_test_set(rows_per_unit=3):
    records = []
    for unit in range(1, N_UNITS + 1):
        for cycle in range(1, rows_per_unit + 1):
            row = [unit, cycle] + [float(cycle + i) for i in range(N_FEATURES)]
            records.append(row)
    columns = ["unit_nr", "time_cycles"] + [f"f_{i}" for i in range(N_FEATURES)]
    df = pd.DataFrame(records, columns=columns)
    return df.groupby(by="unit_nr")


def test_testing_returns_rmse_and_sorted_result():
    from testing import testing

    group_test = _build_grouped_test_set()
    y_test = pd.DataFrame({"RUL": list(range(1, N_UNITS + 1))})

    rmse, result = testing(group_test, y_test, ConstantModel(10.0))

    assert np.asarray(rmse).size == 1
    assert float(np.ravel(rmse)[0]) >= 0
    assert len(result) == N_UNITS
    # result is sorted by the true RUL in descending order
    assert result["RUL"].tolist() == sorted(result["RUL"].tolist(), reverse=True)


def test_prediction_decrement_is_clamped_at_zero():
    from testing import testing

    group_test = _build_grouped_test_set()
    y_test = pd.DataFrame({"RUL": [0] * N_UNITS})

    # A model that always predicts <=1 must yield a non-negative RMSE built
    # entirely from clamped-at-zero predictions.
    rmse, result = testing(group_test, y_test, ConstantModel(0.5))

    preds = result.drop(columns="RUL").to_numpy().ravel()
    assert (preds >= 0).all()


def test_perfect_predictions_give_zero_rmse():
    from testing import testing

    # The loop decrements the prediction by 1 on the final step, so a model
    # emitting 1.0 yields 0.0 predictions, matching an all-zero ground truth.
    group_test = _build_grouped_test_set()
    y_test = pd.DataFrame({"RUL": [0] * N_UNITS})

    rmse, _ = testing(group_test, y_test, ConstantModel(1.0))

    assert float(np.ravel(rmse)[0]) == 0.0
