import os

import matplotlib.pyplot as plt
import pandas as pd


def test_visualize_saves_figure(tmp_path, monkeypatch):
    from visualize import visualize

    # Avoid opening any interactive window during the test.
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)

    result = pd.DataFrame({"true": [120, 80, 40], "pred": [118, 85, 30]})
    rmse = 12.5

    visualize(result, rmse)

    assert os.path.exists(os.path.join(tmp_path, f"Transformer({rmse}).png"))
