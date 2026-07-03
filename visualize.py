import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def visualize(result: pd.DataFrame, rmse: np.ndarray) -> None:
    """Plot the true vs. predicted RUL curves and save the figure to disk.

    Args:
        result: Frame whose first column is the true RUL and remaining
            columns are the predictions.
        rmse: RMSE of the predictions, used in the output filename.
    """
    # the true remaining useful life of the testing samples
    true_rul = result.iloc[:, 0:1].to_numpy()
    # the predicted remaining useful life of the testing samples
    pred_rul = result.iloc[:, 1:].to_numpy()

    plt.figure(figsize=(10, 6))
    plt.axvline(x=100, c='r', linestyle='--')
    plt.plot(true_rul, label='Actual Data')
    plt.plot(pred_rul, label='Predicted Data')
    plt.title('RUL Prediction on CMAPSS Data')
    plt.legend()
    plt.xlabel("Samples")
    plt.ylabel("Remaining Useful Life")
    plt.savefig(f'Transformer({rmse}).png')
    plt.show()
