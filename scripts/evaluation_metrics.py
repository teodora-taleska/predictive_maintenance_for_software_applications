from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def calculate_r2(y_true, y_pred):

    """Calculate R^2 score manually."""

    mean_y = np.mean(y_true)

    # Calculate sum of squared errors
    ss_total = np.sum((y_true - mean_y) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)

    # Check if ss_total is zero
    if ss_total == 0:
        r2 = np.nan  # Set R^2 to NaN if denominator is zero
    else:
        # Calculate R^2
        r2 = 1 - (ss_residual / ss_total)

    return r2


def calculate_metrics(y_true, y_pred):

    """Calculate evaluation metrics: MSE, RMSE, MAE, R2."""

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = calculate_r2(y_true, y_pred)

    return mse, rmse, mae, r2


def print_metrics(mse, rmse, mae, r2):

    """Print evaluation metrics."""

    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")


def calculate_and_print_metrics(y_true, y_pred):

    """Calculate and print evaluation metrics."""

    mse, rmse, mae, r2 = calculate_metrics(y_true, y_pred)
    print_metrics(mse, rmse, mae, r2)


if __name__ == "__main__":

    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])

    calculate_and_print_metrics(y_true, y_pred)
