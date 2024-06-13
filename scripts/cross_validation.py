from sklearn.model_selection import LeaveOneOut
from scripts.evaluation_metrics import calculate_metrics
from sklearn.model_selection import cross_validate, KFold
import numpy as np


def loocv(X, y, model):
    loo = LeaveOneOut()
    mse_loocv = []
    r2_loocv = []
    mae_loocv = []
    rmse_loocv = []
    predictions = []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions.append(y_pred[0])

        # Calculate evaluation metrics
        mse, rmse, mae, r2 = calculate_metrics(y_test, y_pred)
        r2_loocv.append(r2)
        mse_loocv.append(mse)
        mae_loocv.append(mae)
        rmse_loocv.append(rmse)

    avg_r2_loocv = np.mean(r2_loocv)
    avg_mse_loocv = np.mean(mse_loocv)
    avg_mae_loocv = np.mean(mae_loocv)
    avg_rmse_loocv = np.sqrt(avg_mse_loocv)

    print("Average R^2 (LOOCV):", avg_r2_loocv)
    print("Average MSE (LOOCV):", avg_mse_loocv)
    print("Average RMSE (LOOCV):", avg_rmse_loocv)
    print("Average MAE (LOOCV):", avg_mae_loocv)

    return predictions


def k_fold_cv(X, y, model, cv=6):
    scorers = {
        'mse': 'neg_mean_squared_error',
        'mae': 'neg_mean_absolute_error',
        'r2': 'r2'
    }

    cv_results = cross_validate(model, X, y, cv=cv, scoring=scorers)

    mse = -np.mean(cv_results['test_mse'])
    mae = -np.mean(cv_results['test_mae'])
    r2 = np.mean(cv_results['test_r2'])

    print(f"Mean R2: {r2:.4f}")
    print(f"Mean MSE: {mse:.4f}")
    print(f"RMSE: {np.sqrt(mse):.4f}")
    print(f"Mean MAE: {mae:.4f}")


