from sklearn.model_selection import LeaveOneOut
from scripts.evaluation_metrics import calculate_metrics, calculate_r2
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
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

    scoring = {
        'mse': make_scorer(mean_squared_error),
        'mae': make_scorer(mean_absolute_error),
        'r2': make_scorer(calculate_r2),
    }

    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)

    scorers = {
        'mse': 'mean squared error',
        'mae': 'mean absolute error',
        'r2': 'r squared',
    }

    mse = 0
    for scorer_name, scorer in scorers.items():
        scores = cv_results['test_' + scorer_name]
        mean_score = np.mean(scores)
        if scorer_name == 'mse':
            mse = mean_score
        print(f"Mean {scorer_name.upper()}: {mean_score}")
    print('RMSE', np.sqrt(mse))


