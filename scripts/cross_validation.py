from sklearn.model_selection import LeaveOneOut, KFold
from scripts.evaluation_metrics import calculate_metrics, calculate_r2
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt


def loocv(X, y, model):
    loo = LeaveOneOut()
    mse_loocv = []
    mae_loocv = []
    rmse_loocv = []
    predictions = []
    testing = []

    for train_index, test_index in loo.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions.append(y_pred[0])
        testing.append(y_test.values[0])

        # Calculate evaluation metrics
        mse, rmse, mae, r2 = calculate_metrics(y_test, y_pred)
        mse_loocv.append(mse)
        mae_loocv.append(mae)
        rmse_loocv.append(rmse)

    avg_mse_loocv = np.mean(mse_loocv)
    avg_mae_loocv = np.mean(mae_loocv)
    avg_rmse_loocv = np.sqrt(avg_mse_loocv)
    r_squared_loocv = r2_score(predictions, testing)

    print("R squared (LOOCV):", r_squared_loocv)
    print("Average MSE (LOOCV):", avg_mse_loocv)
    print("Average RMSE (LOOCV):", avg_rmse_loocv)
    print("Average MAE (LOOCV):", avg_mae_loocv)

    # Visualize predictions vs true values with different colors
    plt.scatter(range(len(testing)), testing, color='blue', label='True Values')
    plt.scatter(range(len(predictions)), predictions, color='red', label='Predicted Values')
    plt.xlabel('Index')
    plt.ylabel('NUMDEFECTS')
    plt.title('Predicted vs True Values (LOOCV)')
    plt.legend()
    plt.show()

    # Visualize residuals
    residuals = np.array(testing) - np.array(predictions)
    plt.hist(residuals, bins=20)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Histogram')
    plt.show()


def k_fold_cv(X, y, model, cv=6):
    scoring = {
        'mse': make_scorer(mean_squared_error),
        'mae': make_scorer(mean_absolute_error),
        'r2': make_scorer(r2_score),
    }

    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)

    for scorer_name in scoring.keys():
        mean_score = np.mean(cv_results['test_' + scorer_name])
        print(f"Mean {scorer_name.upper()}: {mean_score}")

    print('RMSE', np.sqrt(np.mean(cv_results['test_mse'])))

    # Collect predictions for visualization
    predictions = np.zeros(len(y))
    kf = KFold(n_splits=cv)
    for train_index, test_index in kf.split(X):
        model.fit(X.iloc[train_index], y.iloc[train_index])
        y_pred = model.predict(X.iloc[test_index])
        predictions[test_index] = y_pred

    # True values (provided y)
    testing = y

    # Visualize predictions vs true values with different colors
    plt.scatter(range(len(testing)), testing, color='blue', label='True Values')
    plt.scatter(range(len(predictions)), predictions, color='red', label='Predicted Values')
    plt.xlabel('Index')
    plt.ylabel('NUMDEFECTS')
    plt.title(f'Predicted vs True Values ({cv}-Fold CV)')
    plt.legend()
    plt.show()

    # Visualize residuals
    residuals = np.array(testing) - np.array(predictions)
    plt.hist(residuals, bins=20, color='purple')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title('Residuals Histogram')
    plt.show()
