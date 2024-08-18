from scripts.evaluation_metrics import calculate_metrics
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold, GridSearchCV, LeaveOneOut
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
    # print("Predictions: ", predictions, "Testing values: ", testing)
    r_squared_loocv = r2_score(testing, predictions)

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

    return r_squared_loocv, avg_mse_loocv, avg_rmse_loocv, avg_mae_loocv


def k_fold_cv(X, y, model, cv=6, param_grid=None):

    if param_grid:
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1, verbose=2)
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        print("Best Parameters:", grid_search.best_params_)
    else:
        best_model = model

    # Collect predictions for visualization
    predictions = np.zeros(len(y))
    kf = KFold(n_splits=cv)
    for train_index, test_index in kf.split(X):
        best_model.fit(X.iloc[train_index], y.iloc[train_index])
        y_pred = best_model.predict(X.iloc[test_index])
        print('ypred', y_pred)
        predictions[test_index] = y_pred
        print('predictions', predictions)

    # True values (provided y)
    testing = y
    print('testing', y)

    # Calculate evaluation metrics
    mae = mean_absolute_error(testing, predictions)
    mse = mean_squared_error(testing, predictions)
    rmse = np.sqrt(mse)
    r = r2_score(testing, predictions)
    print("R squared (CV):", r)
    print("Average MSE (CV):", mse)
    print("Average RMSE (CV):", rmse)
    print("Average MAE (CV):", mae)

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


def k_fold_cv_with_deviance(X, y, model, cv=6, param_grid=None):
    """
    Perform k-fold cross-validation and plot the deviance for a given model.

    Parameters:
    X (DataFrame): The input samples.
    y (Series): The target values (class labels).
    model: The model.
    cv (int): Number of k-folds.
    param_grid (dict): The parameter grid for hyperparameter tuning.
    """

    if param_grid:
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        print("Best Parameters:", grid_search.best_params_)
    else:
        best_model = model

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    n_estimators = best_model.get_params()['n_estimators']

    train_deviance = np.zeros(n_estimators, dtype=np.float64)
    test_deviance = np.zeros(n_estimators, dtype=np.float64)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Initialize a new model for each fold
        model_fold = best_model.__class__(**best_model.get_params())
        model_fold.fit(X_train, y_train)

        # Collect predictions for each iteration
        train_predictions = np.zeros((len(X_train), n_estimators))
        test_predictions = np.zeros((len(X_test), n_estimators))

        for i in range(n_estimators):
            model_fold.n_estimators = i + 1
            model_fold.fit(X_train, y_train)  # Fit model with the updated number of estimators

            train_predictions[:, i] = model_fold.predict(X_train)
            test_predictions[:, i] = model_fold.predict(X_test)

        for i in range(n_estimators):
            train_deviance[i] += mean_absolute_error(y_train, train_predictions[:, i])
            test_deviance[i] += mean_absolute_error(y_test, test_predictions[:, i])

    # Average deviance over all folds
    train_deviance /= cv
    test_deviance /= cv

    # Plot deviance
    fig = plt.figure(figsize=(10, 6))
    plt.title("Deviance")
    plt.plot(
        np.arange(n_estimators) + 1,
        train_deviance,
        "b-",
        label="Training Set Deviance",
    )
    plt.plot(
        np.arange(n_estimators) + 1, test_deviance, "r-", label="Test Set Deviance"
    )
    plt.legend(loc="upper right")
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Deviance")
    fig.tight_layout()
    plt.show()


def k_fold_cv_with_deviance_gbr(X, y, model, cv=6, param_grid=None):
    """
    Perform k-fold cross-validation and plot the deviance for a Gradient Boosting Regressor.

    Parameters:
    X (DataFrame): The input samples.
    y (Series): The target values (class labels).
    model (GradientBoostingRegressor): The gradient boosting model.
    cv (int): Number of k-folds.
    param_grid (dict): The parameter grid for hyperparameter tuning.
    """

    if param_grid:
        grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=2)
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        print("Best Parameters:", grid_search.best_params_)
    else:
        best_model = model

    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    train_deviance = np.zeros((best_model.get_params()['n_estimators'],), dtype=np.float64)
    test_deviance = np.zeros((best_model.get_params()['n_estimators'],), dtype=np.float64)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        best_model.fit(X_train, y_train)

        for i, (y_train_pred, y_test_pred) in enumerate(zip(best_model.staged_predict(X_train), best_model.staged_predict(X_test))):
            train_deviance[i] += mean_absolute_error(y_train, y_train_pred)
            test_deviance[i] += mean_absolute_error(y_test, y_test_pred)

    train_deviance /= cv
    test_deviance /= cv

    # Plot deviance
    fig = plt.figure(figsize=(10, 6))
    plt.title("Deviance")
    plt.plot(
        np.arange(best_model.get_params()['n_estimators']) + 1,
        train_deviance,
        "b-",
        label="Training Set Deviance",
    )
    plt.plot(
        np.arange(best_model.get_params()['n_estimators']) + 1, test_deviance, "r-", label="Test Set Deviance"
    )
    plt.legend(loc="upper right")
    plt.xlabel("Boosting Iterations")
    plt.ylabel("Deviance")
    fig.tight_layout()
    plt.show()


def plot_feature_importance(X, model, top_n=15):
    """
    Plot feature importance for a trained model, including MDI.
    Only shows the top `top_n` features.

    Parameters:
    X (DataFrame): The input samples.
    model: The trained model.
    top_n (int): Number of top features to display.
    """

    # Feature Importance (MDI)
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)

    # Select top_n features
    top_idx = sorted_idx[-top_n:]
    top_features = np.array(X.columns)[top_idx]
    top_importance = feature_importance[top_idx]

    # Print numerical results for MDI
    print("Feature Importance (MDI):")
    for feature, importance in zip(top_features, top_importance):
        print(f"{feature}: {importance:.4f}")

    fig = plt.figure(figsize=(18, 6))

    # Plot MDI Feature Importance
    plt.subplot(1, 2, 1)
    pos = np.arange(top_n) + 0.5
    plt.barh(pos, top_importance, align="center")
    plt.yticks(pos, top_features)
    plt.title("Feature Importance (MDI)")

    fig.tight_layout()
    plt.show()
