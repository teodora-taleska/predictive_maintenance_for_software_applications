import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer


def load_data():
    """
    Load data from an ARFF file, remove duplicates, and fill missing values with the mean.

    Returns:
    - X (pd.DataFrame): Feature matrix.
    - y (pd.Series): Target variable.
    """

    file_path = '../data/kc1-class-level-numericdefect.arff'

    # Load the ARFF file
    data, meta = arff.loadarff(file_path)

    # Convert ARFF data to a Pandas DataFrame
    df = pd.DataFrame(data)

    # Remove duplicates
    df = df.drop_duplicates()

    # Fill missing values with the mean
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Separate features and target
    y = df_imputed['NUMDEFECTS']
    X = df_imputed.drop('NUMDEFECTS', axis=1)

    return X, y


def exclude_outliers(X, y, contamination=0.1, random_state=42):
    """
    Exclude outliers from the dataset using Isolation Forest.

    Parameters:
    - X (pd.DataFrame): Feature matrix.
    - y (pd.Series): Target variable.
    - contamination (float): The proportion of outliers in the data set.
    - random_state (int): The seed used by the random number generator.

    Returns:
    - X_filtered (pd.DataFrame): Feature matrix without outliers.
    - y_filtered (pd.Series): Target variable without outliers.
    """

    model = IsolationForest(contamination=contamination, random_state=random_state)
    outliers = model.fit_predict(X)

    # Identify number of outliers
    num_outliers = len(outliers[outliers == -1])
    print("Number of outliers: ", num_outliers)

    # Identify inliers
    inlier_indices = np.where(outliers == 1)[0]

    # Filter the original data based on inliers
    X_filtered = X.iloc[inlier_indices]
    y_filtered = y.iloc[inlier_indices]

    return X_filtered, y_filtered


def scale_data(X):
    """
    Scale the data using Standard Scaler.

    Parameters:
    - X (pd.DataFrame): Feature matrix.

    Returns:
    - X_scaled (pd.DataFrame): Scaled feature matrix.
    """

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    return X_scaled
