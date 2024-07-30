import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import make_scorer, mean_squared_error, r2_score, mean_absolute_error
from scripts.cross_validation import loocv, k_fold_cv

class BaselineModel(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.average_target = None

    def fit(self, X, y):
        self.average_target = y.mean()
        return self

    def predict(self, X):
        return np.full(len(X), self.average_target)

    def get_params(self, deep=False):
        return {}

    def set_params(self, **params):
        return self
