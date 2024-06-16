from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np


class BaselineModel(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.average_target = None

    def fit(self, X, y):
        self.average_target = np.mean(y)
        return self

    def predict(self, X):
        return np.full(len(X), self.average_target)

    def get_params(self, deep=False):
        return {}

    def set_params(self, **params):
        return self
