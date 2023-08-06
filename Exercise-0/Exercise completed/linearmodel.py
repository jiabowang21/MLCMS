import numpy as np


class LinearModel:
    """
    This is a linear model, between data y and x, so that y = a*x+t.
    """

    def __init__(self):
        self.coefficients = None

    def fit(self, X, y):
        """
        Computes the least squares coefficients between x and y.
        Augments x with ones so that the linear model actually is y=ax+t, not y=ax.

        :param X:
        :param y:
        :return:
        """
        x_augment = np.column_stack([X, np.ones(X.shape)])
        self.coefficients = np.linalg.lstsq(x_augment, y, rcond=1e-5)[0]

    def transform(self, X):
        if self.coefficients is None:
            raise ValueError("call fit first")

        x_augment = np.column_stack([X, np.ones(X.shape)])
        return self.coefficients @ x_augment.T
