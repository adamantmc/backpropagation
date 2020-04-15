import numpy as np


def entropy_loss(y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-7, 1.0 - 1e-7)
    return -(np.sum(
        np.sum(
            (
                y_true * np.log(y_pred) +
                (1-y_true) * np.log(1-y_pred)
            ),
            axis=0,
            keepdims=True
        )
    ) / y_pred.shape[1])


def entropy_loss_derivative(y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-7, 1.0 - 1e-7)
    return -(np.divide(y_true, y_pred) - np.divide(1-y_true, 1-y_pred))
