import numpy as np


def relu(x):
    return np.maximum(x, 0)


def relu_derivative(x):
    _x = np.copy(x)

    _x[_x <= 0] = 0
    _x[_x > 0] = 1

    return _x


def sigmoid(x):
    r = 1 / (1 + np.exp(-x))
    return r


def sigmoid_derivative(x):
    r = sigmoid(x) * (1 - sigmoid(x))
    return r


def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - tanh(x) ** 2


ACTIVATION_FUNCTIONS = {
    "relu": (relu, relu_derivative),
    "tanh": (tanh, tanh_derivative),
    "sigmoid": (sigmoid, sigmoid_derivative)
}