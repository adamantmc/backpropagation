import numpy as np


def relu(x):
    return x * np.asarray(x > 0, dtype=np.int)


def relu_derivative(x):
    return np.asarray(x > 0, dtype=np.int)


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