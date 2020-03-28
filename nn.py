import numpy as np
import os
import shutil
from tabulate import tabulate

def relu(x):
    return np.maximum(x, 0)

def relu_derivative(x):
    _x = np.copy(x)

    _x[_x <= 0] = 0
    _x[_x > 0] = 1

    return _x

def sigmoid(x):
    x = np.clip(x, a_min=1e-7, a_max=None)
    r = 1 / (1 + np.exp(-x))
    return r

def sigmoid_derivative(x):
    r = sigmoid(x) * (1 - sigmoid(x))
    return r

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - tanh(x) ** 2

def entropy_loss(y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-7, 1.0-1e-7)
    return -(np.sum((y_true * np.log(y_pred) + (1-y_true)*np.log(1-y_pred))) / y_pred.shape[1])

def entropy_loss_derivative(y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-7, 1.0-1e-7)
    return -(y_true/y_pred) + ((1-y_true)/(1-y_pred))

def dump_values(x, a, z, w, dw, db, epoch):
    dir = "./epoch_{}_values".format(epoch)

    if os.path.exists(dir) and os.path.isdir(dir):
        shutil.rmtree(dir)

    os.makedirs(dir)
    data = {"a": a, "z": z, "w": w, "dw": dw, "db": db}

    for d in data:
        vals = data[d]

        for index, arr in enumerate(vals):
            with open(os.path.join(dir, "{}{}".format(d, index+1)), "w") as f:
                table = tabulate(arr)
                f.write(table + "\n")

    with open(os.path.join(dir,"x"), "w") as f:
        table = tabulate(x)
        f.write(table + "\n")

class BatchProvider:
    def __init__(self, data, batch_size):
        self.data = data
        self.no_examples = len(data)
        self.batch_size = batch_size
        self.index = 0

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= self.no_examples:
            raise StopIteration

        start = self.index
        end = self.index + self.batch_size
        if end > self.no_examples:
            end = self.no_examples

        self.index += self.batch_size

        return self.data[start:end]



class NeuralNetwork(object):
    def __init__(self, layer_units, lr=1.2, epochs=1, batch_size=-1):
        self.layer_units = layer_units
        layers = len(layer_units)

        self._activation_functions = [tanh if i != layers - 1 else sigmoid for i in range(layers)]
        self._activation_function_derivatives = [tanh_derivative if i != layers - 1 else sigmoid_derivative for i in range(layers)]

        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

        self._weights = []
        self._biases = []

        self._a_values_cache = []
        self._z_values_cache = []

    def _initialize_weights(self, no_features):
        np.random.seed(2)
        # Initialize weights
        prev_units = no_features
        for units in self.layer_units:
            w = np.random.randn(units, prev_units)*0.01
            print(w)
            b = np.zeros((units, 1))

            self._weights.append(w)
            self._biases.append(b)

            prev_units = units

    def fit(self, train_x, train_y):
        # Get number of features of each training example
        no_features = len(train_x[0])
        no_examples = len(train_x)

        x = np.asarray(train_x)
        y = np.asarray(train_y)

        if self.batch_size == -1:
            self.batch_size = no_examples

        x_batches = BatchProvider(x, self.batch_size)
        y_batches = BatchProvider(y, self.batch_size)

        self._initialize_weights(no_features)

        for i in range(self.epochs):
            print("Epoch " + str(i))
            for x_batch, y_batch in zip(x_batches, y_batches):
                x_batch = x_batch.T
                y_batch = y_batch.reshape(1, -1)

                self._forward_propagation(x_batch)
                dw, db = self._backpropagation(y_batch)
                # dump_values(x_batch, self._a_values_cache, self._z_values_cache, self._weights, dw, db, i)
                self._update_weights(dw, db)

            self._forward_propagation(x.T)
            preds = self._a_values_cache[-1]
            print(preds.shape)
            loss = self._loss(preds, y.reshape(1, -1))
            print(loss)


    def _forward_propagation(self, x):
        self._z_values_cache.clear()
        self._a_values_cache.clear()

        layers = len(self.layer_units)

        prev_activations = x
        for i in range(layers):
            w = self._weights[i]
            b = self._biases[i]

            z = np.matmul(w, prev_activations) + b

            a = self._activation_functions[i](z)

            prev_activations = a

            self._z_values_cache.append(z)
            self._a_values_cache.append(a)

        return self._z_values_cache, self._a_values_cache

    def _backpropagation(self, y):
        dw_array = []
        db_array = []

        layers = len(self.layer_units)
        batch_size = y.shape[1]

        for i in reversed(range(layers)):
            # dZ[l] = dL/dA[l] * dA[l]/dZ[l] = dA[l] * g[l]'(z[l])

            # Calculate dA
            if i == layers - 1:
                da = self._loss_derivative(self._a_values_cache[i], y)
            else:
                # dA[l-1] = dL/dA[l] * dA[l]/dZ[l] * dZ[l]/dA[l-1] = dL/dZ[l] * W[l] = W[l].T * dZ[l]
                da = np.matmul(self._weights[i+1].T, dz) # dz will be defined from previous iteration

            dz = da * self._activation_function_derivatives[i](self._z_values_cache[i])
            # Calculate dw and db
            # dw = dL/dA[l] * dA[l]/dZ[l] * dZ[l]/dW[l] = 1/m * dZ[l] * A[l-1].T
            dw = (1/batch_size) * np.matmul(dz, self._a_values_cache[i-1].T)
            # db = dL/dA[l] * dA[l]/dZ[l] * dZ[l]/db[l] = 1/m * sum(dZ[l])
            db = (1/batch_size) * np.sum(dz, axis=1, keepdims=True)

            dw_array.append(dw)
            db_array.append(db)

        return list(reversed(dw_array)), list(reversed(db_array))

    def _update_weights(self, dw, db):
        layers = len(self.layer_units)

        for i in range(layers):
            self._weights[i] = self._weights[i] - self.lr * dw[i]
            self._biases[i] = self._biases[i] - self.lr * db[i]

    def _activation_function(self, z):
        return relu(z)

    def _activation_function_derivative(self, z):
        return relu_derivative(z)

    def _loss(self, a, y):
        return entropy_loss(a, y)

    def _loss_derivative(self, a, y):
        return entropy_loss_derivative(a, y)