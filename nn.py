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
    return -(np.divide(y_true, y_pred) - np.divide(1-y_true, 1-y_pred))

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
    def __init__(self, layer_units, lr=0.0001, epochs=1, batch_size=-1):
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
            w = np.zeros((units, prev_units))
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
                y_batch = y_batch.T
                z_cache, a_cache = self._forward_propagation(x_batch, self._weights, self._biases)

                self._z_values_cache = z_cache
                self._a_values_cache = a_cache
                dw, db = self._backpropagation(x_batch, y_batch)

                # self._gradient_check(x_batch, y_batch, self._weights, self._biases, dw, db)
                # dump_values(x_batch, self._a_values_cache, self._z_values_cache, self._weights, dw, db, i)

                self._update_weights(dw, db)

            self._forward_propagation(x.T, self._weights, self._biases)
            preds = self._a_values_cache[-1]
            loss = self._loss(preds, y.T)
            print(loss)

        print(self._a_values_cache[-1])

    def _forward_propagation(self, x, weights, biases):
        z_cache = []
        a_cache = []

        prev_activations = x
        for i, units in enumerate(self.layer_units):
            w = weights[i]
            assert w.shape[0] == units
            b = biases[i]
            assert b.shape[0] == units

            z = np.matmul(w, prev_activations) + b
            assert z.shape == (units, self.batch_size)

            a = self._activation_functions[i](z)
            assert a.shape == z.shape

            prev_activations = a

            z_cache.append(z)
            a_cache.append(a)

        return z_cache, a_cache

    def _backpropagation(self, x, y):
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

            assert da.shape[0] == self.layer_units[i]
            assert da.shape[1] == self.batch_size

            dz = da * self._activation_function_derivatives[i](self._z_values_cache[i])
            assert dz.shape[0] == self.layer_units[i]
            assert dz.shape[1] == self.batch_size

            # Calculate dw and db
            # dw = dL/dA[l] * dA[l]/dZ[l] * dZ[l]/dW[l] = 1/m * dZ[l] * A[l-1].T
            prev_activations = self._a_values_cache[i-1] if i != 0 else x
            dw = (1/batch_size) * np.matmul(dz, prev_activations.T)
            assert dw.shape == self._weights[i].shape

            # db = dL/dA[l] * dA[l]/dZ[l] * dZ[l]/db[l] = 1/m * sum(dZ[l])
            db = (1/batch_size) * np.sum(dz, axis=1, keepdims=True)
            assert db.shape == self._biases[i].shape

            dw_array.append(dw)
            db_array.append(db)

        return list(reversed(dw_array)), list(reversed(db_array))

    def _gradient_check(self, x, y, weights, biases, dw, db):
        layers = len(self.layer_units)
        epsilon = 1e-6

        for l in range(layers):
            wl = weights[l]
            for i in range(wl.shape[0]):
                for j in range(wl.shape[1]):
                    w = weights[l][i][j]

                    weights[l][i][j] = w + epsilon
                    right_z, right_a = self._forward_propagation(x, weights, biases)
                    weights[l][i][j] = w - epsilon
                    left_z, left_a = self._forward_propagation(x, weights, biases)

                    weights[l][i][j] = w

                    right_loss = self._loss(right_a[-1], y)
                    left_loss = self._loss(left_a[-1], y)

                    approx_dw = (right_loss - left_loss) / (2 * epsilon)
                    assert approx_dw - dw[l][i][j] < 1e-7


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