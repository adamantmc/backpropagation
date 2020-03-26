import numpy as np


def relu(x):
    return np.maximum(x, 0)


def relu_derivative(x):
    _x = np.copy(x)

    _x[_x <= 0] = 0
    _x[_x > 0] = 1

    return _x


def entropy_loss(y_pred, y_true):
    y_pred = np.clip(y_pred, 1e-7, 1.0-1e-7)
    return -(np.sum((y_true * np.log(y_pred) + (1-y_true)*np.log(1-y_pred))) / y_pred.shape[1])


def entropy_loss_derivative(y_pred, y_true):
    return -(y_true/y_pred) + ((1-y_true)/(1-y_pred))


class NeuralNetwork(object):
    def __init__(self, layer_units, lr=0.0001, batch_size=-1):
        self.layer_units = layer_units
        self.lr = lr
        self.batch_size = batch_size

        self._weights = []
        self._biases = []

        self._a_values_cache = []
        self._z_values_cache = []

    def _initialize_weights(self, no_features):
        # Initialize weights
        prev_units = no_features
        for units in self.layer_units:
            w = np.random.random((units, prev_units))
            b = np.random.random((units, 1))

            self._weights.append(w)
            self._biases.append(b)

            prev_units = units

    def fit(self, train_x, train_y):
        # Get number of features of each training example
        no_features = len(train_x[0])

        x = np.asarray(train_x).T
        y = np.asarray(train_y).reshape(1, -1)

        self._initialize_weights(no_features)

        self._forward_propagation(x)
        self._backpropagation(y)
        
    def _forward_propagation(self, x):
        layers = len(self.layer_units)

        prev_activations = x
        for i in range(layers):
            w = self._weights[i]
            b = self._biases[i]

            print(w.shape, prev_activations.shape, b.shape)

            z = np.matmul(w, prev_activations) + b
            a = self._activation_function(z)

            print(a.shape)

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

            dz = da * self._activation_function_derivative(self._z_values_cache[i])

            # Calculate dw and db
            # dw = dL/dA[l] * dA[l]/dZ[l] * dZ[l]/dW[l] = 1/m * dZ[l] * A[l-1].T
            dw = (1/batch_size) * np.matmul(dz, self._a_values_cache[i-1].T)
            # db = dL/dA[l] * dA[l]/dZ[l] * dZ[l]/db[l] = 1/m * sum(dZ[l])
            db = (1/batch_size) * np.sum(dz, axis=1, keepdims=True)

            dw_array.append(dw)
            db_array.append(db)

        return reversed(dw_array), reversed(db_array)

    def _activation_function(self, z):
        return relu(z)

    def _activation_function_derivative(self, z):
        return relu_derivative(z)

    def _loss(self, a, y):
        return entropy_loss(a, y)

    def _loss_derivative(self, a, y):
        return entropy_loss_derivative(a, y)