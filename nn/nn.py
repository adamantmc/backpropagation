import numpy as np
from .batch_provider import BatchProvider
from .activation_functions import ACTIVATION_FUNCTIONS
from .loss_functions import *
from scipy.sparse.csr import csr_matrix

class NeuralNetwork(object):
    def __init__(self, layer_units, lr=0.0001, activation_dict=None, epochs=1, batch_size=64, val_x=None, val_y=None):
        """
        Initializes a Neural Network model
        :param layer_units: List containing number of neurons per layer
        :param lr: Learning rate
        :param activation_dict: Dictionary that maps layer indexes to activation functions.
                Default activation function is ReLU.
        :param epochs: Number of passes over training set
        :param batch_size: Number of examples per iteration - single forward pass and back-propagation
        :param val_x: validation set examples
        :param val_y: validation set labels
        """
        self.layer_units = layer_units
        layers = len(layer_units)

        default_activation = "relu"
        self._activation_functions = []

        for i in range(layers):
            self._activation_functions.append(default_activation)
        if activation_dict is not None:
            for i in activation_dict:
                f = activation_dict[i]
                self._check_activation_function(i, f)
                self._activation_functions[i] = f

        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs

        self._validation_set = (val_x, val_y) \
            if val_x is not None and val_y is not None \
            else None

        self._weights = []
        self._biases = []

        self._a_values_cache = []
        self._z_values_cache = []

        self.training_losses = []
        self.validation_losses = []

    def _check_activation_function(self, layer, function):
        assert function in ACTIVATION_FUNCTIONS, \
            "Unknown activation function for layer {layer} ({function}) - " \
            "the activation functions currently supported are {valid_functions}".format(
                layer=layer, function=function, valid_functions=ACTIVATION_FUNCTIONS.keys()
            )

    def _initialize_weights(self, no_features):
        # Initialize weights
        prev_units = no_features
        for units in self.layer_units:
            # He et al
            print(units, prev_units)
            w = np.random.normal(scale=np.sqrt(2/prev_units), size=(units, prev_units))
            b = np.zeros((units, 1))

            self._weights.append(w)
            self._biases.append(b)

            prev_units = units

    def fit(self, train_x, train_y):
        self.training_losses = []
        self.validation_losses = []

        x = train_x
        y = train_y

        if type(train_x) == list:
            x = np.asarray(x)
        if type(train_y) == list:
            y = np.asarray(y)

        # Get number of features of each training example
        no_features = x.shape[1]
        no_examples = x.shape[0]

        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        if self.batch_size == -1:
            self.batch_size = no_examples

        x_batches = BatchProvider(x, self.batch_size)
        y_batches = BatchProvider(y, self.batch_size)

        self._initialize_weights(no_features)

        for i in range(self.epochs):
            training_loss = 0
            steps = 0

            for x_batch, y_batch in zip(x_batches, y_batches):
                if type(x_batch) == csr_matrix:
                    x_batch = np.asarray(x_batch.todense())

                x_batch = x_batch.T
                y_batch = y_batch.T
                z_cache, a_cache = self._forward_propagation(x_batch, self._weights, self._biases)

                self._z_values_cache = z_cache
                self._a_values_cache = a_cache
                dw, db = self._backpropagation(x_batch, y_batch)
                loss = self._loss(self._a_values_cache[-1], y_batch)
                training_loss += loss
                steps += 1

                # self._gradient_check(x_batch, y_batch, self._weights, self._biases, dw, db)
                # dump_values(x_batch, self._a_values_cache, self._z_values_cache, self._weights, dw, db, i)

                self._update_weights(dw, db)

            self.training_losses.append(training_loss / steps)
            val_str = ""
            if self._validation_set is not None:
                val_loss = self.evaluate(self._validation_set[0], self._validation_set[1])
                self.validation_losses.append(val_loss)
                val_str = "Validation Loss: {}".format(val_loss)

            print("Epoch {} Training Loss: {} {}".format(i+1, self.training_losses[-1], val_str))

    def evaluate(self, x, y):
        """
        Evaluates model performance on given set
        :param x: np.array with shape (#EXAMPLES, #FEATURES)
        :param y: np.array with shape (#EXAMPLES, #OUTPUT_DIMENSIONS)
        :return: model loss on given data
        """
        z_values, a_values = self._forward_propagation(x.T, self._weights, self._biases)
        preds = a_values[-1]
        loss = self._loss(preds, y.T)
        return loss

    def predict(self, x, prob_threshold=0.5):
        z_values, a_values = self._forward_propagation(x.T, self._weights, self._biases)
        preds = a_values[-1]

        if prob_threshold is not None:
            preds[preds > prob_threshold] = 1
            preds[preds <= prob_threshold] = 0

        return preds.T

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
            assert z.shape[0] == units

            a = self._activation_function(z, i)
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
            # assert da.shape[1] == self.batch_size

            dz = da * self._activation_function_derivative(self._z_values_cache[i], i)
            assert dz.shape[0] == self.layer_units[i]

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
                    assert abs(approx_dw - dw[l][i][j]) < 1e-7, \
                        "{} - {} = {}".format(approx_dw, dw[l][i][j], abs(approx_dw - dw[l][i][j]))

    def _update_weights(self, dw, db):
        layers = len(self.layer_units)

        for i in range(layers):
            self._weights[i] = self._weights[i] - self.lr * dw[i]
            self._biases[i] = self._biases[i] - self.lr * db[i]

    def _activation_function(self, z, layer):
        func_name = self._activation_functions[layer]
        return ACTIVATION_FUNCTIONS[func_name][0](z)

    def _activation_function_derivative(self, z, layer):
        func_name = self._activation_functions[layer]
        return ACTIVATION_FUNCTIONS[func_name][1](z)

    def _loss(self, a, y):
        return entropy_loss(a, y)

    def _loss_derivative(self, a, y):
        return entropy_loss_derivative(a, y)