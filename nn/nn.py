import numpy as np
from .batch_provider import BatchProvider
from .activation_functions import ACTIVATION_FUNCTIONS
from .loss_functions import *

class NeuralNetwork(object):
    def __init__(self, layer_units, lr=0.0001, activation_dict=None, epochs=1,
                 batch_size=64, l2_lambda=None, val_x=None, val_y=None):
        """
        Initializes a Neural Network model
        :param layer_units: List containing number of neurons per layer
        :param lr: Learning rate
        :param activation_dict: Dictionary that maps layer indexes to activation functions.
                Default activation function is ReLU.
        :param epochs: Number of passes over training set
        :param batch_size: Number of examples per iteration - single forward pass and back-propagation
        :param l2_lambda: L2 Regularization parameter - if None, no regularization is applied
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
        self._regularization_param = l2_lambda
        self._validation_set = (val_x, val_y) \
            if val_x is not None and val_y is not None \
            else None

        self._weights = []
        self._biases = []

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
            if i == 55:
                self.lr = 0.01
            training_loss = 0
            steps = 0

            for x_batch, y_batch in zip(x_batches, y_batches):
                x_batch = x_batch.T
                y_batch = y_batch.T

                loss = self._train_batch(x_batch, y_batch)

                training_loss += loss
                steps += 1

            self.training_losses.append(training_loss / steps)

            val_str = ""
            if self._validation_set is not None:
                val_loss = self.evaluate(self._validation_set[0], self._validation_set[1])
                self.validation_losses.append(val_loss)
                val_str = "Validation Loss: {}".format(val_loss)

            print("Epoch {} Training Loss: {} {}".format(i+1, self.training_losses[-1], val_str))

    def _train_batch(self, x_batch, y_batch):
        z_cache, a_cache = self._forward_propagation(x_batch, self._weights, self._biases)

        dw, db = self._backpropagation(x_batch, y_batch, z_cache, a_cache)
        # self._gradient_check(x_batch, y_batch, dw, db)
        loss = self._loss(a_cache[-1], y_batch)
        self._update_weights(dw, db)

        return loss

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

    def predict(self, x, prob_threshold=0.5, batch_size=-1):
        batch_provider = BatchProvider(x, batch_size)
        preds_per_batch = []

        for x_batch in batch_provider:
            z_values, a_values = self._forward_propagation(x_batch.T, self._weights, self._biases)
            preds = a_values[-1]

            if prob_threshold is not None:
                preds[preds > prob_threshold] = 1
                preds[preds <= prob_threshold] = 0

            preds_per_batch.append(preds.T)

        return np.vstack(preds_per_batch)

    def _forward_propagation(self, x, weights, biases):
        z_cache = []
        a_cache = []

        prev_activations = x
        for i, units in enumerate(self.layer_units):
            w = weights[i]
            b = biases[i]

            z = np.matmul(w, prev_activations) + b
            a = self._activation_function(z, i)

            # Sanity-check assertions
            assert w.shape[0] == units
            assert b.shape[0] == units
            assert z.shape[0] == units
            assert a.shape == z.shape

            prev_activations = a

            z_cache.append(z)
            a_cache.append(a)

        return z_cache, a_cache

    def _backpropagation(self, x, y, z_cache, a_cache):
        """
        Backpropagation algorithm - calculate partial derivatives of loss function
        with regard to weights (dw) and biases (db) using the chain rule. The following
        equations are used.

        dw[l] = dL/da[l] * da[l]/dz[l] * dz[l]/dW[l] = 1/m * dz[l] * A[l-1].T
        db[l] = dL/da[l] * da[l]/dz[l] * dz[l]/db[l] = 1/m * sum(dz[l])

        where:
            dz[l] = dL/da[l] * da[l]/dZ[l] = da[l] * g[l]'(z[l])
            da[l-1] = dL/da[l] * da[l]/dZ[l] * dz[l]/da[l-1] = dL/dz[l] * W[l] = W[l].T * dz[l]

            A: activation matrix (self._a_values_caache)
            W: weights matrix (self._weights)
            g[l]: activation function of layer l
            dx = partial derivative of loss fucntion L w.r.t. x

        first da (last layer) equals the derivative loss with regard to its input (a)

        :param x:
        :param y:
        :return:
        """
        dw_array = []
        db_array = []

        layers = len(self.layer_units)
        batch_size = y.shape[1]

        for i in reversed(range(layers)):
            # Calculate da
            if i == layers - 1:
                da = self._loss_derivative(a_cache[i], y)
            else:
                da = np.matmul(self._weights[i+1].T, dz) # dz will be defined from previous iteration

            dz = da * self._activation_function_derivative(z_cache[i], i)

            # Calculate dw and db
            prev_activations = a_cache[i-1] if i != 0 else x
            dw = (1/batch_size) * np.matmul(dz, prev_activations.T)

            if self._regularization_param is not None:
                dw += (self._regularization_param / batch_size) * self._weights[i]

            db = (1/batch_size) * np.sum(dz, axis=1, keepdims=True)

            # Sanity-check assertions
            assert da.shape[0] == self.layer_units[i]
            assert da.shape[1] == batch_size
            assert dz.shape[0] == self.layer_units[i]
            assert dw.shape == self._weights[i].shape
            assert db.shape == self._biases[i].shape

            dw_array.append(dw)
            db_array.append(db)

        return list(reversed(dw_array)), list(reversed(db_array))

    def _gradient_check(self, x, y, dw, db):
        layers = len(self.layer_units)
        epsilon = 1e-6

        for l in range(layers):
            w_shape = self._weights[l].shape
            for i in range(w_shape[0]):
                for j in range(w_shape[1]):
                    w = self._weights[l][i][j]

                    self._weights[l][i][j] = w + epsilon
                    right_z, right_a = self._forward_propagation(x, self._weights, self._biases)
                    right_loss = self._loss(right_a[-1], y)

                    self._weights[l][i][j] = w - epsilon
                    left_z, left_a = self._forward_propagation(x, self._weights, self._biases)
                    left_loss = self._loss(left_a[-1], y)

                    self._weights[l][i][j] = w

                    approx_dw = (right_loss - left_loss) / (2 * epsilon)
                    assert abs(approx_dw - dw[l][i][j]) < 1e-7, \
                        "{} - {} = {}, left: {} right: {}, w: {} epsilon: {}".format(
                            approx_dw, dw[l][i][j], abs(approx_dw - dw[l][i][j]), left_loss, right_loss, w, epsilon)

    def l2_regularization_loss(self, weights, no_examples):
        cum_sq_norm = 0

        for w in weights:
            cum_sq_norm += np.power(np.linalg.norm(w), 2)

        l2_loss = self._regularization_param * cum_sq_norm / (2 * no_examples)
        return l2_loss

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
        loss = entropy_loss(a, y)

        if self._regularization_param is not None:
            loss += self.l2_regularization_loss(self._weights, a.shape[1])

        return loss

    def _loss_derivative(self, a, y):
        return entropy_loss_derivative(a, y)