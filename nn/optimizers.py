import numpy as np

class Optimizer(object):
    def __init__(self):
        pass

    def optimize(self, dw, db, weights, biases):
        pass


class GradientDescent(Optimizer):
    def __init__(self, learning_rate):
        super(GradientDescent, self).__init__()
        self._lr = learning_rate

    def optimize(self, dw, db, weights, biases):
        """
        Update weights and biases with regular gradient descent
        :param dw:
        :param db:
        :param weights:
        :param biases:
        :return:
        """
        # len(dw) == len(db) == len(weights) == len(biases)
        for i in range(len(weights)):
            weights[i] = weights[i] - self._lr * dw[i]
            biases[i] = biases[i] - self._lr * db[i]

        return weights, biases


class GradientDescentMomentum(Optimizer):
    def __init__(self, learning_rate, beta):
        self._lr = learning_rate
        self._beta = beta

        self._vdw = []
        self._vdb = []

    def _initialize_v_values(self, w_dims, b_dims):
        for w_dim, b_dim in zip(w_dims, b_dims):
            self._vdw.append(np.zeros(w_dim))
            self._vdb.append(np.zeros(b_dim))

    def optimize(self, dw, db, weights, biases):
        """
        Update weights and biases with regular gradient descent
        :param dw:
        :param db:
        :param weights:
        :param biases:
        :return:
        """
        # len(dw) == len(db) == len(weights) == len(biases)

        if len(self._vdw) == 0:
            w_dims = [w.shape for w in weights]
            b_dims = [b.shape for b in biases]
            self._initialize_v_values(w_dims, b_dims)

        for i in range(len(weights)):
            self._vdw[i] = self._beta * self._vdw[i] + (1 - self._beta) * dw[i]
            self._vdb[i] = self._beta * self._vdb[i] + (1 - self._beta) * db[i]

            weights[i] = weights[i] - self._lr * self._vdw[i]
            biases[i] = biases[i] - self._lr * self._vdb[i]

        return weights, biases


class RMSProp(Optimizer):
    def __init__(self, learning_rate, beta, e=1e-8):
        self._lr = learning_rate
        self._beta = beta
        self._e = e

        self._sdw = []
        self._sdb = []

    def _initialize_s_values(self, w_dims, b_dims):
        for w_dim, b_dim in zip(w_dims, b_dims):
            self._sdw.append(np.zeros(w_dim))
            self._sdb.append(np.zeros(b_dim))

    def optimize(self, dw, db, weights, biases):
        """
        Update weights and biases with regular gradient descent
        :param dw:
        :param db:
        :param weights:
        :param biases:
        :return:
        """
        # len(dw) == len(db) == len(weights) == len(biases)

        if len(self._sdw) == 0:
            w_dims = [w.shape for w in weights]
            b_dims = [b.shape for b in biases]
            self._initialize_s_values(w_dims, b_dims)

        for i in range(len(weights)):
            self._sdw[i] = self._beta * self._sdw[i] + (1 - self._beta) * np.power(dw[i], 2)
            self._sdb[i] = self._beta * self._sdb[i] + (1 - self._beta) * np.power(db[i], 2)

            weights[i] = weights[i] - dw[i] * (self._lr / (np.sqrt(self._sdw[i]) + self._e))
            biases[i] = biases[i] - db[i] * (self._lr / (np.sqrt(self._sdb[i]) + self._e))

        return weights, biases


class Adam(Optimizer):
    def __init__(self, learning_rate, beta1, beta2, e=1e-8):
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2
        self._e = e

        # first moment estimates
        self._mdw = []
        self._mdb = []

        # second moment estimates
        self._vdw = []
        self._vdb = []

        self._t = 0

    def _initialize_values(self, w_dims, b_dims):
        for w_dim, b_dim in zip(w_dims, b_dims):
            self._mdw.append(np.zeros(w_dim))
            self._mdb.append(np.zeros(b_dim))
            self._vdw.append(np.zeros(w_dim))
            self._vdb.append(np.zeros(b_dim))

    def optimize(self, dw, db, weights, biases):
        """
        Update weights and biases with regular gradient descent
        :param dw:
        :param db:
        :param weights:
        :param biases:
        :return:
        """
        # len(dw) == len(db) == len(weights) == len(biases)
        # increase time value
        self._t += 1

        if len(self._mdw) == 0:
            w_dims = [w.shape for w in weights]
            b_dims = [b.shape for b in biases]
            self._initialize_values(w_dims, b_dims)

        for i in range(len(weights)):
            self._mdw[i] = self._beta1 * self._mdw[i] + (1 - self._beta1) * dw[i]
            self._mdb[i] = self._beta1 * self._mdb[i] + (1 - self._beta1) * db[i]
            self._vdw[i] = self._beta2 * self._vdw[i] + (1 - self._beta2) * np.power(dw[i], 2)
            self._vdb[i] = self._beta2 * self._vdb[i] + (1 - self._beta2) * np.power(db[i], 2)

            # bias correction
            mdw = self._mdw[i] / (1-self._beta1**self._t)
            mdb = self._mdb[i] / (1-self._beta1**self._t)
            vdw = self._vdw[i] / (1-self._beta2**self._t)
            vdb = self._vdb[i] / (1-self._beta2**self._t)

            weights[i] = weights[i] - mdw * (self._lr / (np.sqrt(vdw) + self._e))
            biases[i] = biases[i] - mdb * (self._lr / (np.sqrt(vdb) + self._e))

        return weights, biases
