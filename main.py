from nn import NeuralNetwork
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import normalize
import numpy as np
import matplotlib.pyplot as plt

# x, y = load_breast_cancer(return_X_y=True)
# x = np.asarray(x)
#
# corrs = np.corrcoef(x, y, rowvar=False)
# indexed = []
#
# for c in corrs[-1][:-1]:
#     indexed.append((c, len(indexed)))
#
# indexed.sort(key=lambda x: x[0])
#
# cols = []
# for i in range(8):
#     cols.append(indexed[i][1])
#
# # x = x[:, cols]
#
# print(x.shape)
#
# x = normalize(x)
# print(y)


def load_planar_dataset():
    m = 400  # number of examples
    N = int(m / 2)  # number of points per class
    D = 2  # dimensionality
    X = np.zeros((m, D))  # data matrix where each row is a single example
    Y = np.zeros((m, 1), dtype='uint8')  # labels vector (0 for red, 1 for blue)
    a = 4  # maximum ray of the flower

    for j in range(2):
        ix = range(N * j, N * (j + 1))
        t = np.linspace(j * 3.12, (j + 1) * 3.12, N) + np.random.randn(N) * 0.2  # theta
        r = a * np.sin(4 * t) + np.random.randn(N) * 0.2  # radius
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        Y[ix] = j

    return X, Y

np.random.seed(1)
x, y = load_planar_dataset()
# plt.figure()
# plt.scatter(x[:, 0], x[:, 1], c=y[:, 0], s=40, cmap=plt.cm.Spectral)
# plt.show()
# print(x.shape, y.shape)

net = NeuralNetwork([4, 1], epochs=10000)
net.fit(x, y)