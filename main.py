from nn.nn import NeuralNetwork
import numpy as np
import pickle
import json
import os

def accuracy(y_pred, y_true):
    tp, tn, fp, fn = 0, 0, 0, 0

    for p, y in zip(y_pred, y_true):
        if p == y:
            if y == 1:
                tp += 1
            else:
                tn += 1
        else:
            if y == 1:
                fn += 1
            else:
                fp += 1

    return (tp + tn) / (tp + tn + fp + fn)

# net = NeuralNetwork([y.shape[1]], epochs=10, activation_dict={-1: "sigmoid"}, lr=0.001, batch_size=64, val_x=val_x, val_y=val_y)
# net.fit(x, y)
# net.plot_loss()
