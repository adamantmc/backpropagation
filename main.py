from nn import NeuralNetwork

x = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]

y = [1, 2, 3]

net = NeuralNetwork([3, 2, 1])
net.fit(x, y)