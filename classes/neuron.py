import numpy as np


class Neuron(object):
    def __init__(self, input_size, bias=0.0):
        limit = np.sqrt(2.0 / input_size)
        self.weights = (np.random.randn(input_size) * limit).astype(np.float32)
        self.b = np.float32(bias)
        self.last_input = None
        self.delta = 0

    def strength(self, values):
        return np.dot(self.weights, values) + self.b

    def regularization(self):
        return np.sum(np.square(self.weights))
