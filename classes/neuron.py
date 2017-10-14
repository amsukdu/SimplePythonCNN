import numpy as np


class Neuron(object):
    def __init__(self, input_size, bias=0.0):
        self.weights = (np.random.randn(input_size) * np.sqrt(2.0 / input_size)).astype(np.float32)
        self.b = np.float32(bias)
        self.last_input = None
        self.delta = None

    def strength(self, values):
        return np.dot(self.weights, values) + self.b

    def regularization(self):
        return np.sum(np.square(self.weights))