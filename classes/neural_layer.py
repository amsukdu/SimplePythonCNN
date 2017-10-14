from classes.layer import Layer
from classes.neuron import Neuron
import numpy as np
import classes.utils as u


class NeuralLayer(Layer):

    def __init__(self, input_size, k, u_type='adam', a_type='relu', dropout=1):
        self.neurons = []
        self.forward_result = None
        self.k = k
        self.dropout = dropout

        self.u_type = u_type
        self.a_type= a_type

        if isinstance(input_size, tuple):
            input_size = np.prod(input_size)

        for n in range(k):
            n = Neuron(input_size)

            if u_type == 'adam':
                n.m, n.v = 0, 0
            elif u_type == 'm':
                n.v = 0, 0
            elif u_type == 'nag':
                n.v, n.v_prev = 0, 0
            elif u_type == 'rmsprop':
                n.cache, n.v = 0, 0

            self.neurons.append(n)

    def predict(self, batch):

        if batch.ndim > 2:
            batch = batch.reshape(batch.shape[0], -1).T

        forward_result = []
        for n in self.neurons:
            if self.activation:
                if self.a_type == 'relu':
                    forward_result.append(u.relu(n.strength(batch)))
                elif self.a_type == 'sigmoid':
                    forward_result.append(u.sigmoid(n.strength(batch)))
            else:
                forward_result.append(n.strength(batch))

        self.forward_result = np.array(forward_result)
        return self.forward_result

    def forward(self, batch):

        if batch.ndim > 2:
            batch = batch.reshape(batch.shape[0], -1).T
        forward_result = []
        l2 = 0
        for n in self.neurons:
            if self.activation:
                if self.a_type == 'relu':
                    forward_result.append(u.relu(n.strength(batch)))
                elif self.a_type == 'sigmoid':
                    forward_result.append(u.sigmoid(n.strength(batch)))
            else:
                forward_result.append(n.strength(batch))

            n.last_input = batch
            l2 += n.regularization()

        self.forward_result = np.array(forward_result)
        return self.forward_result, l2

    def backward(self, d, need_d=True):
        weights = []

        if self.activation:
            if self.a_type == 'sigmoid':
                delta = d * u.sigmoid_d(self.forward_result)
            else:
                delta = d * u.relu_d(self.forward_result)
        else:
            delta = d

        for index, n in enumerate(self.neurons):
            n.delta = delta[index]
            if need_d:
                weights.append(n.weights)

        if not need_d:
            return
        else:
            weights = np.array(weights)
            return weights.T.dot(delta)

    def output_size(self):
        if self.forward_result:
            return self.forward_result.shape[1:]
        else:
            return self.k

    def update(self, lr, l2_reg, t=0):
        if self.u_type == 'adam':
            u.adam_update(self.neurons, lr, t=t, l2_reg=l2_reg)
        elif self.u_type == 'rmsprop':
            u.rmsprop(self.neurons, lr, l2_reg=l2_reg)
        elif self.u_type == 'm':
            u.momentum_update(self.neurons, lr, l2_reg=l2_reg)
        elif self.u_type == 'nag':
            u.nag_update(self.neurons, lr, l2_reg=l2_reg)
        elif self.u_type == 'v':
            u.vanila_update(self.neurons, lr, l2_reg=l2_reg)
