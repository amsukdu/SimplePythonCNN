import numpy as np
from classes.neural_layer import NeuralLayer
import classes.utils as u


class ConvLayer(NeuralLayer):

    def __init__(self, input_size, k, f=3, s=1, p=1, u_type='adam', a_type='relu', dropout=1):
        self.image_size = 0
        self.w = input_size[2]
        self.h = input_size[1]
        self.d = input_size[0]

        self.k = k
        self.f = f
        self.s = s
        self.p = p

        self.w2 = int((self.w - self.f + 2 * self.p) / self.s + 1)
        self.h2 = int((self.h - self.f + 2 * self.p) / self.s + 1)
        self.d2 = k

        super(ConvLayer, self).__init__(f*f*self.d, k, u_type=u_type, a_type=a_type, dropout=dropout)

    def predict(self, batch):
        self.image_size = batch.shape[0]
        cols = u.im2col_indices(batch, self.f, self.f, self.p, self.s)
        sum_weights = []
        bias = []
        for n in self.neurons:
            bias.append(n.b)
            sum_weights.append(n.weights)

        sum_weights = np.array(sum_weights)
        strength = (sum_weights.dot(cols) + np.array(bias).reshape(sum_weights.shape[0], 1)).reshape(self.k, self.h2, self.w2, -1).transpose(3, 0, 1, 2)

        if self.activation:
            if self.a_type == 'sigmoid':
                return u.sigmoid(strength)
            else:
                return u.relu(strength)
        else:
            return strength

    def forward(self, batch):
        self.image_size = batch.shape[0]
        cols = u.im2col_indices(batch, self.f, self.f, self.p, self.s)
        l2 = 0
        sum_weights = []
        bias = []
        for n in self.neurons:
            n.last_input = cols
            sum_weights.append(n.weights)
            bias.append(n.b)
            l2 += n.regularization()

        sum_weights = np.array(sum_weights)
        strength = (sum_weights.dot(cols) + np.array(bias).reshape(sum_weights.shape[0], 1))
        strength = strength.reshape(self.k, self.h2, self.w2, -1).transpose(3, 0, 1, 2)

        if self.activation:
            if self.a_type == 'sigmoid':
                self.forward_result = u.sigmoid(strength)
            else:
                self.forward_result = u.relu(strength)
        else:
            self.forward_result = strength

        return self.forward_result, l2

    def backward(self, d, need_d=True):
        if d.ndim < 4:
            d = d.reshape(self.w2, self.h2, self.k, -1).T

        if self.activation:
            if self.a_type == 'sigmoid':
                delta = d * u.sigmoid_d(self.forward_result)
            else:
                delta = d * u.relu_d(self.forward_result)
        else:
            delta = d

        sum_weights = []
        for index, n in enumerate(self.neurons):
            n.delta = delta[:, index, :, :].transpose(1, 2, 0).flatten()
            if need_d:
                rot = np.rot90(n.weights.reshape(self.d, self.f, self.f), k=2, axes=(1, 2))
                sum_weights.append(rot)

        if not need_d:
            return

        padding = ((self.w - 1) * self.s + self.f - self.w2) // 2
        cols = u.im2col_indices(delta, self.f, self.f, padding=padding, stride=self.s)

        sum_weights = np.array(sum_weights).transpose(1, 0, 2, 3).reshape(self.d, -1)

        result = sum_weights.dot(cols)
        im = result.reshape(self.d, self.h, self.w, -1).transpose(3, 0, 1, 2)

        return im

    def output_size(self):
        return (self.d2, self.h2, self.w2)

    def update(self, lr, l2_reg, t=0):
        super(ConvLayer, self).update(lr, l2_reg, t)
