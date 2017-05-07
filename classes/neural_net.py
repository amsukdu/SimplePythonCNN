from neural_layer import NeuralLayer
from conv_layer import ConvLayer
from pool_layer import PoolLayer
import numpy as np
import utils

class NeuralNetwork(object):
    def __init__(self, input_shape, layer_list, lr, l2_reg=0, dropout_p=1, loss='softmax'):
        self.layers = []
        self.lr = np.float32(lr)
        self.l2_reg = np.float32(l2_reg)
        self.loss = loss

        # dropout
        self.dropout_p = dropout_p
        self.dropout_masks = []
        self.t = 0

        next_input_size = input_shape
        for l in layer_list:
            if l['type'] == 'conv':
                l.pop('type')
                conv = ConvLayer(next_input_size, **l)
                self.layers.append(conv)
                next_input_size = conv.output_size()

            elif l['type'] == 'pool':
                l.pop('type')
                pool = PoolLayer(next_input_size, **l)
                self.layers.append(pool)
                next_input_size = pool.output_size()

            elif l['type'] == 'fc':
                l.pop('type')
                fc = NeuralLayer(next_input_size, **l)
                self.layers.append(fc)
                next_input_size = fc.output_size()

            elif l['type'] == 'output':
                l.pop('type')
                fc = NeuralLayer(next_input_size, **l)
                fc.is_output = True
                fc.activation = False
                self.layers.append(fc)
                next_input_size = fc.output_size()


    def predict(self, batch, label):
        next_input = batch
        for index, layer in enumerate(self.layers):
            next_input = layer.predict(next_input)

            if (self.dropout_p < 1) and (type(layer).__name__ == 'NeuralLayer') and not layer.is_output:
                next_input *= self.dropout_p

        result = np.array(next_input)
        if self.loss == 'softmax':
            loss, delta = utils.softmax_loss(result, label)
        elif self.loss == 'logistic':
            loss, delta = utils.logistic_loss(result, label)

        max_result = np.argmax(result, axis=0)
        correct_count = np.sum(max_result == label)

        return loss, correct_count / float(len(max_result)) * 100

    def epoch(self, batch, label):
        # forward
        self.dropout_masks = []
        l2 = 0
        next_input = batch
        for index, layer in enumerate(self.layers):
            layer_result = layer.forward(next_input)
            next_input = layer_result[0]
            l2 += layer_result[1]
            if (self.dropout_p < 1) and (type(layer).__name__ == 'NeuralLayer') and not layer.is_output:
                dropout_mask = np.random.rand(*next_input.shape) < self.dropout_p
                self.dropout_masks.append(dropout_mask)
                next_input *= dropout_mask

        result = np.array(next_input)
        if self.loss == 'softmax':
            loss, delta = utils.softmax_loss(result, label)
        elif self.loss == 'logistic':
            loss, delta = utils.logistic_loss(result, label)

        max_result = np.argmax(result, axis=0)
        correct_count = np.sum(max_result == label)

        # backprop
        back_input = delta.T
        for index, layer in enumerate(reversed(self.layers)):
            is_input_layer = index < len(self.layers) - 1
            back_input = layer.backward(back_input, is_input_layer)

            if self.dropout_masks:
                dropout_mask = self.dropout_masks.pop()
                back_input *= dropout_mask


        # update
        for index, layer in enumerate(self.layers):
            layer.update(self.lr, l2_reg=self.l2_reg, t=self.t)


        return loss + self.l2_reg * l2 / 2, correct_count / float(len(max_result)) * 100