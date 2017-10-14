from classes.neural_layer import NeuralLayer
from classes.conv_layer import ConvLayer
from classes.pool_layer import PoolLayer
import classes.utils as utils
import numpy as np

class NeuralNetwork(object):
    def __init__(self, input_shape, layer_list, lr, l2_reg=0, loss='softmax'):
        self.layers = []
        self.lr = np.float32(lr)
        self.l2_reg = np.float32(l2_reg)
        self.loss = loss
        self.epoch_count = 0

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
        l2 = 0
        next_input = batch
        for index, layer in enumerate(self.layers):
            layer_result = layer.forward(next_input)
            next_input = layer_result[0]
            l2 += layer_result[1]
            if layer.dropout < 1 and not layer.is_output:
                dropout_mask = np.random.rand(*next_input.shape) < layer.dropout
                next_input *= dropout_mask / layer.dropout
                self.dropout_masks.append(dropout_mask)


        result = np.array(next_input)
        if self.loss == 'softmax':
            loss, delta = utils.softmax_loss(result, label)
        elif self.loss == 'logistic':
            loss, delta = utils.logistic_loss(result, label)

        loss += 0.5 * self.l2_reg * l2
        max_result = np.argmax(result, axis=0)
        correct_count = np.sum(max_result == label)

        # backprop
        back_input = delta.T
        for index, layer in enumerate(reversed(self.layers)):
            is_input_layer = index < len(self.layers) - 1

            if layer.dropout < 1 and not layer.is_output and self.dropout_masks:
                dropout_mask = self.dropout_masks.pop()
                if dropout_mask.ndim > 2 and back_input.ndim == 2:
                    back_input *= dropout_mask.T.reshape(-1, back_input.shape[1])
                else:
                    back_input *= dropout_mask

            back_input = layer.backward(back_input, is_input_layer)

        # update
        for index, layer in enumerate(self.layers):
            layer.update(self.lr, l2_reg=self.l2_reg, t=self.t)

        return loss + self.l2_reg * l2 / 2, correct_count / float(len(max_result)) * 100
