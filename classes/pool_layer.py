import numpy as np
from classes.layer import Layer
import classes.utils as u


class PoolLayer(Layer):
    def __init__(self, input_size, f=2, s=2, method='max', dropout=1):
        # assert f == s

        self.image_size = 0
        self.d = input_size[0]
        self.h = input_size[1]
        self.w = input_size[2]
        self.method = method
        self.argmax = None
        self.dropout = dropout

        self.f = f
        self.s = s

        assert self.h % self.f == 0
        assert self.w % self.f == 0

        self.w2 = int((self.w - self.f) / self.s + 1)
        self.h2 = int((self.h - self.f) / self.s + 1)


    def predict(self, batch):
        self.image_size = batch.shape[0]
        return self.forward(batch)[0]

    def backward(self, d, need_d=True):
        if d.ndim < 4:
            d = d.reshape(self.image_size, self.d, self.h2, self.w2)

        dout_flat = d.transpose(2, 3, 0, 1).ravel()

        if self.method == 'max':
            dX_col = np.zeros((self.f**2, self.h2 * self.w2 * self.d * self.image_size)).astype(np.float32)
            dX_col[self.argmax, range(self.argmax.size)] = dout_flat
        elif self.method == 'average':
            field_size = self.f**2
            dout_flat /= field_size
            dX_col = np.zeros((self.f**2, self.h2 * self.w2 * self.d * self.image_size)).astype(np.float32)
            dX_col[range(field_size), :] = dout_flat
        else:
            raise ValueError('pool layer type error!')

        dX = u.col2im_indices(dX_col, (self.image_size * self.d, 1, self.h, self.w), self.f, self.f, padding=0, stride=self.s)
        return dX.reshape(self.image_size, self.d, self.h, self.w)

    def forward(self, batch):
        self.image_size = batch.shape[0]
        batch_reshaped = batch.reshape(self.image_size * self.d, 1, self.h, self.w)
        X_col = u.im2col_indices(batch_reshaped, self.f, self.f, padding=0, stride=self.s)

        if self.method == 'max':
            self.argmax = np.argmax(X_col, axis=0)
            out = X_col[self.argmax, range(self.argmax.size)]
        elif self.method == 'average':
            out = np.average(X_col, axis=0)
        else:
            raise ValueError('pool layer type error!')

        return out.reshape(self.h2, self.w2, self.image_size, self.d).transpose(2, 3, 0, 1), 0

    def update(self, lr, l2_reg, t=0):
        pass

    def output_size(self):
        return (self.d, self.h2, self.w2)
