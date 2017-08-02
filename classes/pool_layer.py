import numpy as np
from classes.layer import Layer
import classes.utils as u


# TODO only not overlapping f & s works...
class PoolLayer(Layer):
    def __init__(self, input_size, f=2, s=2):

        assert f == s

        self.image_size = 0
        self.d = input_size[0]
        self.h = input_size[1]
        self.w = input_size[2]
        self.argmax = None

        self.f = f
        self.s = s

        assert self.h % self.f == 0
        assert self.w % self.f == 0

        self.w2 = int((self.w - self.f) / self.s + 1)
        self.h2 = int((self.h - self.f) / self.s + 1)

        self.indices = []

        field_size = self.f * self.f

        offset = 0
        i_offset = 0
        for i in range(self.h):
            if i % self.f == 0:
                start = self.w * i
                offset = start
                i_offset = i
            else:
                start = self.f * (i - i_offset) + offset

            for j in range(int(self.w / self.f)):
                self.indices += range(start, start + self.f)
                start += field_size



    def predict(self, batch):
        self.image_size = batch.shape[0]
        return self.forward(batch)[0]

    def backward(self, d, need_d=True):
        if d.ndim < 4:
            d = d.reshape(self.image_size, self.d, self.h2, self.w2)

        temp = d.reshape(-1, self.d, self.h2*self.w2, 1)
        result = (temp * self.argmax).reshape(d.shape[0], self.d, -1)
        result = result[:, :, self.indices].reshape(d.shape[0], self.d, self.h, self.w)

        return result

    def forward(self, batch):
        self.image_size = batch.shape[0]
        k, i, j = u.get_im2col_indices(batch.shape, self.f, self.f, 0, self.s)
        cols = batch[:, k, i, j].reshape(batch.shape[0], self.d, -1, self.h2 * self.w2).transpose(0, 1, 3, 2)
        temp = np.amax(cols, axis=3, keepdims=True)
        self.argmax = cols == temp
        result = temp.reshape(self.image_size, self.d, self.h2, self.w2)

        return (result, 0)

    def update(self, lr, l2_reg, t=0):
        pass

    def output_size(self):
        return (self.d, self.h2, self.w2)
