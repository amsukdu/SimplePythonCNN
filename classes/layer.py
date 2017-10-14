from abc import ABCMeta, abstractmethod


class Layer(object):
    __metaclass__ = ABCMeta

    is_output = False
    activation = True
    u_type = 'adam'
    dropout = 1

    @abstractmethod
    def forward(self, batch):
        pass

    @abstractmethod
    def backward(self, d, need_d=True):
        pass

    @abstractmethod
    def output_size(self):
        pass

    @abstractmethod
    def update(self, lr, l2_reg, t=0):
        pass

    @abstractmethod
    def predict(self, batch):
        pass
