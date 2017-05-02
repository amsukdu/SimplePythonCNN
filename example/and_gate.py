import sys, os
sys.path.insert(1, os.path.split(sys.path[0])[0])
import numpy as np
from classes.neural_net import NeuralNetwork


cnn = NeuralNetwork(2,
                    [
                        {'type': 'output', 'k': 1, 'u_type': 'v', 'a_type': 'sigmoid'},
                    ]
                    , 0.001, loss='logistic')

input = np.array([[0,0], [0,1], [1,0], [1,1]]).reshape(4,1,2)
output = [0,1,1,1]

for i in range(9999999):
    loss, acc = cnn.epoch(input, output)
    print loss
    print acc