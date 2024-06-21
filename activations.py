from activation import Activation
import numpy as np


class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2
        super().__init__(tanh, tanh_prime)


class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)


class Relu(Activation):
    def __init__(self):
        relu = lambda x: np.where(x > 0, x, 0)
        relu_prime = lambda x: 1. * (x > 0)
        super().__init__(relu, relu_prime)
