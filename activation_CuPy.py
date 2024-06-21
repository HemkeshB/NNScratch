from layer import Layer
import cupy as np


class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_grad, learning_rate):
        return np.multiply(output_grad, self.activation_prime(self.input))
