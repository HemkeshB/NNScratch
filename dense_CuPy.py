from layer import Layer
import cupy as np

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size).astype('float32')
        self.biases = np.random.randn(output_size, 1).astype('float32')

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.biases

    def backward(self, output_grad, learning_rate):
        weights_grad = np.dot(output_grad, self.input.T)

        # Needs to be computed before updated in the next lines
        input_grad = np.dot(self.weights.T, output_grad)

        self.weights -= learning_rate * weights_grad
        self.biases -= learning_rate * output_grad
        return input_grad