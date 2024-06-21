import cupy as np
import cupyx.scipy.signal as signal
from layer import Layer


class Convolutional(Layer):
    def __init__(self, input_shape, kernal_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernal_size + 1, input_width - kernal_size + 1)
        self.kernal_shape = (depth, input_depth, kernal_size, kernal_size)
        self.kernals = np.random.randn(*self.kernal_shape).astype('float32')
        self.biases = np.random.randn(*self.output_shape).astype('float32')

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.copy(self.biases)  # this is just because they have the same shape
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.outputs[i] += signal.correlate2d(self.inputs[j], self.kernals[i, j], "valid")
        return self.outputs

    def backward(self, output_grad, learning_rate):
        kernels_gradient = np.zeros(self.kernals.shape)
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.inputs[j], output_grad[i], "valid")
                # notice convolve vs correlate
                input_gradient[j] = signal.convolve2d(output_grad[i], self.kernals[i, j], "full")

        self.kernals -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_grad
        return input_gradient
