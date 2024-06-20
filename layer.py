class Layer:
    def __init__(self):
        self.inputs = None
        self.outputs = None

    def forward(self, input):
        # TODO: return output
        pass

    def backward(self, output_grad, learning_rate):
        # TODO: update parameters and return input grad (backProp)
        pass