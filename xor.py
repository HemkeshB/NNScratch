from dense import Dense
from activations import Tanh
from loss_functions import mse, mse_prime
import numpy as np


X = np.reshape([[0,0],[0,1],[1,0],[1,1]], (4,2,1))
Y = np.reshape([[0],[1],[1],[0]], (4,1,1))

neural_network = [
    Dense(2,3),
    Tanh(),
    Dense(3,1),
    Tanh()
]

epochs = 1000
learning_rate = 0.1
# Training loop
for epoch in range(epochs):
    # Not needed for the actual training but usefull to print/show
    error = 0
    for x, y in zip(X,Y):
        # Going through the NN (forward)
        output = x
        for layer in neural_network:
            output = layer.forward(output)

        # Calculating error unneeded
        error = error + mse(y, output)

        # Going backward and making adjustments (backward)
        grad = mse_prime(y, output)
        for layer in reversed(neural_network):
            grad = layer.backward(grad,learning_rate)

    error = error / len(X)
    print('%d/%d, error = %f' % (epoch + 1, epochs, error))
