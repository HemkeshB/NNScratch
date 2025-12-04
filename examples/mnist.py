import numpy as np
from keras._tf_keras.keras.datasets import mnist
from keras._tf_keras.keras.utils import to_categorical

from nnscratch.layers.dense import Dense
from nnscratch.layers.activations import Tanh
from nnscratch.losses.loss_functions import mse, mse_prime
from nnscratch.core.network import train, predict

# Preprocess data function
def preprocess_data(x, y, limit):
    x = x.reshape(x.shape[0], 28 * 28, 1)  # Flatten images and add channel dimension
    x = x.astype("float32") / 255  # Normalize pixel values
    y = to_categorical(y)  # One-hot encode labels
    y = y.reshape(y.shape[0], 10, 1)  # Reshape to (N, 10, 1) to match the network output format
    return x[:limit], y[:limit]

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 1000)
x_test, y_test = preprocess_data(x_test, y_test, 20)

# Define the neural network
network = [
    Dense(28 * 28, 40),
    Tanh(),
    Dense(40, 10),
    Tanh()
]

# Train the network
train(network, mse, mse_prime, x_train, y_train, epochs=100, learning_rate=0.1)

# Test the network
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print('pred:', np.argmax(output), '\ttrue:', np.argmax(y))

