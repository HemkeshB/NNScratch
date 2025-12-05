import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from datasets import load_dataset

# Import your custom classes and functions
from nnscratch import Dense, Tanh, mse, mse_prime, train, predict

# Preprocess data function
def preprocess_data(ds, limit):
    x_list, y_list = [], []
    for i, item in enumerate(ds):
        if i >= limit:
            break
        image = np.array(item['image'])
        label = item['label']
        x = image.reshape(28 * 28, 1).astype("float32") / 255
        y = np.zeros((10, 1))
        y[label] = 1
        x_list.append(x)
        y_list.append(y)
    return np.array(x_list), np.array(y_list)

# Load MNIST data
ds_train = load_dataset("ylecun/mnist", split="train")
ds_test = load_dataset("ylecun/mnist", split="test")

x_train, y_train = preprocess_data(ds_train, 1000)
x_test, y_test = preprocess_data(ds_test, 20)

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