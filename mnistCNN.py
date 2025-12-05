import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from datasets import load_dataset
import time

from nnscratch import Dense, Convolutional, Reshape, Sigmoid, Tanh, binary_cross_entropy, binary_cross_entropy_prime, mse_prime, mse, train, predict

def preprocess_data(ds, limit):
    x_list, y_list = [], []
    for i, item in enumerate(ds):
        if i >= limit:
            break
        image = np.array(item['image'])
        label = item['label']
        x = image.reshape(1, 28, 28).astype('float32') / 255
        y = np.zeros((10, 1))
        y[label] = 1
        x_list.append(x)
        y_list.append(y)
    return np.array(x_list), np.array(y_list)


# Load MNIST data
ds_train = load_dataset("ylecun/mnist", split="train")
ds_test = load_dataset("ylecun/mnist", split="test")

x_train, y_train = preprocess_data(ds_train, 10000)
x_test, y_test = preprocess_data(ds_test, 1000)

network = [
    Convolutional((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 10),
    Sigmoid()
]

start_time = time.time()
# train
train(
    network,
    mse,
    mse_prime,
    x_train,
    y_train,
    epochs=200,
    learning_rate=0.005
)
print("--- %s seconds ---" % (time.time() - start_time))

# test modified to show accuracy
def calculate_accuracy(network, x_test, y_test):
    correct_predictions = 0
    total_predictions = len(x_test)

    for x, y in zip(x_test, y_test):
        output = predict(network, x)
        predicted_label = np.argmax(output)
        true_label = np.argmax(y)

        if predicted_label == true_label:
            correct_predictions += 1

    accuracy = correct_predictions / total_predictions
    return accuracy

accuracy = calculate_accuracy(network, x_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")