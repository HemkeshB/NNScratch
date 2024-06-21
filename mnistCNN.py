import numpy as np
from keras._tf_keras.keras.datasets import mnist
from keras._tf_keras.keras.utils import to_categorical
import time

from dense import Dense
from convolutional import Convolutional
from reshape import Reshape
from activations import Sigmoid, Tanh
from loss_functions import binary_cross_entropy, binary_cross_entropy_prime, mse_prime, mse
from network import train, predict

def preproces_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    two_index = np.where(y == 2)[0][:limit]
    all_indices = np.hstack((zero_index, one_index, two_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype('float32') / 255
    y = to_categorical(y)
    y = y.reshape(len(y), 3, 1)
    return x, y


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preproces_data(x_train, y_train, 100)
x_test, y_test = preproces_data(x_test, y_test, 100)

network = [
    Convolutional((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 3),
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
    epochs=100,
    learning_rate=0.01
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