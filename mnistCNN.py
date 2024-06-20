import numpy as np
from keras._tf_keras.keras.datasets import mnist
from keras._tf_keras.keras.utils import to_categorical

from dense import Dense
from convolutional import Convolutional
from reshape import Reshape
from activations import Sigmoid, Tanh
from loss_functions import binary_cross_entropy, binary_cross_entropy_prime
from network import train, predict


def to_one_hot(y_value):
    num_classes = np.max(y_value) + 1
    one_hot_matrix = np.eye(num_classes)[y_value.reshape(-1)]
    return one_hot_matrix.reshape(list(y_value.shape) + [num_classes])

def preproces_data(x, y, limit):
    zero_index = np.where(y == 0)[0][:limit]
    one_index = np.where(y == 1)[0][:limit]
    all_indices = np.hstack((zero_index, one_index))
    all_indices = np.random.permutation(all_indices)
    x, y = x[all_indices], y[all_indices]
    x = x.reshape(len(x), 1, 28, 28)
    x = x.astype('float32') / 255
    y = to_categorical(y)

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
    Dense(100, 2),
    Sigmoid()
]

# train
train(
    network,
    binary_cross_entropy,
    binary_cross_entropy_prime,
    x_train,
    y_train,
    epochs=20,
    learning_rate=0.1
)

# test
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")