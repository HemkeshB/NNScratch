
import numpy as np
import cupy as xp
from keras._tf_keras.keras.datasets import mnist
from keras._tf_keras.keras.utils import to_categorical

import time  # testing

from dense_CuPy import Dense
from convolutional_CuPy import Convolutional
from reshape_CuPy import Reshape
from activations_CuPy import Sigmoid, Tanh
from loss_functions_CuPy import binary_cross_entropy, binary_cross_entropy_prime, mse_prime, mse
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
    y = to_categorical(y).astype('float32')
    y = y.reshape(len(y), 3, 1)
    return x, y


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preproces_data(x_train, y_train, 1000)
x_test, y_test = preproces_data(x_test, y_test, 100)
x_train = xp.asarray(x_train, dtype='float32')
x_test = xp.asarray(x_test, dtype='float32')
y_train = xp.asarray(y_train, dtype='float32')
y_test = xp.asarray(y_test, dtype='float32')

network = [
    Convolutional((1, 28, 28), 3, 5),
    Sigmoid(),
    Reshape((5, 26, 26), (5 * 26 * 26, 1)),
    Dense(5 * 26 * 26, 100),
    Sigmoid(),
    Dense(100, 3),
    Sigmoid()
    # Dense(28 * 28, 40),
    # Tanh(),
    # Dense(40, 10),
    # Tanh()
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

# # test
# for x, y in zip(x_test, y_test):
#     output = predict(network, x)
#     print(f"pred: {np.argmax(output)}, true: {np.argmax(y)}")