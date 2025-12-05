# Neural Network From Scratch

## Running Comparisons

To run the model comparisons, simply execute:

```
python compare.py
```

## File Structure

```
├── compare.py              # Comparison script for all models
├── mnist.py                # Simple NN for MNIST
├── mnistCNN.py             # CNN for MNIST
├── mnist_transformer.py    # Transformer for MNIST
│
└── nnscratch/              # From-scratch NN implementation
    ├── __init__.py
    ├── core/
    │   ├── layer.py        # Base Layer class
    │   └── network.py      # train() and predict() functions
    ├── layers/
    │   ├── activation.py   # Base Activation class
    │   ├── activations.py  # Tanh, Sigmoid, Softmax, ReLU
    │   ├── convolutional.py # Convolutional layer
    │   ├── dense.py        # Dense (fully connected) layer
    │   └── reshape.py      # Reshape layer
    └── losses/
        └── loss_functions.py # MSE, Binary Cross-Entropy
```

###### Generated with Cursor Opus 4.5 "In the readme just say that to run the comparions just do compare.py and then also list the general file structure for the from scratch NN and CNN"