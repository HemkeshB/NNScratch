"""
MNIST Model Comparison Script
Trains and compares Dense, CNN, and Transformer networks on identical MNIST data

OpenCode Code Generated: With Sonnet 4.5
Prompt 1 plan: Lets make the compare method intialize the model on its own and then have it train the 3 models on the same data one after another with the variable at the top of the file for train params so this is the comparison script with little differences, also lets seed it on both numpy and pytorch to 598        
Prompt 2 plan: 1. lets do the 100 epochs for all, 2. different loss functions is fine, 3. lets do verbose output, 4. lets do total params, along with test acc, 5. do a results.json 6. also have a boolean varible which will store the models in a saved_models/ folder 
Prompt 3 build: lets build the implementation
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
import time
import json
from datetime import datetime
import pickle

# Custom nnscratch imports
from nnscratch import (
    Dense, Convolutional, Reshape, 
    Sigmoid, Tanh,
    mse, mse_prime,
    train, predict
)

# ============ CONFIGURATION ============
SEED = 598
TRAIN_SIZE = 40000
TEST_SIZE = 4000
BATCH_SIZE = 64  # For transformer

# Training parameters - ALL MODELS USE 100 EPOCHS
DENSE_CONFIG = {
    'epochs': 50,
    'learning_rate': 0.01
}

CNN_CONFIG = {
    'epochs': 50,
    'learning_rate': 0.01
}

TRANSFORMER_CONFIG = {
    'epochs': 50,
    'learning_rate': 0.001,
    'batch_size': 64
}

# Output control
VERBOSE = True  # Show training progress
SAVE_MODELS = False  # Toggle to save trained models
RESULTS_FILE = 'results.json'
MODELS_DIR = 'saved_models'

# Device setup for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============ SEED MANAGEMENT ============
def set_all_seeds(seed):
    """Set seeds for reproducibility across NumPy and PyTorch"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"All seeds set to: {seed}")


# ============ DATA LOADING & PREPROCESSING ============
def load_mnist_data(train_size, test_size):
    """Load MNIST data from Hugging Face datasets"""
    print(f"\nLoading MNIST data...")
    print(f"  Train samples: {train_size}")
    print(f"  Test samples: {test_size}")
    
    ds_train = load_dataset("ylecun/mnist", split="train")
    ds_test = load_dataset("ylecun/mnist", split="test")
    
    return ds_train, ds_test


def preprocess_for_dense(ds, limit):
    """
    Preprocess for Dense network: (784, 1) shape, one-hot labels (10, 1)
    """
    x_list, y_list = [], []
    for i, item in enumerate(ds):
        if i >= limit:
            break
        image = np.array(item['image'])
        label = item['label']
        
        # Flatten to (784, 1)
        x = image.reshape(28 * 28, 1).astype("float32") / 255
        
        # One-hot encode label
        y = np.zeros((10, 1))
        y[label] = 1
        
        x_list.append(x)
        y_list.append(y)
    
    return np.array(x_list), np.array(y_list)


def preprocess_for_cnn(ds, limit):
    """
    Preprocess for CNN: (1, 28, 28) shape, one-hot labels (10, 1)
    """
    x_list, y_list = [], []
    for i, item in enumerate(ds):
        if i >= limit:
            break
        image = np.array(item['image'])
        label = item['label']
        
        # Reshape to (1, 28, 28) for convolution
        x = image.reshape(1, 28, 28).astype('float32') / 255
        
        # One-hot encode label
        y = np.zeros((10, 1))
        y[label] = 1
        
        x_list.append(x)
        y_list.append(y)
    
    return np.array(x_list), np.array(y_list)


def preprocess_for_transformer(ds, limit):
    """
    Preprocess for Transformer: (28, 28) shape, integer labels
    Returns numpy arrays to be converted to tensors
    """
    x_list, y_list = [], []
    for i, item in enumerate(ds):
        if i >= limit:
            break
        image = np.array(item['image'])
        label = item['label']
        
        # Keep as (28, 28), normalize
        x = image.astype("float32") / 255.0
        
        x_list.append(x)
        y_list.append(label)
    
    return np.array(x_list), np.array(y_list)


def create_torch_dataloaders(x_train, y_train, x_test, y_test, batch_size):
    """Create PyTorch DataLoaders from numpy arrays"""
    train_dataset = TensorDataset(
        torch.FloatTensor(x_train),
        torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(x_test),
        torch.LongTensor(y_test)
    )
    
    # No shuffle for reproducibility with seed
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


# ============ PARAMETER COUNTING ============
def count_custom_network_params(network):
    """
    Count parameters in custom nnscratch networks
    Returns total trainable parameters
    """
    total_params = 0
    
    for layer in network:
        # Dense layer: weights + biases
        if hasattr(layer, 'weights') and hasattr(layer, 'biases'):
            total_params += layer.weights.size + layer.biases.size
        
        # Convolutional layer: kernels + biases
        elif hasattr(layer, 'kernals') and hasattr(layer, 'biases'):
            total_params += layer.kernals.size + layer.biases.size
    
    return total_params


def count_pytorch_params(model):
    """
    Count parameters in PyTorch model
    Returns total trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============ MODEL INITIALIZATION ============
def create_dense_network():
    """
    Create Dense Network architecture
    Architecture:
      Dense(784 -> 40) -> Tanh
      Dense(40 -> 10) -> Tanh
    """
    network = [
        Dense(28 * 28, 40),
        Tanh(),
        Dense(40, 10),
        Tanh()
    ]
    return network


def create_cnn_network():
    """
    Create CNN architecture
    Architecture:
      Conv(1x28x28 -> 5x26x26, kernel=3) -> Sigmoid
      Reshape(5x26x26 -> 3380x1)
      Dense(3380 -> 100) -> Sigmoid
      Dense(100 -> 10) -> Sigmoid
    """
    network = [
        Convolutional((1, 28, 28), 3, 10),
        Sigmoid(),
        Reshape((10, 26, 26), (10 * 26 * 26, 1)),
        Dense(10 * 26 * 26, 100),
        Sigmoid(),
        Dense(100, 10),
        Sigmoid()
    ]
    return network


class SimpleTransformer(nn.Module):
    """
    Simple Transformer for MNIST classification
    Architecture:
      Embedding: 1 -> 64
      Positional Encoding: 784 x 64
      Transformer Encoder: 2 layers, 8 heads
      Output: 64 -> 10
    """
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(1, 64)
        self.pos_encoding = nn.Parameter(torch.randn(1, 784, 64))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, 
            nhead=8, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), 784, 1)  # (batch, 784, 1)
        x = self.embedding(x) + self.pos_encoding
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)


def create_transformer_network():
    """Create and return Transformer model"""
    model = SimpleTransformer().to(device)
    return model


# ============ TRAINING & EVALUATION ============
def train_custom_network(network, x_train, y_train, x_test, y_test, config, model_name):
    """
    Train custom nnscratch network and evaluate
    
    Args:
        network: List of layers
        x_train, y_train: Training data
        x_test, y_test: Test data
        config: Dict with 'epochs' and 'learning_rate'
        model_name: String name for display
    
    Returns:
        Dict with results: accuracy, time, params
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Count parameters
    total_params = count_custom_network_params(network)
    trainable_layers = len([l for l in network if hasattr(l, 'weights') or hasattr(l, 'kernals')])
    
    print(f"Architecture: {trainable_layers} trainable layers")
    print(f"Total parameters: {total_params:,}")
    print(f"Epochs: {config['epochs']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print()
    
    # Train
    start_time = time.time()
    train(
        network, 
        mse, 
        mse_prime, 
        x_train, 
        y_train,
        epochs=config['epochs'],
        learning_rate=config['learning_rate'],
        verbose=VERBOSE
    )
    elapsed_time = time.time() - start_time
    
    print(f"\nTraining completed in {elapsed_time:.2f} seconds")
    
    # Evaluate
    print("Evaluating on test set...")
    correct = 0
    total = len(x_test)
    
    for x, y in zip(x_test, y_test):
        output = predict(network, x)
        pred_label = np.argmax(output)
        true_label = np.argmax(y)
        if pred_label == true_label:
            correct += 1
    
    accuracy = (correct / total) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    return {
        'model': model_name,
        'accuracy': accuracy,
        'time': elapsed_time,
        'params': total_params,
        'epochs': config['epochs'],
        'learning_rate': config['learning_rate']
    }


def train_transformer_network(train_loader, test_loader, config, model_name):
    """
    Train PyTorch Transformer and evaluate
    
    Args:
        train_loader: PyTorch DataLoader for training
        test_loader: PyTorch DataLoader for testing
        config: Dict with training params
        model_name: String name for display
    
    Returns:
        Dict with results and trained model
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")
    
    # Create model
    model = create_transformer_network()
    
    # Count parameters
    total_params = count_pytorch_params(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Epochs: {config['epochs']}")
    print(f"Learning rate: {config['learning_rate']}")
    print(f"Batch size: {config['batch_size']}")
    print()
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    start_time = time.time()
    
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_correct, train_total = 0, 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)
        
        train_acc = 100. * train_correct / train_total
        
        if VERBOSE:
            print(f"Epoch {epoch+1}/{config['epochs']} | Train Acc: {train_acc:.2f}%")
    
    elapsed_time = time.time() - start_time
    print(f"\nTraining completed in {elapsed_time:.2f} seconds")
    
    # Final evaluation on test set
    print("Evaluating on test set...")
    model.eval()
    test_correct, test_total = 0, 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            test_correct += predicted.eq(labels).sum().item()
            test_total += labels.size(0)
    
    test_acc = 100. * test_correct / test_total
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    return {
        'model': model_name,
        'accuracy': test_acc,
        'time': elapsed_time,
        'params': total_params,
        'epochs': config['epochs'],
        'learning_rate': config['learning_rate'],
        'pytorch_model': model  # For saving
    }


# ============ MODEL SAVING ============
def save_models(results, dense_net, cnn_net):
    """
    Save trained models to saved_models/ directory
    
    Args:
        results: List of result dicts
        dense_net: Trained dense network
        cnn_net: Trained CNN network
        
    Note: Transformer model is in results[2]['pytorch_model']
    """
    if not SAVE_MODELS:
        return
    
    # Create directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print(f"\n{'='*60}")
    print("Saving models...")
    print(f"{'='*60}")
    
    # Save Dense network
    dense_path = os.path.join(MODELS_DIR, f'dense_network_{timestamp}.pkl')
    with open(dense_path, 'wb') as f:
        pickle.dump(dense_net, f)
    print(f"Dense Network saved to: {dense_path}")
    
    # Save CNN network
    cnn_path = os.path.join(MODELS_DIR, f'cnn_network_{timestamp}.pkl')
    with open(cnn_path, 'wb') as f:
        pickle.dump(cnn_net, f)
    print(f"CNN Network saved to: {cnn_path}")
    
    # Save Transformer (PyTorch)
    transformer_result = [r for r in results if 'Transformer' in r['model']][0]
    if 'pytorch_model' in transformer_result:
        transformer_path = os.path.join(MODELS_DIR, f'transformer_network_{timestamp}.pt')
        torch.save(transformer_result['pytorch_model'].state_dict(), transformer_path)
        print(f"Transformer Network saved to: {transformer_path}")
    
    print(f"All models saved to: {MODELS_DIR}/")


# ============ RESULTS DISPLAY & STORAGE ============
def print_comparison_table(results):
    """
    Print formatted comparison table
    
    Table columns:
      - Model name
      - Total Parameters
      - Test Accuracy
      - Training Time
    """
    print(f"\n{'='*70}")
    print("COMPARISON RESULTS")
    print(f"{'='*70}")
    print(f"\n{'Model':<25} {'Params':<15} {'Accuracy':<15} {'Time (s)':<15}")
    print("-"*70)
    
    for r in results:
        model_name = r['model']
        params = f"{r['params']:,}"
        accuracy = f"{r['accuracy']:.2f}%"
        time_str = f"{r['time']:.2f}"
        
        print(f"{model_name:<25} {params:<15} {accuracy:<15} {time_str:<15}")
    
    print("-"*70)
    
    # Find best model
    best_acc = max(results, key=lambda x: x['accuracy'])
    fastest = min(results, key=lambda x: x['time'])
    smallest = min(results, key=lambda x: x['params'])
    
    print(f"\nBest Accuracy:    {best_acc['model']} ({best_acc['accuracy']:.2f}%)")
    print(f"Fastest Training: {fastest['model']} ({fastest['time']:.2f}s)")
    print(f"Smallest Model:   {smallest['model']} ({smallest['params']:,} params)")


def save_results_json(results):
    """
    Save results to results.json
    
    Saves:
      - Timestamp
      - Configuration used
      - Results for each model
    """
    output_data = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'seed': SEED,
            'train_size': TRAIN_SIZE,
            'test_size': TEST_SIZE,
            'epochs': {
                'dense': DENSE_CONFIG['epochs'],
                'cnn': CNN_CONFIG['epochs'],
                'transformer': TRANSFORMER_CONFIG['epochs']
            },
            'learning_rates': {
                'dense': DENSE_CONFIG['learning_rate'],
                'cnn': CNN_CONFIG['learning_rate'],
                'transformer': TRANSFORMER_CONFIG['learning_rate']
            }
        },
        'results': []
    }
    
    # Clean results (remove pytorch model object)
    for r in results:
        clean_result = {k: v for k, v in r.items() if k != 'pytorch_model'}
        output_data['results'].append(clean_result)
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {RESULTS_FILE}")


# ============ MAIN EXECUTION ============
def main():
    """
    Main execution flow:
      1. Set seeds
      2. Load and preprocess data (once)
      3. Train Dense network
      4. Train CNN network
      5. Train Transformer network
      6. Display comparison
      7. Save results
      8. Optionally save models
    """
    print("="*70)
    print("MNIST MODEL COMPARISON")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print(f"Seed: {SEED}")
    print(f"All models trained for {DENSE_CONFIG['epochs']} epochs")
    print("="*70)
    
    # Set global seeds
    set_all_seeds(SEED)
    
    # Load raw data once
    ds_train, ds_test = load_mnist_data(TRAIN_SIZE, TEST_SIZE)
    
    # Preprocess for each model type
    print("\nPreprocessing data for each model...")
    x_train_dense, y_train_dense = preprocess_for_dense(ds_train, TRAIN_SIZE)
    x_test_dense, y_test_dense = preprocess_for_dense(ds_test, TEST_SIZE)
    print(f"  Dense: x_train={x_train_dense.shape}, y_train={y_train_dense.shape}")
    
    x_train_cnn, y_train_cnn = preprocess_for_cnn(ds_train, TRAIN_SIZE)
    x_test_cnn, y_test_cnn = preprocess_for_cnn(ds_test, TEST_SIZE)
    print(f"  CNN:   x_train={x_train_cnn.shape}, y_train={y_train_cnn.shape}")
    
    x_train_tf, y_train_tf = preprocess_for_transformer(ds_train, TRAIN_SIZE)
    x_test_tf, y_test_tf = preprocess_for_transformer(ds_test, TEST_SIZE)
    train_loader, test_loader = create_torch_dataloaders(
        x_train_tf, y_train_tf, x_test_tf, y_test_tf, 
        TRANSFORMER_CONFIG['batch_size']
    )
    print(f"  Transformer: x_train={x_train_tf.shape}, y_train={y_train_tf.shape}")
    
    # Store results
    results = []
    
    # ============ TRAIN DENSE NETWORK ============
    set_all_seeds(SEED)  # Reset seed for fair initialization
    dense_network = create_dense_network()
    dense_results = train_custom_network(
        dense_network,
        x_train_dense, y_train_dense,
        x_test_dense, y_test_dense,
        DENSE_CONFIG,
        "Dense Network"
    )
    results.append(dense_results)
    
    # ============ TRAIN CNN NETWORK ============
    set_all_seeds(SEED)  # Reset seed for fair initialization
    cnn_network = create_cnn_network()
    cnn_results = train_custom_network(
        cnn_network,
        x_train_cnn, y_train_cnn,
        x_test_cnn, y_test_cnn,
        CNN_CONFIG,
        "CNN Network"
    )
    results.append(cnn_results)
    
    # ============ TRAIN TRANSFORMER NETWORK ============
    set_all_seeds(SEED)  # Reset seed for fair initialization
    transformer_results = train_transformer_network(
        train_loader,
        test_loader,
        TRANSFORMER_CONFIG,
        "Transformer Network"
    )
    results.append(transformer_results)
    
    # ============ DISPLAY & SAVE RESULTS ============
    print_comparison_table(results)
    save_results_json(results)
    
    # Optionally save models
    if SAVE_MODELS:
        save_models(results, dense_network, cnn_network)
    
    print(f"\n{'='*70}")
    print(f"Comparison complete!")
    print(f"Ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
