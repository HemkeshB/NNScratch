# Claude Code Generated: With Sonnet 4.5
# Prompt 1: lets make a super simple pytorch transformer to do the same task as the mnistCNN.py file
# Prompt 2: lets make a super simple pytorch transformer to do the same task

# OpenCode Code Generated: With Sonnet 4.5
#prompt 3 on plan: look at the current repo lets look at all three mnist files and lets make them all use the same data set with the same amount of training and testing
#prompt 4 on build: Make the changes
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load MNIST data using Hugging Face datasets
def preprocess_data(ds, limit):
    x_list, y_list = [], []
    for i, item in enumerate(ds):
        if i >= limit:
            break
        image = np.array(item['image'])
        label = item['label']
        x = image.astype("float32") / 255.0
        x_list.append(x)
        y_list.append(label)
    return np.array(x_list), np.array(y_list)

ds_train = load_dataset("ylecun/mnist", split="train")
ds_test = load_dataset("ylecun/mnist", split="test")

x_train, y_train = preprocess_data(ds_train, 10000)
x_test, y_test = preprocess_data(ds_test, 1000)

# Convert to PyTorch tensors
train_dataset = TensorDataset(
    torch.FloatTensor(x_train), 
    torch.LongTensor(y_train)
)
test_dataset = TensorDataset(
    torch.FloatTensor(x_test), 
    torch.LongTensor(y_test)
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Simple Transformer model
class SimpleTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(1, 64)
        self.pos_encoding = nn.Parameter(torch.randn(1, 784, 64))
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(x.size(0), 784, 1)  # (batch, 784, 1)
        x = self.embedding(x) + self.pos_encoding
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.fc(x)

model = SimpleTransformer().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
print("Training...")
test_acc = 0.0
for epoch in range(200):
    model.train()
    correct, total = 0, 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    train_acc = 100. * correct / total

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    test_acc = 100. * correct / total
    print(f"Epoch {epoch+1}/200 | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
