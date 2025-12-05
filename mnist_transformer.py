# Claude Code Generated: With Sonnet 4.5
# Prompt 1: lets make a super simple pytorch transformer to do the same task as the mnistCNN.py file
# Prompt 2: lets make a super simple pytorch transformer to do the same task
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load MNIST data using PyTorch built-in
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

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
for epoch in range(10):
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
    print(f"Epoch {epoch+1}/10 | Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")

print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
