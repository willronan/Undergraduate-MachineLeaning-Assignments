import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10

# prepare a transform to normalize the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# import the cifar10 dataset
cifar10_full = CIFAR10(root='root', train=True, download=True, transform=transform)


num_classes = 10

# split data into train, val and test sets
train_size = int(0.8 * len(cifar10_full))
val_size = int(0.1 * len(cifar10_full))
test_size = len(cifar10_full) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(cifar10_full, [train_size, val_size, test_size])

# loaders for iterating through datasets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# initialize the vgg model
vgg11_model = models.vgg11(pretrained=True)

# modify the output layer to have 10 classes
vgg11_model.classifier[6] = nn.Linear(4096, num_classes)

# define cost function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg11_model.parameters(), lr=0.001, momentum=0.9)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg11_model.to(device)
num_epochs = 10

# perform training and validation
for epoch in range(num_epochs):
    vgg11_model.train()
    running_loss = 0.0

    #
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = vgg11_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}")

    vgg11_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = vgg11_model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy}")


torch.save(vgg11_model.state_dict(), 'vgg11_cifar10.pth')