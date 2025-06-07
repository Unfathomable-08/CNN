import torch
from torch.utils.data import DataLoader, random_split, Dataset
import os
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class CustomDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_folder):
                continue
            for img_name in os.listdir(cls_folder):
                if img_name.lower().endswith('.jpg'):
                    self.image_paths.append(os.path.join(cls_folder, img_name))
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('L')  # grayscale
        image = image.resize((64, 64))
        image = np.array(image).astype(np.float32) / 255.0  # scale to [0,1]
        image = (image - 0.5) / 0.5  # normalize to [-1,1]
        image = torch.tensor(image).unsqueeze(0)  # [1, 64, 64]

        return image, torch.tensor(label).long()

# Usage:
dataset = CustomDataset('./dataset')

train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

print("Classes:", dataset.classes)
print(f"Train samples: {len(train_dataset)} | Test samples: {len(test_dataset)}")



class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.bn4 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 8 * 8)
        x = self.dropout(F.relu(self.bn4(self.fc1(x))))
        x = self.fc2(x)
        return x


def train(model, train_loader, criterion, optimizer, device, epochs=10):
    model.to(device)
    model.train()  # set model to training mode
    i = 0
    
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()           # clear gradients
            outputs = model(inputs)         # forward pass
            loss = criterion(outputs, labels)  # compute loss
            
            loss.backward()                 # backward pass
            optimizer.step()                # update weights
            
            running_loss += loss.item() * inputs.size(0)
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            print(epoch + i)
            i = i + 1
        
        scheduler.step()

        epoch_loss = running_loss / total
        epoch_acc = correct / total * 100
        
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.2f}%")


model = CNN(num_classes=36)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train(model, train_loader, criterion, optimizer, device, epochs=10)


def test(model, test_loader, criterion, device):
    model.to(device)
    model.eval()  # evaluation mode disables dropout etc.

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # no gradient calculation
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / total
    accuracy = correct / total * 100
    print(f"Test Loss: {avg_loss:.4f} - Test Accuracy: {accuracy:.2f}%")
    return avg_loss, accuracy

test(model, test_loader, criterion, device)