# train_left_hand_cnn.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# ========== Configuration ==========
DATA_DIR = 'left_hand_images'
BATCH_SIZE = 32
EPOCHS = 10
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_PATH = 'left_hand_open_closed_model.pth'

# ========== Data Augmentation & Loading ==========
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ========== Define Model Choices ==========
def get_model(name):
    if name == 'mobilenet':
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.last_channel, 2)
    elif name == 'resnet18':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    else:
        raise ValueError("Unknown model name")
    return model

# ========== Training Function ==========
def train_model(model, name):
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    losses = []
    accuracies = []

    print(f"\nTraining {name}...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        epoch_loss = total_loss / total
        epoch_acc = correct / total
        losses.append(epoch_loss)
        accuracies.append(epoch_acc)
        print(f"Epoch {epoch+1}: Loss={epoch_loss:.4f}, Accuracy={epoch_acc:.4f}")
    return model, losses

# ========== Try Multiple Models ==========
model_losses = {}

for model_name in ['mobilenet', 'resnet18', 'efficientnet_b0']:
    model = get_model(model_name)
    trained_model, losses = train_model(model, model_name)
    model_losses[model_name] = losses
    torch.save(trained_model.state_dict(), f"{model_name}_left_hand.pth")
    print(f"✅ Saved {model_name}_left_hand.pth")

# ========== Plotting ==========
plt.figure(figsize=(10, 6))
for name, losses in model_losses.items():
    plt.plot(range(1, EPOCHS+1), losses, label=name)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss per Epoch for Each Model")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("left_hand_model_losses.png")
print("📉 Saved loss comparison plot as left_hand_model_losses.png")
