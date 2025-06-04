import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torchvision.transforms.functional as F
import random
import matplotlib.pyplot as plt
import json
from collections import defaultdict

class RandomBlur:
    def __init__(self, p=0.5, kernel_size=5):
        self.p = p
        self.kernel_size = kernel_size

    def __call__(self, img):
        if random.random() < self.p:
            return F.gaussian_blur(img, kernel_size=self.kernel_size)
        return img

# ==== CONFIG ====
DATA_DIR = "./data_split"
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
NUM_CLASSES = 5
epoch_train_acc = []
epoch_test_acc = []
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ==== TRANSFORMS ====
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

data_augmented = {
    "with_aug": transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ]),
    "no_aug": transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
    ])
}

all_results = defaultdict(lambda: {"train_acc": [], "test_acc": []})

# ==== TEST DIFFERENT MODELS AND AUGMENTATION SETTING ====
architectures = {
    "mobilenet_v2": models.mobilenet_v2,
    "resnet18": models.resnet18,
    "efficientnet_b0": models.efficientnet_b0
}

for aug_name, train_transform in data_augmented.items():
    test_transform = data_augmented["no_aug"]  # Always use same test transform

    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transform)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=test_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    

    for name, model_fn in architectures.items():
        train_acc_history, test_acc_history = [], []
        train_loss_history = []
        print(f"\n===== Training {name} ({aug_name}) =====")

        weights = None
        if "mobilenet" in name:
            weights = models.MobileNet_V2_Weights.DEFAULT
        elif "resnet" in name:
            weights = models.ResNet18_Weights.DEFAULT
        elif "efficientnet" in name:
            weights = models.EfficientNet_B0_Weights.DEFAULT

        model = model_fn(weights=weights)

        if "resnet" in name:
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
        elif "efficientnet" in name:
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
        else:
            model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)

        model = model.to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        for epoch in range(NUM_EPOCHS):
            model.train()
            total, correct = 0, 0
            running_loss = 0.0

            for images, labels in train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

            train_acc = correct / total
            print(f"[Epoch {epoch+1}] Train Loss: {running_loss:.4f} | Train Acc: {train_acc:.2%}")

            model.eval()
            total, correct = 0, 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(DEVICE), labels.to(DEVICE)
                    outputs = model(images)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            test_acc = correct / total
            print(f"[Epoch {epoch+1}] Test Acc: {test_acc:.2%}")
            epoch_train_acc.append(train_acc)
            epoch_test_acc.append(test_acc)
            train_acc_history.append(train_acc)
            train_loss_history.append(running_loss / len(train_loader))
            test_acc_history.append(test_acc)
            key = f"{name}_{aug_name}"
            all_results[key]["train_acc"] = train_acc_history.copy()
            all_results[key]["test_acc"] = test_acc_history.copy()

        save_name = f"checkpoints/finger_{name}_{aug_name}.pt"
        torch.save(model.state_dict(), save_name)
        print(f"✅ Saved model to {save_name}")
        log_dir = "accuracy_logs"
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{name}_{aug_name}.json")
        with open(log_path, "w") as f:
            json.dump({"train_acc": epoch_train_acc, "test_acc": epoch_test_acc}, f)
        epochs = list(range(1, NUM_EPOCHS + 1))
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_loss_history, label="Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training Loss ({name} + {aug_name})")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_acc_history, label="Train Acc")
        plt.plot(epochs, test_acc_history, label="Test Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title(f"Train/Test Accuracy ({name} + {aug_name})")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        filename = f"{name}_{aug_name}_results.png".replace(" ", "_").lower()
        plt.savefig(filename)
        print(f"📈 Saved training curve figure as {filename}")

        plt.figure(figsize=(10, 6))
for key, result in all_results.items():
    plt.plot(range(1, NUM_EPOCHS + 1), result["test_acc"], label=key)
plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.title("Comparison of Test Accuracy Across Models")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("comparison_test_accuracy.png")
print("📊 Saved overall comparison figure as comparison_test_accuracy.png")

# Plot comparison of training accuracies across all models
plt.figure(figsize=(10, 6))
for key, result in all_results.items():
    plt.plot(range(1, NUM_EPOCHS + 1), result["train_acc"], label=key)
plt.xlabel("Epoch")
plt.ylabel("Train Accuracy")
plt.title("Comparison of Train Accuracy Across Models")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("comparison_train_accuracy.png")
print("📊 Saved overall comparison figure as comparison_train_accuracy.png")
