# resnet18_hyperparam_search.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from itertools import product
import pandas as pd

# ==== Configuration ====
TRAIN_DIR = 'left_hand_images/train'
TEST_DIR = 'left_hand_images/test'
IMG_SIZE = 128
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FREEZE_BACKBONE = False  # Set to True to only fine-tune head
USE_PRETRAINED = True    # Set to False to train from scratch

# ==== Transforms ====
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==== Hyperparameter Grid ====
learning_rates = [1e-4, 1e-3, 5e-4]
batch_sizes = [16, 32, 64]
optimizers = ['adam', 'sgd']

# ==== Results Container ====
results = []
all_epochs = []

# ==== Model Definition ====
def get_resnet18():
    weights = models.ResNet18_Weights.DEFAULT if USE_PRETRAINED else None
    model = models.resnet18(weights=weights)
    if FREEZE_BACKBONE:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
        for param in model.fc.parameters():
            param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model

# ==== Training and Evaluation ====
def train_and_evaluate(lr, batch_size, opt_name):
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
    test_dataset = datasets.ImageFolder(TEST_DIR, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = get_resnet18().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr) if opt_name == 'adam' else optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=0.9)

    train_accs, test_accs, train_losses = [], [], []
    best_test_acc, best_epoch = 0, -1

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        epoch_train_acc = correct / total
        epoch_train_loss = total_loss / total

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                preds = outputs.argmax(1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        epoch_test_acc = correct / total
        if epoch_test_acc > best_test_acc:
            best_test_acc = epoch_test_acc
            best_epoch = epoch

        train_accs.append(epoch_train_acc)
        test_accs.append(epoch_test_acc)
        train_losses.append(epoch_train_loss)

        all_epochs.append({
            "epoch": epoch+1,
            "lr": lr,
            "batch_size": batch_size,
            "optimizer": opt_name,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "test_acc": epoch_test_acc
        })

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | Test Acc: {epoch_test_acc:.4f}")

    results.append({
        "lr": lr,
        "batch_size": batch_size,
        "optimizer": opt_name,
        "test_acc": best_test_acc,
        "best_epoch": best_epoch,
        "train_accs": train_accs,
        "test_accs": test_accs,
        "train_losses": train_losses
    })

# ==== Run Grid Search ====
for lr, batch_size, opt_name in product(learning_rates, batch_sizes, optimizers):
    train_and_evaluate(lr, batch_size, opt_name)

# ==== Save & Plot ====
df = pd.DataFrame([{k: v for k, v in r.items() if k not in ["train_accs", "test_accs", "train_losses"]} for r in results])
df.to_csv("resnet18_hyperparam_results.csv", index=False)

epoch_df = pd.DataFrame(all_epochs)
epoch_df.to_csv("resnet18_epochwise_results.csv", index=False)

# Plot curves
for r in results:
    label = f"{r['optimizer'].upper()} bs={r['batch_size']} lr={r['lr']}"
    epochs = list(range(1, EPOCHS + 1))
    plt.plot(epochs, r["test_accs"], label=label)

plt.xlabel("Epoch")
plt.ylabel("Test Accuracy")
plt.title("ResNet18 Test Accuracy across Epochs")
plt.grid(True)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("resnet18_test_accuracy_curves.png")
print("📊 Saved epoch-wise accuracy plot as resnet18_test_accuracy_curves.png")
