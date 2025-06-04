import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
from collections import defaultdict

# Paths and device setup
root = "./data_split/test/"
MODEL_PATH = "checkpoints/finger_mobilenetv2.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, 5)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)

# Transform (same as training)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Initialize metrics
confusion = np.zeros((5, 5), dtype=int)
misclassified = []

correct = 0
total = 0

for label in sorted(os.listdir(root)):
    folder = os.path.join(root, label)
    if not os.path.isdir(folder):
        continue
    for fname in sorted(os.listdir(folder)):
        img_path = os.path.join(folder, fname)
        try:
            img = Image.open(img_path).convert("RGB")
        except:
            print(f"❌ Failed to open: {img_path}")
            continue

        img_tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.argmax(output, dim=1).item() + 1  # Labels 1–5

        gt = int(label)
        confusion[gt - 1][pred - 1] += 1
        if pred == gt:
            correct += 1
        else:
            misclassified.append((img_path, gt, pred))
        total += 1

        print(f"{img_path} | GT: {gt}, Pred: {pred}")

# Print final stats
print(f"\n✅ Final Accuracy: {correct}/{total} = {correct / total:.2%}\n")

# Print confusion matrix
print("📊 Confusion Matrix (rows = GT, cols = Pred):")
print(confusion)

# List misclassified images
if misclassified:
    print("\n❌ Misclassified Images:")
    for path, gt, pred in misclassified:
        print(f"{path} | GT: {gt}, Pred: {pred}")
else:
    print("\n🎯 No misclassifications!")
