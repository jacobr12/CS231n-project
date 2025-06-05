import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ===== Configuration =====
MODEL_PATH = 'resnet18_left_hand.pth'
IMAGE_PATH = 'left_hand_images/test/closed/frame_1.jpg'  # Change to your test image
IMG_SIZE = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Load Model =====
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# ===== Image Transform =====
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ===== Load & Predict =====
img = Image.open(IMAGE_PATH).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    output = model(img_tensor)
    pred_class = torch.argmax(output, dim=1).item()

label_map = {0: 'closed', 1: 'open'}
print(f"Predicted class: {pred_class} → {label_map[pred_class]}")
