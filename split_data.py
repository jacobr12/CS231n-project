import os
import shutil
import random

SOURCE_DIR = "./data"
DEST_DIR = "./data_split"
TRAIN_RATIO = 0.8  # 80% train, 20% test

# Make folders
for split in ["train", "test"]:
    for label in os.listdir(SOURCE_DIR):
        label_path = os.path.join(SOURCE_DIR, label)
        if not os.path.isdir(label_path):
            continue  # Skip .DS_Store or other non-directories
        os.makedirs(os.path.join(DEST_DIR, split, label), exist_ok=True)

# Split and copy files
for label in os.listdir(SOURCE_DIR):
    label_path = os.path.join(SOURCE_DIR, label)
    if not os.path.isdir(label_path):
        continue  # Skip .DS_Store or other non-directories

    images = [f for f in os.listdir(label_path) if f.endswith(".jpg")]
    random.shuffle(images)

    split_idx = int(len(images) * TRAIN_RATIO)
    train_imgs = images[:split_idx]
    test_imgs = images[split_idx:]

    for img in train_imgs:
        src = os.path.join(label_path, img)
        dst = os.path.join(DEST_DIR, "train", label, img)
        shutil.copyfile(src, dst)

    for img in test_imgs:
        src = os.path.join(label_path, img)
        dst = os.path.join(DEST_DIR, "test", label, img)
        shutil.copyfile(src, dst)


print("✅ Done. Split data into ./data_split/train and ./data_split/test")
