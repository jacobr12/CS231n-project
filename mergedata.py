import os
import shutil

src_root = "data_diff"
dst_root = "data"

for label in os.listdir(src_root):
    src_folder = os.path.join(src_root, label)
    dst_folder = os.path.join(dst_root, label)
    
    if not os.path.isdir(src_folder):
        continue  # skip .DS_Store and other non-folder files

    os.makedirs(dst_folder, exist_ok=True)

    src_folder = os.path.join(src_root, label)
    dst_folder = os.path.join(dst_root, label)
    os.makedirs(dst_folder, exist_ok=True)

    for fname in os.listdir(src_folder):
        src_path = os.path.join(src_folder, fname)
        dst_path = os.path.join(dst_folder, fname)

        # To avoid name conflicts, you can prefix files from data_diff
        if os.path.exists(dst_path):
            base, ext = os.path.splitext(fname)
            new_fname = f"{base}_diff{ext}"
            dst_path = os.path.join(dst_folder, new_fname)

        shutil.copy2(src_path, dst_path)

print("✅ Merged all data from data_diff/ into data/")
