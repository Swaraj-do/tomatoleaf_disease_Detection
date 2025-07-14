import os
import cv2
import numpy as np
import pickle

# ----- Setup paths -----
train_dir = 'train'
val_dir = 'val'
output_dir = 'clean_data'

os.makedirs(output_dir, exist_ok=True)

# ----- Preprocessing -----
def preprocess(image):
    image = cv2.resize(image, (100, 100))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # for grayscale
    return image

# ----- Load and label images -----
all_images = []
all_labels = []

def load_images_from_folder(folder):
    count = 0
    for class_name in os.listdir(folder):
        class_path = os.path.join(folder, class_name)
        if not os.path.isdir(class_path):
            continue
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = preprocess(img)
                all_images.append(img)
                all_labels.append(class_name)
                count += 1
    print(f"Loaded {count} images from {folder}")

load_images_from_folder(train_dir)
load_images_from_folder(val_dir)

# ----- Convert and save -----
all_images = np.array(all_images)
all_labels = np.array(all_labels)

print(f"Total images: {len(all_images)} | Total labels: {len(all_labels)}")

with open(os.path.join(output_dir, 'images.p'), 'wb') as f:
    pickle.dump(all_images, f)

with open(os.path.join(output_dir, 'labels.p'), 'wb') as f:
    pickle.dump(all_labels, f)

print("✅ Saved to clean_data/images.p and labels.p")

