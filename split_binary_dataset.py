import os
import shutil
import random

source_dir = "dataset"
base_dir = "dataset_split"

train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

classes = ["cancer", "normal"]

for cls in classes:
    os.makedirs(os.path.join(base_dir, "train", cls), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "val", cls), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "test", cls), exist_ok=True)

    images = os.listdir(os.path.join(source_dir, cls))
    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    for img in train_images:
        shutil.copy(
            os.path.join(source_dir, cls, img),
            os.path.join(base_dir, "train", cls, img)
        )

    for img in val_images:
        shutil.copy(
            os.path.join(source_dir, cls, img),
            os.path.join(base_dir, "val", cls, img)
        )

    for img in test_images:
        shutil.copy(
            os.path.join(source_dir, cls, img),
            os.path.join(base_dir, "test", cls, img)
        )

print("Dataset split completed successfully!")
