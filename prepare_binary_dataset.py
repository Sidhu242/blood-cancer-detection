import os
import shutil

# Path to your original dataset
source_path = r"D:\RM - early detection of blood cancer\RM data collection\first data set"

# New dataset folder
target_path = r"D:\RM - early detection of blood cancer\model\dataset"

# Create cancer and normal folders
os.makedirs(os.path.join(target_path, "cancer"), exist_ok=True)
os.makedirs(os.path.join(target_path, "normal"), exist_ok=True)

# Folder names in original dataset
malignant_folders = [
    "[Malignant] early Pre-B",
    "[Malignant] Pre-B",
    "[Malignant] Pro-B"
]

benign_folder = "Benign"

# Copy malignant images → cancer
for folder in malignant_folders:
    folder_path = os.path.join(source_path, folder)
    for file in os.listdir(folder_path):
        src_file = os.path.join(folder_path, file)
        dst_file = os.path.join(target_path, "cancer", file)
        shutil.copy(src_file, dst_file)

# Copy benign images → normal
benign_path = os.path.join(source_path, benign_folder)
for file in os.listdir(benign_path):
    src_file = os.path.join(benign_path, file)
    dst_file = os.path.join(target_path, "normal", file)
    shutil.copy(src_file, dst_file)

print("Binary dataset prepared successfully!")
