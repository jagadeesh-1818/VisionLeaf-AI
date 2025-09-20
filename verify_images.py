# verify_images.py
import os
from PIL import Image

# Set the path to your main dataset folder
dataset_path = 'dataset/'

print("Starting image verification...")
corrupted_files = []

for root, _, files in os.walk(dataset_path):
    for file_name in files:
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            file_path = os.path.join(root, file_name)
            try:
                # Try to open the image and load its data
                with Image.open(file_path) as img:
                    img.load()
            except Exception as e:
                print(f"Found corrupted file: {file_path}")
                print(f"  Error: {e}")
                corrupted_files.append(file_path)

if not corrupted_files:
    print("Verification complete. No corrupted images found.")
else:
    print(f"\nVerification complete. Found {len(corrupted_files)} corrupted files.")
    print("Please delete or replace these files and try training again.")