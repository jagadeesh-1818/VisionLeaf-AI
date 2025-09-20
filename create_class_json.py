# A faster version of create_class_json.py
import os
import json

# Define the path to your training directory
train_dir = 'dataset/train'

try:
    # Get a list of all entries in the directory and filter for subdirectories
    class_names = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

    # Save the list of folder names to the JSON file
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f)

    print("class_names.json file created successfully ")
    print(f"Found {len(class_names)} classes: {class_names}")

except FileNotFoundError:
    print(f"Error: The directory was not found at '{train_dir}'")
    print("Please make sure your folder structure is correct.")