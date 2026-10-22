import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from prepare_data import train_loader, validation_loader, train_dataset
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- 1. Setup Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# --- 2. Load Pretrained Model ---
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze the last few convolutional blocks for fine-tuning
for name, param in model.features[-3:].named_parameters():
    param.requires_grad = True

# Modify classifier to match your dataset
num_classes = len(train_dataset.classes)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)
model = model.to(device)

# --- 3. Define Loss, Optimizer & Scheduler ---
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0008,
    weight_decay=1e-4
)

scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# --- 4. Training Config ---
NUM_EPOCHS = 15
best_accuracy = 0.0

# --- NEW: Track History ---
history = {
    "train_loss": [],
    "val_loss": [],
    "val_accuracy": []
}

print("\n🌿 Starting VisionLeaf AI training...\n")
start_time = time.time()

# --- 5. Training + Validation Loop ---
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", colour="green")
    for images, labels in train_loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        train_loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / len(train_loader.dataset)

    # --- Validation ---
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    epoch_accuracy = 100 * correct_predictions / total_predictions
    val_loss = val_loss / len(validation_loader.dataset)

    print(f"\n📊 Epoch {epoch+1}: "
          f"Train Loss = {epoch_loss:.4f}, "
          f"Val Loss = {val_loss:.4f}, "
          f"Val Accuracy = {epoch_accuracy:.2f}%")

    # --- NEW: Save values ---
    history["train_loss"].append(epoch_loss)
    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(epoch_accuracy)

    # Scheduler
    scheduler.step()

    # Save best model
    if epoch_accuracy > best_accuracy:
        best_accuracy = epoch_accuracy
        torch.save(model.state_dict(), "best_plant_disease_model.pth")
        print(f"💾 New best model saved with accuracy: {best_accuracy:.2f}%")

end_time = time.time()
print(f"\n✅ Training completed in {(end_time - start_time) / 60:.2f} minutes.")
print(f"🏆 Best validation accuracy: {best_accuracy:.2f}%")

# --- NEW: Save training history ---
with open("history.json", "w") as f:
    json.dump(history, f, indent=4)
print("📁 Saved training history → history.json")

# --- NEW: Plot Loss Curve ---
plt.figure(figsize=(8,5))
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.savefig("loss_plot.png", dpi=300)
plt.close()
print("📈 Saved → loss_plot.png")

# --- NEW: Plot Accuracy Curve ---
plt.figure(figsize=(8,5))
plt.plot(history["val_accuracy"], label="Validation Accuracy", color="green")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Validation Accuracy Curve")
plt.legend()
plt.savefig("accuracy_plot.png", dpi=300)
plt.close()
print("📈 Saved → accuracy_plot.png")
