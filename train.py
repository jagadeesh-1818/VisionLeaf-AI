# train_improved.py
import os
import time
import copy
import random
import math# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from prepare_data import train_loader, validation_loader, train_dataset
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import time

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

# Train only parameters that require gradients
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0008,
    weight_decay=1e-4
)

# Reduce LR by half every 5 epochs
scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

# --- 4. Training Config ---
NUM_EPOCHS = 15
best_accuracy = 0.0

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

    # Step the learning rate scheduler
    scheduler.step()

    # Save best model
    if epoch_accuracy > best_accuracy:
        best_accuracy = epoch_accuracy
        torch.save(model.state_dict(), "best_plant_disease_model.pth")
        print(f"💾 New best model saved with accuracy: {best_accuracy:.2f}%")

end_time = time.time()
print(f"\n✅ Training completed in {(end_time - start_time) / 60:.2f} minutes.")
print(f"🏆 Best validation accuracy: {best_accuracy:.2f}%")

from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import WeightedRandomSampler
from torchvision import models

from prepare_data import train_loader, validation_loader, train_dataset, class_counts  # optional class_counts
from tqdm import tqdm

# -------------------------
# Config / Hyperparameters
# -------------------------
SEED = 42
NUM_EPOCHS = 25
BATCH_SIZE = None  # not used here (loader already created), but tune in prepare_data
IMG_SIZE = 224
NUM_WORKERS = 4
BEST_MODEL_PATH = "best_plant_disease_model.pth"
PATIENCE = 6  # early stopping patience (epochs)
LR = 2e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
MIXED_PRECISION = True  # use AMP for faster training on RTX 3050
UNFREEZE_LAST_N_LAYERS = 15  # how many final parameters to unfreeze for fine-tuning (tuneable)

# reproducibility
def set_seed(seed=SEED):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -------------------------
# Model (MobileNetV2 transfer)
# -------------------------
model = models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')

# Replace classifier
num_classes = len(train_dataset.classes)
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)

# Freeze all params first
for param in model.parameters():
    param.requires_grad = False

# Unfreeze last N params (selective fine-tuning)
# We'll unfreeze classifier already, plus last UNFREEZE_LAST_N_LAYERS parameters of features
params = list(model.features.parameters()) + list(model.classifier.parameters())
if UNFREEZE_LAST_N_LAYERS > 0:
    # get feature parameters as flat list
    feature_params = [p for p in model.features.parameters()]
    for p in feature_params[-UNFREEZE_LAST_N_LAYERS:]:
        p.requires_grad = True

# Always train classifier
for p in model.classifier.parameters():
    p.requires_grad = True

model = model.to(device)

# -------------------------
# Loss / Metrics
# -------------------------
# Option: class weighting if imbalance (requires class_counts from dataset)
if 'class_counts' in globals() and class_counts is not None:
    # compute weights: inverse frequency
    class_counts_tensor = torch.tensor(class_counts, dtype=torch.float)
    class_weights = 1.0 / (class_counts_tensor + 1e-8)
    class_weights = class_weights / class_weights.sum() * len(class_counts_tensor)  # normalize
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    print("Using weighted CrossEntropyLoss due to class imbalance.")
else:
    criterion = nn.CrossEntropyLoss()

# Option: label smoothing (PyTorch 2.x has label_smoothing argument) - use if desired
# criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

# optimizer: only parameters with requires_grad=True
trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)

# scheduler: CosineAnnealingWarmRestarts or ReduceLROnPlateau
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, verbose=True)

# -------------------------
# Optional: Weighted sampler
# -------------------------
train_sampler = None
if 'class_counts' in globals() and class_counts is not None:
    # create a weight per sample for the training sampler
    labels_list = [int(y) for _, y in train_dataset.samples]  # adjust if your dataset stores differently
    counts = torch.tensor(class_counts, dtype=torch.float)
    weights_per_class = 1.0 / (counts + 1e-8)
    sample_weights = torch.tensor([weights_per_class[l] for l in labels_list], dtype=torch.double)
    train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    print("Using WeightedRandomSampler for balanced mini-batches.")

# If your train_loader is created outside and accepts sampler, prefer recreating it. Otherwise continue.

# -------------------------
# Training Loop (with AMP + validation + early stopping)
# -------------------------
scaler = torch.cuda.amp.GradScaler(enabled=MIXED_PRECISION)

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
epochs_no_improve = 0
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_start = time.time()
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0

    train_iter = train_loader
    if train_sampler is not None:
        # if original train_loader wasn't created with sampler, you should recreate it.
        # For now assume train_loader honors sampler; else ignore this block.
        pass

    loop = tqdm(train_iter, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}] train", leave=False)
    for inputs, labels in loop:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=MIXED_PRECISION):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        # gradient clipping
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        scaler.step(optimizer)
        scaler.update()

        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
        total_samples += inputs.size(0)

        loop.set_postfix(loss=loss.item())

    epoch_loss = running_loss / total_samples
    epoch_acc = 100.0 * running_corrects / total_samples

    # Validation
    model.eval()
    val_running_corrects = 0
    val_total = 0
    val_running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in validation_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=MIXED_PRECISION):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            val_running_corrects += torch.sum(preds == labels.data).item()
            val_running_loss += loss.item() * inputs.size(0)
            val_total += inputs.size(0)

    val_loss = val_running_loss / val_total if val_total else 0.0
    val_acc = 100.0 * val_running_corrects / val_total if val_total else 0.0

    # Scheduler step (ReduceLROnPlateau needs metric)
    scheduler.step(val_acc)

    # Print epoch summary
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
    f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | "
    f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # Save best
    if val_acc > best_acc + 1e-4:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"Saved new best model: {BEST_MODEL_PATH} (Acc: {best_acc:.2f}%)")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    # Early stopping
    if epochs_no_improve >= PATIENCE:
        print(f"Early stopping after {epoch+1} epochs (no improvement for {PATIENCE} epochs).")
        break

total_time = (time.time() - start_time) / 60.0
print(f"Training complete in {total_time:.2f} minutes. Best val Acc: {best_acc:.2f}%")

# load best weights
model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), "best_model_final.pth")
print("Final model saved to best_model_final.pth")
