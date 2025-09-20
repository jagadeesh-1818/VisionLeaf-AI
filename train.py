import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from prepare_data import train_loader, validation_loader, train_dataset
import time

# --- 1. Set up Model, Loss Function, and Optimizer ---

# Define the device to run on (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load a pre-trained MobileNetV2 model
# We use a lightweight model perfect for potential mobile deployment
model = models.mobilenet_v2(weights='MobileNet_V2_Weights.DEFAULT')

# Freeze all the base model parameters
for param in model.parameters():
    param.requires_grad = False

# Get the number of classes from our dataset
num_classes = len(train_dataset.classes)

# Replace the final classifier layer with a new one for our specific number of classes
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes)

# Move the model to the selected device
model = model.to(device)

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier[1].parameters(), lr=0.001)

# --- 2. Training and Validation Loop ---

NUM_EPOCHS = 10  # An epoch is one full pass through the entire training dataset
best_accuracy = 0.0

print("Starting training...")
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    # --- Training Phase ---
    model.train()  # Set the model to training mode
    running_loss = 0.0
    for images, labels in train_loader:
        # Move images and labels to the device (GPU/CPU)
        images, labels = images.to(device), labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)

    # --- Validation Phase ---
    model.eval()  # Set the model to evaluation mode
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad(): # We don't need to calculate gradients during validation
        for images, labels in validation_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
    epoch_accuracy = 100 * correct_predictions / total_predictions
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | "
            f"Training Loss: {epoch_loss:.4f} | "
            f"Validation Accuracy: {epoch_accuracy:.2f}%")
    
    # Save the model if it has the best validation accuracy so far
    if epoch_accuracy > best_accuracy:
        best_accuracy = epoch_accuracy
        torch.save(model.state_dict(), 'best_plant_disease_model.pth')
        print(f"New best model saved with accuracy: {best_accuracy:.2f}%")

end_time = time.time()
print(f"Training finished in {(end_time - start_time) / 60:.2f} minutes.")
print(f"Best validation accuracy: {best_accuracy:.2f}%")