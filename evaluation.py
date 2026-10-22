import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import json

# ----------------------
# Load Class Names
# ----------------------
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# ----------------------
# Transformations
# ----------------------
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

# ----------------------
# Load Test Dataset
# ----------------------
test_dataset = datasets.ImageFolder(
    root="dataset/test",
    transform=test_transform
)

test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ----------------------
# Load Model
# ----------------------
model = torchvision.models.mobilenet_v2(weights=None)
in_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_features, len(class_names))

model.load_state_dict(torch.load("best_plant_disease_model.pth", map_location="cpu"))
model.eval()

# ----------------------
# Evaluation
# ----------------------
y_true = []
y_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        y_true.extend(labels.tolist())
        y_pred.extend(predicted.tolist())

# ----------------------
# Print Metrics
# ----------------------
print("\n==== Classification Report ====\n")
print(classification_report(y_true, y_pred, target_names=class_names))

print("\n==== Confusion Matrix ====\n")
print(confusion_matrix(y_true, y_pred))

# ----------------------
# Compute Accuracy
# ----------------------
correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
accuracy = correct / len(y_true) * 100

print(f"\n==== Overall Test Accuracy: {accuracy:.2f}% ====\n")
