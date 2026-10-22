# 🌿 VisionLeaf AI — Medicinal Plant Disease Detection

VisionLeaf AI is a computer vision project designed to identify medicinal plant diseases from leaf images and provide suggested treatment information.

## ✨ Features

- 📷 Upload a leaf image for analysis
- 🧠 Deep learning-based image classification using MobileNetV2
- 🌱 Display a predicted plant disease class
- 💡 Provide treatment suggestions using a CSV-based lookup
- 📊 Include model evaluation and training visualizations

## 🛠️ Tech Stack

- **Python**
- **PyTorch** — deep learning and model inference
- **Torchvision** — image model architecture and transformations
- **Gradio** — interactive user interface
- **Pandas** — treatment information and CSV handling
- **Pillow** — image processing

## 📁 Project Structure

```text
VisionLeaf-AI/
├── Web.py
├── best_plant_disease_model.pth
├── class_names.json
├── solutions.csv
├── train.py
├── evaluation.py
├── plots.py
├── prepare_data.py
├── split_data.py
├── verify_images.py
├── create_class_json.py
├── accuracy_plot.png
├── loss_plot.png
├── history.json
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/jagadeesh-1818/VisionLeaf-AI.git
cd VisionLeaf-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install gradio torch torchvision pandas pillow
```

Install any additional dependencies required by the project.

### 4. Run the application

```bash
python Web.py
```

Open the local URL displayed in the terminal, usually `http://127.0.0.1:7860`.

## 🧠 Model

The project uses a trained MobileNetV2 image-classification model saved as `best_plant_disease_model.pth`.

The application loads the saved checkpoint to make predictions; retraining is not required to run inference when the necessary model and class-label files are available.

## 📊 Evaluation

Training and evaluation visualizations are included in the repository. Refer to `accuracy_plot.png`, `loss_plot.png`, and `history.json` for the available recorded results.

## 🖥️ Demo

![VisionLeaf AI prediction demo 1](screenshots/Demo1.png)

![VisionLeaf AI prediction demo 2](screenshots/Demo2.png)

## ⚠️ Disclaimer

Predictions and treatment suggestions are informational and should be verified before applying treatments to real plants.

## 👨‍💻 Author

Developed as an AI/ML project exploring deep learning and medicinal plant disease detection.

---

If you find this project useful, consider giving the repository a ⭐.
