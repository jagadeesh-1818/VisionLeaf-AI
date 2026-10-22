import gradio as gr
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import json
import os

# Disable HuggingFace warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# -------- Load Model --------
def load_model():
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, len(class_names))
    model.load_state_dict(torch.load("best_plant_disease_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model, class_names

model, class_names = load_model()
solutions = pd.read_csv("solutions.csv")

# -------- Prediction Function --------
def predict(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_t = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_t)
        probs = torch.nn.functional.softmax(output, dim=1)
        conf, idx = torch.max(probs, 1)

    pred = class_names[idx.item()]
    confidence = conf.item() * 100

    if "Healthy" in pred:
        result = f"✅ The plant is **Healthy!**\nConfidence: {confidence:.2f}%"
        treatment = "🌿 Keep regular watering, sunlight, and nutrient balance."
    else:
        row = solutions[solutions["disease_name"] == pred]
        treatment = row["solution"].values[0] if not row.empty else "No treatment info found."
        result = f"⚠️ Disease detected: **{pred.replace('_', ' ')}**\nConfidence: {confidence:.2f}%"

    return result, treatment


custom_css = """
html, body, .gradio-container {
    background: linear-gradient(180deg, #3e2a1e, #7a5230, #b08b56) !important;
    margin: 0;
    padding: 0;
    height: 100%;
    width: 100%;
    overflow-x: hidden !important;
    font-family: 'Poppins', sans-serif !important;
}

/* Fix side white gaps */
.gr-blocks, .gradio-container {
    max-width: 100% !important;
    padding: 0 2% !important;
}

/* Header section */
#header {
    background: linear-gradient(135deg, #1b4332, #2d6a4f, #3a5a40);
    color: #f8f9fa !important;
    padding: 25px 10px;
    border-radius: 12px;
    text-align: center;
    margin: 10px auto 25px auto;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    width: 95%;
}

/* VisionLeaf AI title */
#header h1, h1 {
    color: #a8e6cf !important; /* Light green highlight */
    font-weight: 700;
    letter-spacing: 1px;
}

/* Subtitle: "Medicinal Plant Disease Detection" */
#header h2 {
    color: #ffffff !important;
    font-weight: 500;
}

/* Tagline: "Smart • Sustainable • Accurate" */
#header h3, .tagline {
    color: #a8e6cf !important; /* Same as VisionLeaf AI */
    font-weight: 600;
    font-size: 1rem;
    letter-spacing: 0.5px;
    margin-top: 5px;
}

/* Global text color fixes */
h1, h2, h3 {
    color: #ffffff !important;
}
label, .gr-markdown, span, p, .svelte-1ipelgc, .wrap, .gr-form label {
    color: #000000 !important;
}

/* Image upload area */
.gr-image {
    background: #fffaf0 !important;
    border-radius: 10px !important;
    border: 2px solid #4d7c0f !important;
    box-shadow: 0 0 10px rgba(77, 124, 15, 0.3);
}

/* Treatment text box */
#treatment_box textarea, textarea {
    background: #fffaf0 !important;
    color: #000 !important;
    border-radius: 10px !important;
    border: 2px solid #4d7c0f !important;
    box-shadow: 0 0 10px rgba(77, 124, 15, 0.3);
}

/* Analyze button */
button {
    background: linear-gradient(135deg, #e3b04b, #f2c14e) !important;
    color: #222 !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    border: none !important;
    box-shadow: 0 0 12px rgba(255, 215, 0, 0.3);
    transition: all 0.3s ease-in-out !important;
    padding: 10px 20px;
}

button:hover {
    background: linear-gradient(135deg, #ffd166, #f9c74f);
    box-shadow: 0 0 18px rgba(255, 215, 0, 0.6);
    transform: scale(1.05);
}

/* Footer */
footer, #footer { display: none !important; }
"""

# -------- Gradio UI Layout --------
with gr.Blocks(css=custom_css, title="VisionLeaf AI 🌿") as demo:
    gr.HTML("""
        <div id="header">
            <h1 style="margin: 0; font-size: 2em;">🌿 VisionLeaf AI</h1>
            <h3 style="margin-top: 5px;">Medicinal Plant Disease Detection</h3>
            <p style="color:#b7e4c7; font-weight:500;">Smart • Sustainable • Accurate</p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(label="📸 Upload Leaf Image", type="pil")
            analyze_btn = gr.Button("🔍 Analyze Leaf", variant="primary")
        with gr.Column(scale=1):
            result_box = gr.Markdown()
            treatment_box = gr.Textbox(
                label="🌱 Treatment Suggestion",
                lines=6,
                interactive=False,
                elem_id="treatment_box"
            )

    analyze_btn.click(fn=predict, inputs=image_input, outputs=[result_box, treatment_box])

    gr.Markdown("""
        <div style='text-align:center; color:#fff; margin-top:25px;'>
            Developed with ❤️ using PyTorch + Gradio
        </div>
    """)

demo.launch()
