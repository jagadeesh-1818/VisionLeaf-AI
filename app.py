# app.py — VisionLeaf AI 2.0
import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import pandas as pd
import time

# --- Page Config ---
st.set_page_config(page_title="VisionLeaf AI", page_icon="🌿", layout="centered")

# --- Custom CSS ---
st.markdown("""
    <style>
    body {
        background: linear-gradient(180deg, #0b3d2e, #1b4332, #081c15);
        color: #e9f5ec;
        font-family: 'Poppins', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #caffbf;
    }
    .stButton>button {
        background-color: #52b788;
        color: white;
        font-size: 16px;
        border-radius: 10px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #95d5b2;
        color: black;
        transform: scale(1.05);
    }
    .reportview-container {
        background-color: transparent;
    }
    .css-1y4p8pa {
        font-size: 18px !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🌿 VisionLeaf AI")
st.caption("Medicinal Plant Disease Detection — Smart. Simple. Sustainable.")

# --- Load Model + Class Names ---
@st.cache_resource
def load_model():
    with open("class_names.json", "r") as f:
        class_names = json.load(f)
    num_classes = len(class_names)
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load("best_plant_disease_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model, class_names

# --- Load Treatment Data ---
@st.cache_data
def load_solutions():
    return pd.read_csv("solutions.csv")

# --- Predict Function ---
def predict(model, class_names, image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, pred_idx = torch.max(probs, 1)
    return class_names[pred_idx.item()], confidence.item()

# --- Main App ---
uploaded_file = st.file_uploader("📸 Upload a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    with st.spinner("🌿 Analyzing leaf health..."):
        time.sleep(2)
        model, class_names = load_model()
        pred, conf = predict(model, class_names, image)

    conf_percent = conf * 100
    st.markdown(f"### ✅ Prediction: **{pred.replace('_', ' ')}**")
    st.markdown(f"**Confidence:** {conf_percent:.2f}%")

    solutions = load_solutions()
    if "Healthy" not in pred:
        try:
            sol = solutions[solutions["disease_name"] == pred]["solution"].iloc[0]
            st.warning("💊 Treatment Suggestion:")
            st.info(sol)
        except IndexError:
            st.info("No treatment info available for this disease.")
    else:
        st.success("🌼 The plant appears healthy!")
        st.balloons()
