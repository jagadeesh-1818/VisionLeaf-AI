# app.py
import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import json
import pandas as pd

# --- Functions ---

# Function to load the model (cached so it only runs once)
@st.cache_resource
def load_model():
    """Loads the trained model and class names."""
    with open('class_names.json', 'r') as f:
        class_names = json.load(f)
    num_classes = len(class_names)
    
    # Recreate the exact model architecture
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    
    # Load the saved weights of your best model
    model.load_state_dict(torch.load('best_plant_disease_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model, class_names

# Function to load treatment solutions from the CSV file
@st.cache_data
def load_solutions():
    """Loads the treatment solutions from a CSV file."""
    return pd.read_csv("solutions.csv")

# Function to preprocess the image and make a prediction
def predict(model, class_names, image):
    """Makes a prediction and returns the class name and confidence score."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(image)
        # Apply Softmax to get probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        # Get the top probability and its class index
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    class_name = class_names[predicted_idx.item()]
    confidence_score = confidence.item()
    return class_name, confidence_score

# --- Streamlit Web App Interface ---

st.set_page_config(page_title="VisionLeaf AI", page_icon="🌿")
st.title("🌿 VisionLeaf AI: Plant Disease Detection")
st.write("Upload an image of a medicinal plant leaf to identify its health status.")

# File uploader allows user to upload an image
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Show a "Classifying..." message while working
    st.write("")
    with st.spinner('Classifying...'):
        model, class_names = load_model()
        prediction, confidence = predict(model, class_names, image)
    
    # Convert confidence to percentage
    confidence_percent = confidence * 100
    
    # Display the prediction and confidence
    st.success(f"**Prediction:** {prediction.replace('_', ' ')}")
    st.info(f"**Confidence:** {confidence_percent:.2f}%")

    # Load solutions and display the relevant one
    solutions_df = load_solutions()
    if "Healthy" not in prediction:
        try:
            solution = solutions_df[solutions_df['disease_name'] == prediction]['solution'].iloc[0]
            st.warning("**Treatment Suggestion:**")
            st.write(solution)
        except IndexError:
            st.write("No specific treatment suggestion found in the database for this condition.")
    else:
        st.balloons()
        st.success("The plant appears to be healthy!")