import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image

# ✅ Define class names as per your dataset
class_names = ['cyclone', 'earthquake', 'flood', 'landslide', 'tsunami', 'volcano', 'wildfire', 'normal']  # Example

# ✅ Load model
def load_model():
    model = resnet18(weights=None)  # No pre-trained weights
    model.fc = nn.Linear(model.fc.in_features, len(class_names))  # Adjust output layer
    model.load_state_dict(torch.load("disaster_classifier.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# ✅ Define image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ✅ Streamlit UI
st.title("🌍 Disaster Image Classifier")
uploaded_file = st.file_uploader("Upload an image of a disaster", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        prediction = class_names[predicted.item()]
    
    st.success(f"🧠 Predicted Disaster Class: **{prediction}**")
