import streamlit as st
import torch
import timm
import json
import numpy as np
from PIL import Image
from torchvision import transforms

# --- Config ---
st.set_page_config(
    page_title="Brain Tumor Detector",
    page_icon="🧠",
    layout="centered"
)

# --- Load model ---
@st.cache_resource
def load_model():
    with open('classes.json') as f:
        classes = json.load(f)
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=len(classes))
    model.load_state_dict(torch.load('best_brain_model.pth', map_location='cpu'))
    model.eval()
    return model, classes

model, classes = load_model()

# --- Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --- Class info ---
class_info = {
    "glioma":     {"desc": "Glioma is a tumor that starts in the glial cells of the brain.", "color": "red"},
    "meningioma": {"desc": "Meningioma is a tumor that forms on the membranes around the brain.", "color": "orange"},
    "pituitary":  {"desc": "Pituitary tumor forms in the pituitary gland at the base of the brain.", "color": "orange"},
    "notumor":    {"desc": "No tumor detected. The MRI scan appears normal.", "color": "green"},
}

# --- UI ---
st.title("🧠 Brain Tumor Detection")
st.markdown("Upload an **MRI scan** to detect the presence and type of brain tumor using AI.")
st.markdown("---")

uploaded = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Uploaded MRI Scan", use_column_width=True)

    with col2:
        with st.spinner("Analyzing MRI..."):
            tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1)[0]
            confidence, pred_idx = probs.max(0)
            pred_class = classes[pred_idx.item()]

        info = class_info.get(pred_class, {})
        display_name = pred_class.replace("notumor", "No Tumor").capitalize()

        st.markdown("### Result")
        if pred_class == "notumor":
            st.success(f"✅ **{display_name}**")
        else:
            st.error(f"⚠️ **{display_name} detected**")

        st.metric("Confidence", f"{confidence.item()*100:.1f}%")
        st.info(info.get("desc", ""))

    # --- Top 4 predictions ---
    st.markdown("---")
    st.markdown("### Prediction Breakdown")
    for i in range(len(classes)):
        name = classes[i].replace("notumor", "No Tumor").capitalize()
        prob = probs[i].item()
        st.progress(float(prob), text=f"{name}: {prob*100:.1f}%")

    # --- Disclaimer ---
    st.markdown("---")
    st.warning("⚠️ This tool is for educational purposes only and is not a substitute for professional medical diagnosis.")