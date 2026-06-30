import streamlit as st
import torch
import timm
import json
from PIL import Image
from torchvision import transforms

st.set_page_config(page_title="Brain Tumor Detector", page_icon="🧠", layout="centered")

@st.cache_resource
def load_model():
    with open('classes.json') as f:
        classes = json.load(f)
    model = timm.create_model('efficientnet_b0', pretrained=False, num_classes=len(classes))
    model.load_state_dict(torch.load('best_brain_model_v2.pth', map_location='cpu'))
    model.eval()
    return model, classes

model, classes = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class_info = {
    "glioma": "Glioma — tumor in the glial cells of the brain.",
    "meningioma": "Meningioma — tumor on the membranes around the brain.",
    "pituitary": "Pituitary tumor — forms in the pituitary gland.",
    "notumor": "No tumor detected — scan appears normal."
}

st.title("🧠 Brain Tumor Detection")
st.markdown("Trained on combined multi-source MRI data for better real-world generalization.")
st.markdown("---")

uploaded = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded MRI", use_column_width=True)

    with col2:
        with st.spinner("Analyzing..."):
            tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                probs = torch.softmax(model(tensor), dim=1)[0]
            confidence, pred_idx = probs.max(0)
            pred_class = classes[pred_idx.item()]

        display_name = pred_class.replace("notumor", "No Tumor").capitalize()

        if pred_class == "notumor":
            st.success(f"✅ **{display_name}**")
        else:
            st.error(f"⚠️ **{display_name} detected**")

        st.metric("Confidence", f"{confidence.item()*100:.1f}%")
        st.info(class_info.get(pred_class, ""))

    st.markdown("---")
    st.markdown("### Prediction Breakdown")
    for i in range(len(classes)):
        name = classes[i].replace("notumor", "No Tumor").capitalize()
        st.progress(float(probs[i].item()), text=f"{name}: {probs[i].item()*100:.1f}%")

    st.markdown("---")
    st.warning("⚠️ Educational tool only — not a substitute for professional medical diagnosis.")
