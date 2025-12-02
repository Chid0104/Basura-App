import streamlit as st
import numpy as np
from PIL import Image
import json
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# PAGE CONFIG
st.set_page_config(
    page_title="BasuraNet AI Classifier",
    page_icon="♻️",
    layout="wide",
)

# MODERN GLASS UI CSS
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1e1f26, #2b5876);
    background-attachment: fixed;
}

/* TITLE */
.title {
    font-size: 50px !important;
    font-weight: 900 !important;
    text-align: center;
    background: -webkit-linear-gradient(45deg, #00ffcc, #00aaff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-top: 10px;
}

/* Subtitle */
.subtitle {
    font-size: 20px;
    color: #e0e0e0;
    text-align: center;
}

/* Glass Card */
.glass {
    background: rgba(255, 255, 255, 0.13);
    padding: 25px;
    border-radius: 18px;
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    box-shadow: 0 8px 25px rgba(0,0,0,0.2);
    border: 1px solid rgba(255,255,255,0.15);
}

/* Result Label */
.label {
    padding: 15px;
    background: #00d084;
    color: white;
    font-size: 30px;
    font-weight: bold;
    border-radius: 12px;
}

/* Unknown label */
.unknown {
    padding: 15px;
    background: #e74c3c;
    color: white;
    font-size: 30px;
    font-weight: bold;
    border-radius: 12px;
}

/* Footer */
footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# LOAD MODEL
@st.cache_resource
def load_basuranet(model_path="model/basuranet_final.h5", mapping_path="class_indices.json"):
    model = load_model(model_path)
    with open(mapping_path, "r") as f:
        labels = json.load(f)
    return model, labels

model_path = os.environ.get("MODEL_PATH", "model/basuranet_final.h5")
model, labels = load_basuranet(model_path)

# HEADER
st.markdown('<div class="title">♻️ BasuraNet AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Modern AI Waste Classification with Webcam Support</div>', unsafe_allow_html=True)
st.write("")

# LAYOUT
col1, col2 = st.columns(2)

# INPUT SECTION
with col1:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("📸 Choose Input Method")

    method = st.radio("Select:", ["Upload Image", "Use Webcam"])

    img = None

    if method == "Upload Image":
        uploaded = st.file_uploader("Upload an image...", type=["jpg","jpeg","png"])
        if uploaded:
            img = Image.open(uploaded)
            st.image(img, use_container_width=True)

    else:
        cam = st.camera_input("Take a picture")
        if cam:
            img = Image.open(cam)
            st.image(img, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

# RESULT SECTION
with col2:
    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.subheader("🔍 AI Result")

    if img:
        # Preprocess
        im = img.resize((224, 224))
        im = img_to_array(im) / 255.0
        im = np.expand_dims(im, axis=0)

        pred = model.predict(im)[0]
        top_idx = int(np.argmax(pred))
        conf = pred[top_idx]

        # Unknown detection
        if conf < 0.60:
            st.markdown("<div class='unknown'>❓ Not a Waste Image</div>", unsafe_allow_html=True)
        else:
            result = labels[str(top_idx)]
            st.markdown(f"<div class='label'>{result}</div>", unsafe_allow_html=True)

    else:
        st.info("Upload or capture an image to classify.")

    st.markdown("</div>", unsafe_allow_html=True)

# FOOTER
st.write("<br><center style='color:white;'>BasuraNet © 2025 • PIT Machine Learning Project</center>", unsafe_allow_html=True)
