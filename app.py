import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import gdown
from keras.models import load_model

# ================= PAGE CONFIG =================
st.set_page_config(page_title="Road Anomaly Detection", layout="wide")
st.title("🚦 Road Anomaly Detection System")

# ================= CONSTANTS =================
MODEL_FILE = "road_anomaly_model.h5"
FILE_ID = "1FiHUDZPL1MFyG1g06_jjM4MJV2tH9rpg"
CLASS_NAMES = ["Accident", "Fight", "Fire", "Snatching"]
CONF_THRESHOLD = 0.85
IMG_SIZE = (224, 224)

# ================= MODEL LOAD =================
@st.cache_resource
def load_cnn():
    if not os.path.exists(MODEL_FILE):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_FILE)
    return load_model(MODEL_FILE)

model = load_cnn()

# ================= PREPROCESS =================
def preprocess(img):
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# ================= PREDICT =================
def predict(img):
    preds = model.predict(preprocess(img), verbose=0)
    idx = int(np.argmax(preds))
    conf = float(np.max(preds))
    emergency = conf >= CONF_THRESHOLD
    return CLASS_NAMES[idx], conf, emergency

# ================= SIDEBAR =================
st.sidebar.title("🚘 Navigation")
mode = st.sidebar.radio(
    "Select Mode",
    ["Upload Image", "Upload Video", "About"]
)

# ================= ABOUT =================
if mode == "About":
    st.subheader("About")
    st.markdown("""
    **Project:** Road Anomaly Detection System  
    **Developer:** Vijay Ragavan  
    **College:** Kamarajar Engineering College of Technology  

    CNN-based classification of:
    - Accident
    - Fight
    - Fire
    - Snatching
    """)

# ================= IMAGE UPLOAD =================
elif mode == "Upload Image":
    st.subheader("📷 Upload Image")

    image_file = st.file_uploader(
        "Choose Image",
        type=["jpg", "jpeg", "png"]
    )

    if image_file:
        img_bytes = image_file.read()
        img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), 1)
        st.image(img, channels="BGR", use_container_width=True)

        if st.button("🔍 Predict Image"):
            cls, conf, emg = predict(img)
            st.metric("Prediction", cls)
            st.metric("Confidence", f"{conf*100:.2f}%")
            st.error("🚨 EMERGENCY") if emg else st.success("✅ Normal")

# ================= VIDEO UPLOAD =================
elif mode == "Upload Video":
    st.subheader("🎞 Upload Video")

    video_file = st.file_uploader(
        "Choose Video",
        type=["mp4", "avi", "mov"]
    )

    if video_file:
        video_bytes = video_file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(video_bytes)
            video_path = tfile.name

        if st.button("▶ Analyze Video"):
            cap = cv2.VideoCapture(video_path)
            frame_box = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                cls, conf, emg = predict(frame)
                label = f"{cls} ({conf*100:.1f}%)"
                color = (0, 0, 255) if emg else (0, 255, 0)

                cv2.putText(frame, label, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                frame_box.image(frame, channels="BGR", use_container_width=True)

            cap.release()
            st.success("Video Analysis Completed")
