import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import gdown
from keras.models import load_model

# ================= PAGE CONFIG =================
st.set_page_config("Road Anomaly Detection", layout="wide")
st.title("🚦 Road Anomaly Detection System")

# ================= CONSTANTS =================
MODEL_FILE = "road_anomaly_model.h5"
FILE_ID = "1FiHUDZPL1MFyG1g06_jjM4MJV2tH9rpg"
CLASS_NAMES = ["Accident", "Fight", "Fire", "Snatching"]
CONF_THRESHOLD = 0.85
IMG_SIZE = (224, 224)

# ================= MODEL LOAD =================
if not os.path.exists(MODEL_FILE):
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_FILE)

@st.cache_resource
def load_cnn():
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
    class_id = np.argmax(preds)
    confidence = float(np.max(preds))
    emergency = confidence >= CONF_THRESHOLD
    return CLASS_NAMES[class_id], confidence, emergency

# ================= UPLOAD UI =================
st.markdown("""
<style>
div[data-testid="stFileUploader"] label {display:none;}
div[data-testid="stFileUploader"] section {
    padding:0; border:none; background:transparent;
}
div[data-testid="stFileUploader"] section button {
    width:60px;
    height:60px;
    border-radius:50%;
    font-size:30px;
    background:#4CAF50;
    color:white;
    border:none;
    display:flex;
    justify-content:center;
    align-items:center;
}
div[data-testid="stFileUploader"] section button:hover {
    background:#43a047;
}
div[data-testid="stFileUploader"] section button p {display:none;}
</style>
""", unsafe_allow_html=True)

uploaded = st.file_uploader(
    "+",
    type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
)

# ================= HANDLE FILE =================
if uploaded:
    st.success(f"Uploaded: {uploaded.name}")

    # -------- IMAGE --------
    if uploaded.type.startswith("image"):
        img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), 1)
        st.image(img, channels="BGR")

        if st.button("🔍 Predict"):
            cls, conf, emg = predict(img)
            st.metric("Prediction", cls)
            st.metric("Confidence", f"{conf*100:.2f}%")
            st.error("🚨 EMERGENCY") if emg else st.success("✅ Normal")

    # -------- VIDEO --------
    elif uploaded.type.startswith("video"):
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded.read())

        if st.button("▶ Analyze Video"):
            cap = cv2.VideoCapture(tfile.name)
            frame_box = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                cls, conf, emg = predict(frame)
                label = f"{cls} ({conf*100:.1f}%)"
                color = (0,0,255) if emg else (0,255,0)

                cv2.putText(frame, label, (20,40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                frame_box.image(frame, channels="BGR")

            cap.release()
