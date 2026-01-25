import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import gdown
from keras.models import load_model

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="Road Anomaly Detection", layout="wide")
st.title("Road Anomaly Detection using CNN")

# ===================== CONSTANTS =====================
MODEL_FILE = "road_anomaly_model.h5"
FILE_ID = "1FiHUDZPL1MFyG1g06_jjM4MJV2tH9rpg"
CLASS_NAMES = ['Accident', 'Fight', 'Fire', 'Snatching']
CONF_THRESHOLD = 0.85
IMG_SIZE = (224, 224)

# ===================== DOWNLOAD MODEL =====================
def download_model():
    if not os.path.exists(MODEL_FILE):
        gdown.download(
            f"https://drive.google.com/uc?id={FILE_ID}",
            MODEL_FILE,
            quiet=False
        )

download_model()

@st.cache_resource
def load_cnn_model():
    return load_model(MODEL_FILE)

model = load_cnn_model()

# ===================== PREPROCESS =====================
def preprocess_image(img):
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

# ===================== PREDICT =====================
def predict_anomaly(img):
    inp = preprocess_image(img)
    preds = model.predict(inp, verbose=0)
    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds))
    emergency = 1 if confidence >= CONF_THRESHOLD else 0

    return {
        "class": CLASS_NAMES[class_id],
        "confidence": confidence,
        "emergency": emergency
    }

# ===================== SIDEBAR =====================
mode = st.sidebar.radio(
    "Select Mode",
    ["Upload Image", "Upload Video", "Upload File", "About"]
)

# ===================== ABOUT =====================
if mode == "About":
    st.subheader("About")
    st.markdown("""
    **Project:** Road Anomaly Detection  
    **Developer:** Vijay Ragavan  
    **Description:** CNN-based system to detect road anomalies such as
    accidents, fire, fights, and snatching.
    """)

# ===================== IMAGE UPLOAD =====================
elif mode == "Upload Image":
    st.subheader("Upload Image")

    img_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png"]
    )

    if img_file:
        img_bytes = np.asarray(bytearray(img_file.read()), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        st.image(img, channels="BGR", caption="Uploaded Image")

        if st.button("Predict Image"):
            result = predict_anomaly(img)

            st.write("Class:", result["class"])
            st.write("Confidence:", f"{result['confidence']*100:.2f}%")

            if result["emergency"]:
                st.error("🚨 EMERGENCY DETECTED")
            else:
                st.success("✅ Normal Condition")

# ===================== VIDEO UPLOAD =====================
elif mode == "Upload Video":
    st.subheader("Upload Video")

    video_file = st.file_uploader(
        "Choose a video",
        type=["mp4", "avi", "mov"]
    )

    if video_file:
        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(video_file.read())

        if st.button("Analyze Video"):
            cap = cv2.VideoCapture(temp_video.name)
            frame_area = st.empty()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                result = predict_anomaly(frame)
                label = f"{result['class']} ({result['confidence']*100:.1f}%)"

                color = (0, 0, 255) if result["emergency"] else (0, 255, 0)

                cv2.putText(
                    frame, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
                )

                frame_area.image(
                    frame, channels="BGR", use_container_width=True
                )

            cap.release()

# ===================== FILE UPLOAD =====================
elif mode == "Upload File":
    st.subheader("Upload File")

    file = st.file_uploader(
        "Upload any file",
        type=["csv", "txt", "xlsx", "pdf"]
    )

    if file:
        st.write("File Name:", file.name)
        st.write("File Size:", file.size, "bytes")

        if st.button("Process File"):
            st.success("File uploaded successfully")
