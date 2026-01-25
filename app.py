import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import os
import av
import numpy as np
import tempfile
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

# ===================== MODEL DOWNLOAD =====================
def download_model():
    if not os.path.exists(MODEL_FILE):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_FILE, quiet=False)

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

# ===================== PREDICTION =====================
def predict_anomaly(img):
    inp = preprocess_image(img)
    preds = model.predict(inp, verbose=0)
    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds))
    emergency = 1 if confidence >= CONF_THRESHOLD else 0
    return {"class": CLASS_NAMES[class_id], "confidence": confidence, "emergency": emergency}

# ===================== RTC CONFIG =====================
RTC_CONFIG = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# ===================== VIDEO PROCESSOR =====================
class AnomalyProcessor(VideoProcessorBase):
    def __init__(self):
        self.current_result = {}

    def recv(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")
        result = predict_anomaly(img)

        label = f"{result['class']} ({result['confidence']*100:.1f}%)"
        color = (0, 0, 255) if result["emergency"] else (0, 255, 0)

        cv2.putText(img, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(img, f"EMERGENCY: {result['emergency']}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        self.current_result = result
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ===================== SIDEBAR =====================
mode = st.sidebar.radio("Select Mode", ["📡 Live Webcam", "⬆ Upload", "ℹ About"])

# ===================== ABOUT =====================
if mode == "ℹ About":
    st.subheader("About")
    st.markdown("""
    **Road Anomaly Detection System**  
    Developer: Vijay Ragavan  
    College: Kamarajar Engineering College of Technology
    """)

# ===================== LIVE WEBCAM =====================
elif mode == "📡 Live Webcam":
    st.subheader("Live Road Anomaly Detection")

    ctx = webrtc_streamer(
        key="road-webcam",
        video_processor_factory=AnomalyProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False}
    )

    if ctx and ctx.video_processor:
        res = ctx.video_processor.current_result
        if res:
            st.write("Prediction:", res["class"])
            st.write("Confidence:", f"{res['confidence']*100:.1f}%")

            if res["emergency"]:
                st.error("🚨 EMERGENCY DETECTED")
            else:
                st.success("✅ Normal Condition")

# ===================== UPLOAD =====================
elif mode == "⬆ Upload":

    st.subheader("Upload Image, Video or File")

    uploaded_file = st.file_uploader(
        "➕ Add files",
        type=["jpg", "jpeg", "png", "mp4", "avi", "pdf", "bmp", "webp"],
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        st.success("File uploaded successfully")
        st.write("File name:", uploaded_file.name)
        st.write("File type:", uploaded_file.type)
        st.write("File size:", uploaded_file.size, "bytes")

        # ---------- IMAGE ----------
        if uploaded_file.type.startswith("image"):
            img_bytes = np.asarray(bytearray(uploaded_file.read()), np.uint8)
            img = cv2.imdecode(img_bytes, 1)
            st.image(img, channels="BGR", use_container_width=True)

            if st.button("🔍 Predict Image"):
                result = predict_anomaly(img)
                st.write(result)
                if result["emergency"]:
                    st.error("🚨 EMERGENCY DETECTED")
                else:
                    st.success("✅ Normal Condition")

        # ---------- VIDEO ----------
        elif uploaded_file.type.startswith("video"):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())

            if st.button("▶ Analyze Video"):
                cap = cv2.VideoCapture(tfile.name)
                frame_box = st.empty()

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    result = predict_anomaly(frame)
                    label = f"{result['class']} ({result['confidence']*100:.1f}%)"
                    color = (0, 0, 255) if result["emergency"] else (0, 255, 0)

                    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                    frame_box.image(frame, channels="BGR", use_container_width=True)

                cap.release()

        # ---------- OTHER FILES ----------
        else:
            st.info("File uploaded successfully (non-image/video file)")
