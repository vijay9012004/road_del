import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2, os, av, numpy as np, tempfile, gdown
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

# ================= MODEL LOAD (SAFE) =================
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
    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds))
    emergency = confidence >= CONF_THRESHOLD
    return CLASS_NAMES[class_id], confidence, emergency

# ================= RTC CONFIG =================
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ================= VIDEO PROCESSOR =================
class AnomalyProcessor(VideoProcessorBase):
    def __init__(self):
        self.current_result = None

    def recv(self, frame: av.VideoFrame):
        img = frame.to_ndarray(format="bgr24")
        cls, conf, emg = predict(img)

        label = f"{cls} ({conf*100:.1f}%)"
        color = (0, 0, 255) if emg else (0, 255, 0)

        cv2.putText(img, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(img, f"EMERGENCY: {int(emg)}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        self.current_result = {
            "class": cls,
            "confidence": conf,
            "emergency": emg
        }
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ================= SIDEBAR =================
st.sidebar.title("🚘 Navigation")
mode = st.sidebar.radio(
    "Select Mode",
    ["Live Webcam", "Upload ( + )", "About"]
)

# ================= ABOUT =================
if mode == "About":
    st.subheader("About This Project")
    st.markdown("""
    **Road Anomaly Detection System**  
    Developer: Vijay Ragavan  
    College: Kamarajar Engineering College of Technology  

    CNN-based detection of:
    • Accident  
    • Fight  
    • Fire  
    • Snatching
    """)

# ================= LIVE WEBCAM =================
elif mode == "Live Webcam":
    st.subheader("📡 Live Road Monitoring")

    ctx = webrtc_streamer(
        key="road-webcam",
        video_processor_factory=AnomalyProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=False,
    )

    if ctx and ctx.video_processor and ctx.video_processor.current_result:
        res = ctx.video_processor.current_result
        st.write("### Live Prediction")
        st.write("Class:", res["class"])
        st.write("Confidence:", f"{res['confidence']*100:.2f}%")
        st.error("🚨 EMERGENCY") if res["emergency"] else st.success("✅ Normal")

# ================= UPLOAD UI =================
elif mode == "Upload ( + )":
    st.subheader("⬆ Upload Image or Video")

    st.markdown("""
    <style>
    div[data-testid="stFileUploader"] label {display:none;}
    div[data-testid="stFileUploader"] section button {
        width:60px; height:60px;
        border-radius:50%;
        font-size:30px;
        background:#4CAF50;
        color:white;
        border:none;
        margin:auto;
    }
    </style>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "+",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]
    )

    if uploaded:
        file_bytes = uploaded.read()  # ✅ read ONCE
        st.success(f"Uploaded: {uploaded.name}")

        # -------- IMAGE --------
        if uploaded.type.startswith("image"):
            img = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), 1)
            st.image(img, channels="BGR", use_container_width=True)

            if st.button("🔍 Predict Image"):
                cls, conf, emg = predict(img)
                st.metric("Prediction", cls)
                st.metric("Confidence", f"{conf*100:.2f}%")
                st.error("🚨 EMERGENCY") if emg else st.success("✅ Normal")

        # -------- VIDEO --------
        elif uploaded.type.startswith("video"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                tfile.write(file_bytes)
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
