from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from keras.models import load_model
import gdown
import os

app = Flask(__name__)

# ================= CONFIG =================
MODEL_FILE = "road_anomaly_model.h5"
FILE_ID = "1FiHUDZPL1MFyG1g06_jjM4MJV2tH9rpg"

CLASS_NAMES = ['Accident', 'Fight', 'Fire', 'Snatching']
CONF_THRESHOLD = 0.85
IMG_SIZE = (224, 224)

# ================= DOWNLOAD MODEL =================
if not os.path.exists(MODEL_FILE):
    gdown.download(
        f"https://drive.google.com/uc?id={FILE_ID}",
        MODEL_FILE,
        quiet=False
    )

model = load_model(MODEL_FILE)

# ================= CNN =================
def preprocess(img):
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=0)

def predict(img):
    preds = model.predict(preprocess(img), verbose=0)
    cid = int(np.argmax(preds))
    conf = float(np.max(preds))
    emergency = conf >= CONF_THRESHOLD
    return CLASS_NAMES[cid], conf, emergency

# ================= LIVE CAMERA =================
camera = cv2.VideoCapture(0)

def gen_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break

        label, conf, emergency = predict(frame)
        color = (0,0,255) if emergency else (0,255,0)

        cv2.putText(frame, f"{label} ({conf*100:.1f}%)",
                    (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.putText(frame, f"EMERGENCY: {int(emergency)}",
                    (20,80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

# ================= ROUTES =================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

@app.route("/predict_image", methods=["POST"])
def predict_image():
    file = request.files["image"]
    img = cv2.imdecode(
        np.frombuffer(file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )
    label, conf, emergency = predict(img)
    return jsonify({
        "class": label,
        "confidence": conf,
        "emergency": int(emergency)
    })

# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
