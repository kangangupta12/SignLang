import os
import sys
import tempfile
from pathlib import Path
from collections import deque

import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import CONFIG
from src.utils import load_label_encoder, extract_frames_from_video, overlay_text
from src.model import build_model

st.set_page_config(page_title="Sign Language Recognition", layout="wide")
st.title("Sign Language Recognition Demo")
st.write("CNN + LSTM hybrid model for dynamic sign language gestures.")

label_map_path = os.path.join(CONFIG.artifacts_dir, "label_encoder.json")
if not os.path.exists(label_map_path):
    st.error("Label encoder not found. Train the model first.")
    st.stop()
label_map = load_label_encoder(label_map_path)
inv_label_map = {idx: cls for cls, idx in label_map.items()}

@st.cache_resource
def load_model_cached(path):
    model = build_model(num_classes=len(label_map))
    if path:
        model.load_weights(path)
    return model


def list_models():
    models_dir = "models"
    if not os.path.exists(models_dir):
        return []
    return [os.path.join(models_dir, f) for f in os.listdir(models_dir) if f.endswith((".h5", ".ckpt"))]

model_choice = st.sidebar.selectbox("Select model", options=["Random init"] + list_models())
uploaded_model = st.sidebar.file_uploader("Or upload a Keras model", type=["h5", "ckpt"])

if uploaded_model:
    tmp_model = tempfile.NamedTemporaryFile(delete=False)
    tmp_model.write(uploaded_model.read())
    tmp_model.close()
    model_path = tmp_model.name
elif model_choice != "Random init":
    model_path = model_choice
else:
    model_path = None

model = load_model_cached(model_path)

st.header("Webcam Inference")

class SignLangTransformer(VideoTransformerBase):
    def __init__(self):
        self.buffer = deque(maxlen=CONFIG.seq_len)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (CONFIG.image_size, CONFIG.image_size))
        self.buffer.append(resized / 255.0)
        display = img
        if len(self.buffer) == CONFIG.seq_len:
            tensor = np.array(self.buffer)
            preds = model.predict(tensor[None, ...], verbose=0)[0]
            idx = int(np.argmax(preds))
            label = inv_label_map[idx]
            prob = preds[idx]
            annotated = overlay_text(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), f"{label}: {prob:.2f}")
            return cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        return display

webrtc_streamer(
    key="signlang",
    video_processor_factory=SignLangTransformer,
    media_stream_constraints={"video": True, "audio": False},
)

st.header("Upload Video")
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
if uploaded_video:
    tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_video.write(uploaded_video.read())
    tmp_video.close()
    frames = extract_frames_from_video(tmp_video.name, CONFIG.seq_len, (CONFIG.image_size, CONFIG.image_size))
    preds = model.predict(frames[None, ...], verbose=0)[0]
    top3 = preds.argsort()[-3:][::-1]
    st.subheader("Top predictions")
    for idx in top3:
        st.write(f"{inv_label_map[idx]}: {preds[idx]:.2f}")
    st.video(tmp_video.name)

st.header("Demo Samples")
sample_dir = "examples"
if os.path.exists(sample_dir):
    sample_videos = [f for f in os.listdir(sample_dir) if f.endswith(".mp4")]
    if sample_videos:
        cols = st.columns(len(sample_videos))
        for col, video in zip(cols, sample_videos):
            col.write(video)
            col.video(os.path.join(sample_dir, video))
    else:
        st.info("Place .mp4 demos inside `examples/` to showcase here.")
else:
    st.info("Add the `examples/` directory with sample videos for demos.")

st.sidebar.header("Instructions")
st.sidebar.markdown(
    """
1. Train the model or load provided weights.
2. Ensure `artifacts/label_encoder.json` matches the checkpoint.
3. Use webcam or upload a video to run inference.
"""
)
