# streamlit_app.py
# VisionTrack-YOLO — Beautiful Streamlit UI (Updated + Warning Fixed)

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import tempfile
from config import MODEL_NAME, CONFIDENCE_THRESHOLD


# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="VisionTrack-YOLO | Object Detection",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    .title-text {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 1rem;
    }
    .subtitle-text {
        font-size: 1.2rem;
        text-align: center;
        color: #f0f0f0;
        margin-bottom: 2rem;
    }
    .card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        border: none;
        font-size: 1.1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }
    .uploadedFile {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 0.5rem 0;
    }
    .metric-label {
        font-size: 1rem;
        opacity: 0.9;
    }
    .stAlert {
        border-radius: 10px;
        border-left: 5px solid #667eea;
    }
    </style>
""", unsafe_allow_html=True)


# ============================================
# TITLE & SUBTITLE
# ============================================
st.markdown("<div class='title-text'>VisionTrack-YOLO</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Advanced Image & Video Object Detection</div>", unsafe_allow_html=True)


# ============================================
# LOAD YOLO MODEL
# ============================================
@st.cache_resource
def load_model():
    return YOLO(MODEL_NAME)

model = load_model()


# ============================================
# SIDEBAR
# ============================================
st.sidebar.header("⚙️ Settings")
mode = st.sidebar.radio("Select Mode:", ["Image Detection", "Video Detection"])
conf = st.sidebar.slider("Confidence Threshold:", 0.1, 1.0, CONFIDENCE_THRESHOLD, 0.05)


# ============================================
# IMAGE DETECTION
# ============================================
if mode == "Image Detection":
    st.subheader("📸 Upload Image")

    uploaded = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])

    if uploaded:
        img = Image.open(uploaded)
        img_np = np.array(img)

        st.image(img, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Detect Objects"):
            results = model(img_np, conf=conf)
            annotated = results[0].plot()

            st.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                caption="Detected Objects",
                use_container_width=True
            )


# ============================================
# VIDEO DETECTION
# ============================================
elif mode == "Video Detection":
    st.subheader("🎞️ Upload Video")

    uploaded_video = st.file_uploader("Choose a video", type=['mp4', 'avi', 'mov'])

    if uploaded_video:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_video.read())

        st.video(temp_file.name)

        if st.button("🔍 Run Video Detection"):
            cap = cv2.VideoCapture(temp_file.name)
            frames = []

            progress = st.progress(0)

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_num = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, conf=conf)
                annotated = results[0].plot()
                frames.append(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))

                frame_num += 1
                progress.progress(frame_num / total_frames)

            cap.release()

            st.success("Video Processing Complete 🎉")
            st.write("Showing detected frames:")

            for f in frames[::max(1, len(frames)//10)]:
                st.image(f, use_container_width=True)
