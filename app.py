import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import time

MODEL_PATH = '/Users/jabinwade/Coding/ComputerVision/ComputerVision-FinalProject/powerlifting-pose.pt'
CLASS_NAMES = [] 

if "detection_order" not in st.session_state:
    st.session_state.detection_order = []
if "previously_detected" not in st.session_state:
    st.session_state.previously_detected = set()

st.set_page_config(layout="wide", page_title="Clean & Jerk Detection Overlay")
st.title("Clean & Jerk Detection Overlay")
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload a video", type=['mp4', 'mov', 'avi', 'mkv'])
if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()
    cap = cv2.VideoCapture(tfile.name)
    if not cap.isOpened():
        st.error("Error opening video file")
        st.stop()
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    st.info(f"Video loaded: {total_frames} frames @ {fps} FPS")

    col1, col2 = st.columns([2, 1])
    max_video_width = 600
    with col1:
        current_detection_placeholder = st.empty()    
        st.subheader("Video Feed")
        frame_placeholder = st.empty()

    with col2:
        st.subheader("Detected Phases")
        class_list_placeholder = st.empty()

    if st.button("Reset Detection History"):
        st.session_state.detection_order = []
        st.session_state.previously_detected = set()
        st.rerun()
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        h, w, _ = frame.shape
        if w > max_video_width:
            scale = max_video_width / w
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()
        detected_classes = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                detected_classes.append(class_name)
        for cls in detected_classes:
            if cls not in st.session_state.detection_order:
                st.session_state.detection_order.append(cls)
            st.session_state.previously_detected.add(cls)
        # Current class detection box now above the video!
        if detected_classes:
            main_active = detected_classes[0]
            current_detection_placeholder.markdown(
                f'<div style="text-align:center; font-size:1.5em; color:#fff; '
                f'background:linear-gradient(135deg, #1565c0 0%, #0d47a1 100%); '
                f'border-radius:8px; padding:12px; margin-bottom:8px;">'
                f'<b>CURRENT: {main_active}</b></div>',
                unsafe_allow_html=True
            )
        else:
            current_detection_placeholder.markdown(
                f'<div style="text-align:center; font-size:1.1em; color:#999; '
                f'background:#f5f5f5; border-radius:8px; padding:12px; margin-bottom:8px;">'
                f'<b>No Detection</b></div>',
                unsafe_allow_html=True
            )
        frame_placeholder.image(annotated_frame, channels="BGR", width=max_video_width)
        # Detected phases
        class_list_html = ""
        for idx, cls in enumerate(st.session_state.detection_order, 1):
            is_active = (cls in detected_classes)
            color = "#fff" if is_active else "#333"
            weight = "800" if is_active else "400"
            bgcolor = "#1565c0" if is_active else "#f9f9f9"
            border = "2px solid #1565c0" if is_active else "1px solid #ddd"
            class_list_html += (
                f'<div style="font-size:1em; color:{color}; font-weight:{weight}; '
                f'background:{bgcolor}; border-radius:4px; margin-bottom:4px; padding:8px; border:{border};">'
                f'{idx}. <b>{cls}</b></div>'
            )
        if class_list_html:
            class_list_placeholder.markdown(class_list_html, unsafe_allow_html=True)
        else:
            class_list_placeholder.markdown("*No phases detected yet*")
        time.sleep(0.03)
    cap.release()
    st.success(f"Video processing complete! Processed {frame_count} frames.")

else:
    st.info("Upload a video file to start detection")

