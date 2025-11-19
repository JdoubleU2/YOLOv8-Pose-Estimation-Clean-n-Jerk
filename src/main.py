import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import time

# ===== Configuration =====
MODEL_PATH = '/Users/jabinwade/Coding/ComputerVision/ComputerVision-FinalProject/runs2/powerlifting-pose.pt'  # Update with your model path

# Define class names for clean & jerk phases
CLASS_NAMES = [
    'Drive', 'Drop Under', 'Drop Weight', 'First Pull', 'Front Rack Catch', 
    'Front Rack Recovery', 'Lift Complete', 'Overhead Catch', 'Overhead Catch Recovery', 
    'Prepare For Dip', 'Prepare for Lift', 'Second Pull', 'Stabilize Weight Overhead', 
    'Turn Over'
]

# ===== Initialize Session State =====
if "detection_order" not in st.session_state:
    st.session_state.detection_order = []
if "previously_detected" not in st.session_state:
    st.session_state.previously_detected = set()

# ===== Streamlit App =====
st.set_page_config(layout="wide", page_title="Clean & Jerk Detection Overlay")
st.title("Clean & Jerk Detection Overlay")

# Load model
@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Upload a video", type=['mp4', 'mov', 'avi', 'mkv'])

if uploaded_file is not None:
    # Save uploaded file temporarily
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    tfile.close()
    
    # Open video
    cap = cv2.VideoCapture(tfile.name)
    
    if not cap.isOpened():
        st.error("Error opening video file")
        st.stop()
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    st.info(f"Video loaded: {total_frames} frames @ {fps} FPS")
    
    # Create layout: Left (video), Right (detection list)
    col1, col2 = st.columns([2, 1])
    
    # Placeholders for updating content
    with col1:
        st.subheader("Video Feed")
        frame_placeholder = st.empty()
    
    with col2:
        current_detection_placeholder = st.empty()
        st.subheader("Detected Phases (in order)")
        class_list_placeholder = st.empty()
    
    # Reset detection history button
    if st.button("Reset Detection History"):
        st.session_state.detection_order = []
        st.session_state.previously_detected = set()
        st.rerun()
    
    # Process video frame by frame
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run inference on frame
        results = model(frame, verbose=False)
        
        # Draw bounding boxes/keypoints on frame
        annotated_frame = results[0].plot()
        
        # Extract detected class names for this frame
        detected_classes = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                detected_classes.append(class_name)
        
        # Update session state with detected classes
        for cls in detected_classes:
            if cls not in st.session_state.detection_order:
                st.session_state.detection_order.append(cls)
            st.session_state.previously_detected.add(cls)
        
        # Update the video display
        frame_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
        
        # Update current detection display (large box)
        if detected_classes:
            main_active = detected_classes[0]  # Show first detected class (or use max confidence)
            current_detection_placeholder.markdown(
                f'<div style="text-align:center; font-size:2em; color:#fff; '
                f'background:linear-gradient(135deg, #1565c0 0%, #0d47a1 100%); '
                f'border-radius:12px; padding:20px; margin-bottom:20px; '
                f'box-shadow: 0px 4px 12px rgba(0,0,0,0.3);">'
                f'<b>CURRENT: {main_active}</b>'
                f'</div>',
                unsafe_allow_html=True
            )
        else:
            current_detection_placeholder.markdown(
                f'<div style="text-align:center; font-size:1.5em; color:#999; '
                f'background:#f5f5f5; border-radius:12px; padding:20px; margin-bottom:20px;">'
                f'<b>No Detection</b>'
                f'</div>',
                unsafe_allow_html=True
            )
        
        # Update the detected phases list with styling
        class_list_html = ""
        for idx, cls in enumerate(st.session_state.detection_order, 1):
            is_active = (cls in detected_classes)
            
            # Styling: bright white & bold if active, dull gray if not
            if is_active:
                color = "#fff"
                weight = "900"
                bgcolor = "#263238"
                box_shadow = "0px 0px 14px rgba(30, 136, 229, 0.8)"
                border = "2px solid #1E88E5"
            else:
                color = "#666"
                weight = "400"
                bgcolor = "#f9f9f9"
                box_shadow = "none"
                border = "1px solid #ddd"
            
            class_list_html += (
                f'<div style="font-size:1.1em; color:{color}; font-weight:{weight}; '
                f'background:{bgcolor}; border-radius:8px; margin-bottom:8px; '
                f'padding:12px 16px; box-shadow:{box_shadow}; border:{border};">'
                f'{idx}. <b>{cls}</b>'
                f'</div>'
            )
        
        if class_list_html:
            class_list_placeholder.markdown(class_list_html, unsafe_allow_html=True)
        else:
            class_list_placeholder.markdown("*No phases detected yet*")
        
        # Control playback speed (adjust this value for faster/slower playback)
        time.sleep(0.03)  # ~30 FPS playback
    
    cap.release()
    st.success(f"Video processing complete! Processed {frame_count} frames.")

else:
    st.info("👆 Upload a video file to start detection")
    
    # Show example of what the UI looks like
    st.markdown("---")
    st.subheader("Preview: What the overlay will look like")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.image("https://via.placeholder.com/640x480/1565c0/ffffff?text=Video+Feed", 
                 use_container_width=True)
    
    with col2:
        st.markdown(
            '<div style="text-align:center; font-size:2em; color:#fff; '
            'background:linear-gradient(135deg, #1565c0 0%, #0d47a1 100%); '
            'border-radius:12px; padding:20px; margin-bottom:20px;">'
            '<b>CURRENT: First Pull</b></div>',
            unsafe_allow_html=True
        )
        
        st.write("**Detected Phases (in order)**")
        
        example_phases = ["First Pull", "Second Pull", "Drive"]
        for idx, phase in enumerate(example_phases, 1):
            is_active = (idx == 1)
            if is_active:
                st.markdown(
                    f'<div style="font-size:1.1em; color:#fff; font-weight:900; '
                    f'background:#263238; border-radius:8px; margin-bottom:8px; '
                    f'padding:12px 16px; box-shadow:0px 0px 14px rgba(30, 136, 229, 0.8); '
                    f'border:2px solid #1E88E5;">{idx}. <b>{phase}</b></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div style="font-size:1.1em; color:#666; font-weight:400; '
                    f'background:#f9f9f9; border-radius:8px; margin-bottom:8px; '
                    f'padding:12px 16px; border:1px solid #ddd;">{idx}. <b>{phase}</b></div>',
                    unsafe_allow_html=True
                )
