import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import tempfile
import random
import os

# --- Page Config ---
st.set_page_config(page_title="DeepSORT Traffic Pro", layout="wide", page_icon="📹")

# --- Custom CSS ---
st.markdown("""
<style>
    .metric-card { background-color: #f0f2f6; border-radius: 10px; padding: 15px; }
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if 'unique_ids' not in st.session_state:
    st.session_state.unique_ids = set()

# --- Load Models ---
@st.cache_resource
def load_models():
    # Load YOLO for Object Detection
    model = YOLO('yolov8n.pt') 
    # Initialize DeepSORT
    tracker = DeepSort(max_age=30, n_init=3, nms_max_overlap=1.0)
    return model, tracker

try:
    with st.spinner("Initializing AI Models..."):
        model, tracker = load_models()
        class_names = model.names
except Exception as e:
    st.error(f"Failed to load models: {e}")
    st.stop()

# --- Sidebar ---
st.sidebar.title("⚙️ Settings")
conf_threshold = st.sidebar.slider("Detection Confidence", 0.1, 1.0, 0.45)
speed_limit = st.sidebar.number_input("Speed Limit (km/h)", 50, 150, 80)

if st.sidebar.button("Reset Counter"):
    st.session_state.unique_ids = set()
    st.rerun()

# --- Helper Functions ---
def get_simulated_speed(vehicle_type):
    base = random.uniform(60, 110)
    if vehicle_type in ['truck', 'bus']: base = random.uniform(50, 90)
    if vehicle_type == 'motorcycle': base = random.uniform(70, 120)
    return round(base, 1)

def is_lane_violation(center_x, frame_width):
    return center_x > (frame_width * 0.85)

def draw_info(frame, ltrb, track_id, vehicle_type, speed, overspeed, lane_viol):
    x1, y1, x2, y2 = map(int, ltrb)
    color = (0, 0, 255) if (overspeed or lane_viol) else (0, 255, 0)
    
    # Bounding Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Label
    label = f"ID:{track_id} {vehicle_type}"
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    
    # Speed & Alerts
    info_y = y2 + 25
    cv2.putText(frame, f"{speed} km/h", (x1, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    if overspeed:
        info_y += 25
        cv2.putText(frame, "OVERSPEED!", (x1, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if lane_viol:
        info_y += 25
        cv2.putText(frame, "LANE VIOLATION", (x1, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return frame

# --- Main App ---
st.title("📹 DeepSORT Traffic Analyzer & Recorder")

uploaded_file = st.file_uploader("Upload Video (mp4, avi)", type=['mp4', 'avi', 'mov'])

if uploaded_file:
    # 1. Setup Input Video
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)
    
    # 2. Setup Output Video Writer
    output_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = output_temp_file.name
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Define codec and create VideoWriter
    # 'mp4v' is widely supported for .mp4 containers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Layout
    col_metrics, col_video = st.columns([1, 3])
    with col_metrics:
        st.markdown("### 📊 Live Stats")
        kpi_total = st.empty()
        kpi_current = st.empty()
        st.info("Processing video... Please wait for the loop to finish to download the result.")
        stop_btn = st.button("Stop & Save")

    st_frame = col_video.empty()

    # Processing Loop
    while cap.isOpened() and not stop_btn:
        success, frame = cap.read()
        if not success:
            break
            
        # YOLO Detection
        results = model.predict(frame, conf=conf_threshold, classes=[2, 3, 5, 7], verbose=False)
        
        # Format for DeepSORT
        detections = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            w = x2 - x1
            h = y2 - y1
            detections.append(([x1, y1, w, h], confidence, class_names[class_id]))

        # DeepSORT Tracking
        tracks = tracker.update_tracks(detections, frame=frame)
        current_count = 0
        
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            current_count += 1
            track_id = track.track_id
            st.session_state.unique_ids.add(track_id)
            
            vehicle_type = track.get_det_class()
            if vehicle_type is None: vehicle_type = "Vehicle"
            
            ltrb = track.to_ltrb()
            
            # Draw Logic
            center_x = ltrb[0] + ((ltrb[2] - ltrb[0]) / 2)
            sim_speed = get_simulated_speed(vehicle_type)
            is_over = sim_speed > speed_limit
            is_viol = is_lane_violation(center_x, width)
            
            frame = draw_info(frame, ltrb, track_id, vehicle_type.title(), sim_speed, is_over, is_viol)

        # Write frame to output video
        out.write(frame)

        # Update Live UI
        kpi_total.metric("Total Unique Vehicles", len(st.session_state.unique_ids))
        kpi_current.metric("Vehicles in Frame", current_count)
        st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

    # Cleanup
    cap.release()
    out.release()
    tracker.delete_all_tracks()
    
    st.success("Processing Complete! Video saved.")
    
    # 3. Create Download Button
    # We open the saved file in binary mode and offer it for download
    with open(output_path, 'rb') as f:
        video_bytes = f.read()
        st.download_button(
            label="⬇️ Download Annotated Video",
            data=video_bytes,
            file_name="traffic_analysis_output.mp4",
            mime="video/mp4"
        )
    
    # Optional: Clean up temp file (uncomment if you want to save space)
    # os.remove(output_path)

else:
    st.info("Please upload a video file to start.")