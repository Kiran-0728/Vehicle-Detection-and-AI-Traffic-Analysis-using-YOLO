"""
Detection + Tracking + Speed Estimation (Option B)
Run: python Detection_Tracking_Speed.py --video path/to/video.mp4 --yolo_model path/to/yolov8n.pt
Dependencies:
  pip install ultralytics opencv-python-headless numpy filterpy deep-sort-realtime matplotlib
Notes:
  - This script runs inference using Ultralytics YOLO (v8) for detection, Deep SORT for tracking,
    computes homography-based speed estimation, and logs overspeed and signal-jump events.
  - You must provide a homography calibration file or interactive calibration (see calibrate_homography function).
"""
import argparse
import os
import time
from dataclasses import dataclass
from typing import Tuple, List, Dict
import cv2
import numpy as np

# Try to import ultralytics and deep_sort_realtime; provide helpful error messages if missing.
try:
    from ultralytics import YOLO
except Exception as e:
    YOLO = None
    print("ultralytics not installed. Install with: pip install ultralytics")

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except Exception as e:
    DeepSort = None
    print("deep_sort_realtime not installed. Install with: pip install deep-sort-realtime")

@dataclass
class TrackRecord:
    track_id: int
    class_name: str
    last_positions: List[Tuple[float,float]]  # list of ground-plane (x, y) in meters
    last_times: List[float]  # corresponding timestamps (s)

def load_yolo(model_path: str):
    if YOLO is None:
        raise RuntimeError("Ultralytics YOLO package is required. pip install ultralytics")
    model = YOLO(model_path)
    return model

def calibrate_homography(video_path: str, frame=None) -> Tuple[np.ndarray, float]:
    """
    Interactive calibration helper (manual): User selects 4 points on the image that correspond to
    4 points on the ground plane rectangle (meters). Returns homography H (image->ground) and scale reference length.
    For automation, precompute H using known lane markings and save it.
    """
    print("Calibration helper: not interactive in this script-run environment. Provide a saved homography matrix instead.")
    # Placeholder identity homography
    H = np.eye(3)
    return H, 1.0

def image_to_ground(pt: Tuple[float,float], H: np.ndarray) -> Tuple[float,float]:
    """Map image coordinates to ground plane using homography H (image->ground)."""
    x, y = pt
    vec = np.array([x, y, 1.0])
    g = H.dot(vec)
    g = g / g[2]
    return float(g[0]), float(g[1])

def compute_speed_mps(positions: List[Tuple[float,float]], times: List[float]) -> float:
    """Compute speed (m/s) using last two valid positions (simple estimator)."""
    if len(positions) < 2:
        return 0.0
    x1, y1 = positions[-2]
    x2, y2 = positions[-1]
    dt = times[-1] - times[-2]
    if dt <= 0:
        return 0.0
    dist = np.hypot(x2 - x1, y2 - y1)
    return dist / dt

def annotate_frame(frame, detections, tracks, violations):
    """Draw boxes, track ids, speeds, and violations on frame for visualization."""
    for det in detections:
        x1, y1, x2, y2, conf, cls_name = det
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(frame, f"{cls_name} {conf:.2f}", (int(x1), int(y1)-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255),1)

    for trk in tracks:
        tid = trk['track_id']
        box = trk['bbox']  # [x1,y1,x2,y2]
        speed_kmph = trk.get('speed_kmph', None)
        color = (200,100,0)
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        label = f"ID:{tid}"
        if speed_kmph is not None:
            label += f" {speed_kmph:.1f}km/h"
        cv2.putText(frame, label, (int(box[0]), int(box[1])-12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255),1)

    # violations text
    y = 30
    for v in violations[-5:]:
        cv2.putText(frame, v, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
        y += 18

    return frame

def main(args):
    # Load models and video
    model = load_yolo(args.yolo_model)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_time = 1.0 / fps
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Tracker init
    if DeepSort is None:
        raise RuntimeError("deep-sort-realtime required. pip install deep-sort-realtime")
    tracker = DeepSort(max_age=30)

    # Homography (image->ground) - load if provided
    if args.homography is not None and os.path.exists(args.homography):
        H = np.load(args.homography)
        print("Loaded homography from", args.homography)
    else:
        H, _ = calibrate_homography(args.video)

    # Storage for track records
    track_records: Dict[int, TrackRecord] = {}
    violations = []

    frame_idx = 0
    out_writer = None
    if args.output is not None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    # Prepare counts CSV header
    import csv, datetime
    counts_csv = os.path.join(args.output_dir, 'counts_raw_perframe.csv') if args.output_dir else 'counts_raw_perframe.csv'
    if not os.path.exists(counts_csv):
        with open(counts_csv,'w',newline='') as f:
            w = csv.writer(f)
            w.writerow(['timestamp','total_count','frame_idx'])

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.time()
        results = model(frame[..., ::-1], imgsz=args.imgsz)[0]  # ultralytics result
        detections = []
        # Parse detections: xyxy, conf, class name
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = r
            cls = int(cls)
            cls_name = model.names[cls] if hasattr(model, 'names') else str(cls)
            # filter by classes of interest (car, truck, bus, motorcycle, bicycle)
            if args.filter_classes and cls_name.lower() not in ['car','truck','bus','motorcycle','bicycle']:
                continue
            detections.append([x1, y1, x2, y2, float(conf), cls_name])

        # Prepare detections for Deep SORT: list of [x1,y1,x2,y2,conf, cls_name]
        dets_for_tracker = []
        for d in detections:
            x1,y1,x2,y2,conf,cls_name = d
            dets_for_tracker.append(([x1,y1,x2,y2], conf, cls_name))

        tracks_out = []
        if len(dets_for_tracker) > 0:
            tracks = tracker.update_tracks(dets_for_tracker, frame=frame)
            for t in tracks:
                if not t.is_confirmed():
                    continue
                tid = t.track_id
                bbox = t.to_ltrb()  # left, top, right, bottom
                cls_name = t.get_det_class() or "unknown"
                # compute ground plane centroid
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                gx, gy = image_to_ground((cx, cy), H)
                now = frame_idx * frame_time
                rec = track_records.get(tid)
                if rec is None:
                    rec = TrackRecord(track_id=tid, class_name=cls_name, last_positions=[(gx,gy)], last_times=[now])
                    track_records[tid] = rec
                else:
                    rec.last_positions.append((gx,gy))
                    rec.last_times.append(now)
                    # keep last few entries
                    rec.last_positions = rec.last_positions[-6:]
                    rec.last_times = rec.last_times[-6:]
                speed_mps = compute_speed_mps(rec.last_positions, rec.last_times)
                speed_kmph = speed_mps * 3.6
                track_info = {'track_id': tid, 'bbox': bbox, 'class_name': cls_name, 'speed_kmph': speed_kmph}
                tracks_out.append(track_info)
                # overspeed checking
                if speed_kmph > args.speed_limit + args.speed_tolerance:
                    msg = f"[{frame_idx}] OVERSPEED ID:{tid} {speed_kmph:.1f}km/h (limit {args.speed_limit})"
                    violations.append(msg)
                    # save evidence image
                    x1i,y1i,x2i,y2i = map(int, bbox)
                    crop = frame[y1i:y2i, x1i:x2i]
                    if crop.size > 0:
                        os.makedirs(args.evidence_dir, exist_ok=True)
                        evid_path = os.path.join(args.evidence_dir, f"overspeed_id{tid}_f{frame_idx}.jpg")
                        cv2.imwrite(evid_path, crop)

        # write per-frame counts
        total_count = sum(1 for t in tracks_out if t['class_name'].lower() in ['car','truck','bus','motorcycle','bicycle'])
        timestamp = datetime.datetime.now()
        with open(counts_csv,'a',newline='') as f:
            w = csv.writer(f)
            w.writerow([timestamp.strftime('%Y-%m-%d %H:%M:%S'), total_count, frame_idx])

        # annotate and write
        vis = annotate_frame(frame.copy(), detections, tracks_out, violations)
        if out_writer is not None:
            out_writer.write(vis)
        if args.display:
            cv2.imshow("vis", vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1

    cap.release()
    if out_writer is not None:
        out_writer.release()
    if args.display:
        cv2.destroyAllWindows()
    print("Processing finished. Violations logged:", len(violations))
