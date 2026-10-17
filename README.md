
# Vehicle Detection, Tracking, Speed Estimation & Traffic Forecasting Project

This package contains scripts to run:
- Detection + Tracking + Speed Estimation (Option B)
- Traffic Forecasting (Option C)
- Homography calibration helper, and aggregation utility

## Files
- Detection_Tracking_Speed.py : Run detection+tracking and save overspeed evidence and per-frame counts.
- Traffic_Forecasting.py    : Train an LSTM to forecast aggregated traffic counts.
- compute_homography.py     : Interactive tool to compute homography (image -> ground meters).
- aggregate_counts.py      : Aggregate raw per-frame counts into fixed intervals (e.g., 15 minutes).
- requirements.txt         : Python dependencies list.

## Quick start
1. Create virtualenv and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Compute homography (example):
   ```bash
   python compute_homography.py --image data/frame_sample.jpg --out_homography data/H.npy
   ```
   Click 4 points and enter their real-world coordinates (meters).

3. Run detection+tracking (example):
   ```bash
   python Detection_Tracking_Speed.py --video data/myvideo.mp4 --yolo_model yolov8n.pt --homography data/H.npy --output out.mp4 --output_dir data --evidence_dir evidence --speed_limit 50 --speed_tolerance 5 --display 0
   ```
   The script will also write `data/counts_raw_perframe.csv` with per-frame counts.

4. Aggregate counts to 15-min windows:
   ```bash
   python aggregate_counts.py --raw data/counts_raw_perframe.csv --out data/counts_15min.csv --freq 15T
   ```

5. Forecast:
   ```bash
   python Traffic_Forecasting.py --counts_csv data/counts_15min.csv --freq 15T --seq_len 12 --epochs 50
   ```

## Notes
- Provide `yolov8n.pt` or specify another Ultralytics model (it will auto-download if available).
- Homography calibration is crucial for speed accuracy.
- Check local laws before collecting/persisting license plates or issuing fines.

