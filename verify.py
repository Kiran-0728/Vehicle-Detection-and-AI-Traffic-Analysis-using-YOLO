# verify_installation.py

print("🚦 Traffic AI System - Installation Verification Script")

# -----------------------------
# Check Python Version
# -----------------------------
import sys
print(f"\n✅ Python Version: {sys.version}")

# -----------------------------
# Check Ultralytics / YOLOv8
# -----------------------------
try:
    from ultralytics import YOLO
    print("✅ YOLOv8 installed successfully")
    model = YOLO("yolov8n.pt")  # downloads model if missing
    print("✅ YOLO model loaded successfully")
except Exception as e:
    print("❌ YOLOv8 error:", e)

# -----------------------------
# Check OpenCV
# -----------------------------
try:
    import cv2
    print(f"✅ OpenCV installed (version: {cv2.__version__})")
except Exception as e:
    print("❌ OpenCV error:", e)

# -----------------------------
# Check OCR (EasyOCR & Tesseract)
# -----------------------------
# EasyOCR
try:
    import easyocr
    print("✅ EasyOCR installed successfully")
except Exception as e:
    print("❌ EasyOCR error:", e)

# Tesseract (optional)
try:
    import pytesseract
    print("✅ Tesseract installed successfully")
except Exception:
    print("⚠️ Tesseract not detected (optional, used only if enabled)")

# -----------------------------
# Check Streamlit
# -----------------------------
try:
    import streamlit
    print("✅ Streamlit installed successfully")
except Exception as e:
    print("❌ Streamlit error:", e)

# -----------------------------
# Check GPU / Torch
# -----------------------------
try:
    import torch
    if torch.cuda.is_available():
        print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Running on CPU (GPU not detected)")
except Exception as e:
    print("❌ Torch error:", e)

print("\n🎉 Verification Completed!")
print("If there are ❌ errors, install missing libraries using:")
print("\npip install -r requirements.txt")

