# visiontrack.py — UPGRADED HIGH ACCURACY VERSION

import os
import time
import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
import torch

from config import (
    MODEL_NAME, CONFIDENCE_THRESHOLD, IOU_THRESHOLD, DEVICE,
    OUTPUT_DIR, SAVE_VIDEO, VIDEO_FPS,
    ENABLE_COUNTING, COUNT_TARGET_CLASSES,
    ENABLE_ZONE_ALERT, ZONE_POLYGON, ALERT_CLASSES,
    LIMIT_CLASSES
)

from utils import (
    ensure_dir, draw_fps, draw_counts, save_frame,
    draw_zone, point_in_poly, init_video_writer
)


# ---------------------------------------------------------
# Argument Parser
# ---------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, default="0", help="0 = webcam OR path to video/image")
    p.add_argument("--save", action="store_true", help="Save output video/frames")
    p.add_argument("--show", action="store_true", default=True, help="Show detection window")
    return p.parse_args()


# ---------------------------------------------------------
# Load YOLO Model
# ---------------------------------------------------------
def load_model():
    print(f"[INFO] Loading YOLOv8 model: {MODEL_NAME}")
    model = YOLO(MODEL_NAME)

    # Fuse model for speed
    try:
        model.fuse()
    except:
        pass

    return model


# ---------------------------------------------------------
# Main Detection Function
# ---------------------------------------------------------
def main():
    args = parse_args()

    # Select camera or file as source
    source = args.source
    if source.isnumeric():
        source = int(source)

    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        raise RuntimeError(f"❌ Cannot open source: {args.source}")

    # First frame check
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("❌ Unable to read first frame")

    frame_h, frame_w = frame.shape[:2]
    print(f"[INFO] Frame size: {frame_w} x {frame_h}")

    ensure_dir(OUTPUT_DIR)

    # Setup output video writer
    out_writer = None
    if args.save and SAVE_VIDEO:
        output_path = str(Path(OUTPUT_DIR) / f"output_{int(time.time())}.mp4")
        out_writer = init_video_writer(output_path, fps=VIDEO_FPS,
                                       frame_size=(frame_w, frame_h))
        print(f"[INFO] 🎥 Output video saved at: {output_path}")

    # Load YOLO model
    device = "cuda" if (torch.cuda.is_available() and DEVICE == "cuda") else "cpu"
    model = load_model()

    prev_time = time.time()
    total_counts = {}

    print("[INFO] ✔ Starting real-time detection...")

    # ---------------------------------------------------------
    # REAL-TIME LOOP
    # ---------------------------------------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of stream.")
            break

        # YOLO detection
        results = model.predict(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            iou=IOU_THRESHOLD,
            classes=LIMIT_CLASSES,
            device=device,
            verbose=False
        )

        result = results[0]
        annotated = result.plot()

        # Counting Logic
        per_frame_counts = {}
        if ENABLE_COUNTING:
            for box in result.boxes:
                cls_id = int(box.cls.cpu().numpy())
                cls_name = model.names[cls_id]

                if COUNT_TARGET_CLASSES and cls_name not in COUNT_TARGET_CLASSES:
                    continue

                per_frame_counts[cls_name] = per_frame_counts.get(cls_name, 0) + 1
                total_counts[cls_name] = total_counts.get(cls_name, 0) + 1

                # ZONE ALERT
                if ENABLE_ZONE_ALERT:
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    if cls_name in ALERT_CLASSES and point_in_poly((cx, cy), ZONE_POLYGON):
                        cv2.putText(
                            annotated,
                            f"ALERT: {cls_name} in zone!",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 255), 2
                        )

        # Draw zone if enabled
        if ENABLE_ZONE_ALERT:
            draw_zone(annotated, ZONE_POLYGON)

        # Display per-frame counts
        draw_counts(annotated, per_frame_counts)

        # FPS calculation
        now = time.time()
        fps = 1 / (now - prev_time)
        prev_time = now
        draw_fps(annotated, fps)

        # Show Window
        if args.show:
            cv2.imshow("VisionTrack-YOLO (Accuracy Boosted)", annotated)

        # Save video
        if out_writer:
            out_writer.write(annotated)

        # Key Controls
        key = cv2.waitKey(1) & 0xFF
        if key == 27:   # ESC
            print("[INFO] ❌ Exit pressed.")
            break
        elif key == ord('s'):  # Save frame
            save_path = save_frame(annotated, OUTPUT_DIR)
            print(f"[INFO] 📸 Saved frame → {save_path}")

    # Cleanup
    cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------
# RUN
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
