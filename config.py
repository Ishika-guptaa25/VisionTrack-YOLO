# ---------------------------------------------------------
# VisionTrack-YOLO — HIGH ACCURACY CONFIGURATION
# ---------------------------------------------------------

# 🔥 1. BEST ACCURACY MODEL
# Recommended models:
# yolov8s.pt → good balance
# yolov8m.pt → high accuracy
# yolov8l.pt → very high accuracy (recommended)
# yolov8x.pt → extreme accuracy (slow)
MODEL_NAME = "yolov8l.pt"   # <— BEST BOOST FOR ACCURACY

# 🔥 2. CONFIDENCE THRESHOLD
# Minimum confidence required to accept a detection
# Higher = more accurate detections
CONFIDENCE_THRESHOLD = 0.60   # recommended range: 0.55 – 0.75

# 🔥 3. IOU THRESHOLD (NMS FILTERING)
# Non-Maximum Suppression threshold
# Higher IOU = less overlapping / duplicate boxes
IOU_THRESHOLD = 0.55    # better accuracy & cleaner results

# 🔥 4. DEVICE CONFIG
# "cuda" → GPU (if available)
# "cpu" → CPU fallback
DEVICE = "cuda"    # auto-falls back to CPU if no GPU

# 🔥 5. LIMIT DETECTED CLASSES (optional)
# Set to None = detect all classes
# Example: detect only person & laptop → [0, 63]
LIMIT_CLASSES = None
# To improve accuracy for specific objects:
# LIMIT_CLASSES = [0, 63, 67]   # person, laptop, cellphone

# 🔥 6. OUTPUT SETTINGS
OUTPUT_DIR = "outputs/detections"
SAVE_VIDEO = True
VIDEO_FPS = 20

# 🔥 7. OBJECT COUNTING SETTINGS
ENABLE_COUNTING = True
COUNT_TARGET_CLASSES = ["person", "laptop", "cell phone", "keyboard"]
# Add/remove depending on your use-case

# 🔥 8. RESTRICTED ZONE ALERT (optional)
ENABLE_ZONE_ALERT = False   # turn ON when needed

# Zone polygon example (x,y) points
ZONE_POLYGON = [(50, 50), (450, 50), (450, 350), (50, 350)]

# Which classes should trigger alerts?
ALERT_CLASSES = ["person"]
