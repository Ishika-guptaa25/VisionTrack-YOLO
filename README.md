---
title: VisionTrack-YOLO
emoji: 👁️
colorFrom: purple
colorTo: indigo
sdk: streamlit
sdk_version: "1.29.0"
app_file: streamlit_app.py
pinned: false
---

# 👁️ VisionTrack-YOLO: Real-Time Object Detection System

A powerful, real-time object detection system built with YOLOv8 and Python. Track, count, and monitor objects with advanced zone-based alerts, all wrapped in an intuitive interface.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.7+-red.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

## 🚀 Live Demo

**[Try the Streamlit Web App](https://huggingface.co/spaces/ishikagupta25/VisionTrack-YOLO)** 

---

## ✨ Features

✅ **Real-Time Detection** - Live webcam or video file processing  
✅ **Multi-Source Support** - Webcam, video files, or single images  
✅ **Object Counting** - Count detected objects by class  
✅ **Zone-Based Alerts** - Define restricted zones and get alerts  
✅ **FPS Display** - Real-time performance monitoring  
✅ **Save Outputs** - Export detected videos and snapshots  
✅ **Interactive Controls** - Pause, snapshot, and ESC to quit  
✅ **Web Interface** - Streamlit demo for easy testing  
✅ **GPU Acceleration** - CUDA support for faster processing  

---

## 🎬 Demo
<img width="500" height="333" alt="image" src="https://github.com/user-attachments/assets/664634c5-1e71-4d4f-9b78-bc711aff0ac0" />

### Real-Time Detection
<img width="500" height="333" alt="image" src="https://github.com/user-attachments/assets/13303653-6581-4d2e-aa61-4795ca7d96e7" />

### Object Detection
<img width="500" height="333" alt="image" src="https://github.com/user-attachments/assets/c2e9bd96-217e-44e4-8122-9701f10471ad" />

---

## ⚠️ **Important Note — COCO Dataset Limitation**  
#### *Please read this before testing the project*

<div align="center">
  <img src="https://img.shields.io/badge/Dataset-COCO%2080%20Classes-blueviolet?style=for-the-badge" />
</div>

---

## 🔍 What this means

This project uses **YOLOv8 pretrained on the COCO dataset**, which contains **ONLY 80 object classes**.

Therefore, the system can **only detect known COCO objects**.

---

## ✅ Detectable COCO Objects (Examples)

✔ person  
✔ car / bus / truck  
✔ dog / cat / horse  
✔ bottle / cup / bowl  
✔ fork / knife / spoon  
✔ laptop / tv / keyboard / mouse  
✔ apple / banana / orange  
✔ chair / couch / bed  
✔ microwave / oven / sink / refrigerator  

---

## ❌ NOT Detectable (Not part of COCO dataset)

The model **will NOT detect these objects correctly**:

- whisk  
- spatula  
- ladle  
- tongs  
- Indian kitchen utensils  
- toys  
- makeup products  
- stationery items  
- cartoon / clipart images  
- uncommon tools and objects  

> 🟥 These objects do *not* exist in COCO dataset → YOLO guesses incorrectly.

---

## 🛠️ Want to detect your own objects?

You must train a **custom YOLO model**.

> ✔ Custom training guide available  
> ✔ Works with your utensils, cosmetics, tools, toys  
> ✔ 10× better accuracy for non-COCO items  

---

## 🛠️ Technology Stack

| Category | Technology |
|----------|-----------|
| 🧠 **AI Model** | YOLOv8 (Ultralytics) |
| 👁️ **Computer Vision** | OpenCV 4.7+ |
| 💻 **Language** | Python 3.8+ |
| 🚀 **Acceleration** | CUDA / PyTorch |
| 🎨 **Web UI** | Streamlit |
| 📊 **Processing** | NumPy |

---

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- (Optional) NVIDIA GPU with CUDA for acceleration
- Webcam for live detection (optional)

### Quick Setup

```bash
# 1. Clone the repository
git clone https://github.com/Ishika-guptaa25/VisionTrack-YOLO.git
cd VisionTrack-YOLO

# 2. Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
python visiontrack.py --source 0 --show
```

That's it! The system will start with your webcam.

---

## 🎯 Usage

### Command Line Interface

#### Basic Usage

```bash
# Use webcam (source 0)
python visiontrack.py --source 0 --show

# Use video file
python visiontrack.py --source path/to/video.mp4 --show

# Save output video
python visiontrack.py --source 0 --save --show
```

#### Advanced Options

```bash
python visiontrack.py \
    --source 0 \              # 0 for webcam, or video/image path
    --save \                   # Save output video
    --show \                   # Display window
    --device cuda \            # Use GPU (cuda or cpu)
    --model yolov8n.pt \      # Model size (n/s/m/l/x)
    --conf 0.25                # Confidence threshold
```

### Interactive Controls

While running the application:

| Key | Action |
|-----|--------|
| **ESC** | Exit application |
| **S** | Save current frame snapshot |
| **P** | Pause/Resume detection |

---

## 🌐 Streamlit Web Interface

Launch the web-based demo:

```bash
streamlit run streamlit_app.py
```

### Features:
- 📷 **Image Mode**: Upload and detect objects in images
- 🎥 **Video Mode**: Upload video files for batch processing
- 🎛️ **Interactive**: Adjust confidence threshold on the fly
- 💾 **Download**: Save detection results

---

## ⚙️ Configuration

Edit `config.py` to customize behavior:

### Model Settings

```python
MODEL_NAME = "yolov8n.pt"      # Options: yolov8n/s/m/l/x.pt
CONFIDENCE_THRESHOLD = 0.25    # Detection confidence (0.0-1.0)
IOU_THRESHOLD = 0.45           # Intersection over Union threshold
DEVICE = "cuda"                # "cuda" for GPU, "cpu" for CPU
```

### Detection Settings

```python
# Enable/disable features
ENABLE_COUNTING = True
ENABLE_ZONE_ALERT = True

# Classes to count
COUNT_TARGET_CLASSES = ["person", "car", "bicycle"]

# Define restricted zone (polygon coordinates)
ZONE_POLYGON = [(50,50), (400,50), (400,300), (50,300)]

# Classes that trigger zone alerts
ALERT_CLASSES = ["person"]
```

### Output Settings

```python
OUTPUT_DIR = "outputs/detections"
SAVE_VIDEO = True
VIDEO_FPS = 20
```

---

## 🎨 YOLOv8 Model Options

| Model | Size | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| **YOLOv8n** | 3MB | ⚡ Fast | Good | Webcam, Real-time |
| **YOLOv8s** | 11MB | ⚡ Fast | Better | Balanced |
| **YOLOv8m** | 26MB | 🔄 Medium | Great | Accuracy priority |
| **YOLOv8l** | 44MB | 🐢 Slow | Excellent | High accuracy |
| **YOLOv8x** | 68MB | 🐢 Slower | Best | Maximum accuracy |

Change model in `config.py`:
```python
MODEL_NAME = "yolov8s.pt"  # or yolov8m.pt, yolov8l.pt, yolov8x.pt
```

---

## 📚 Project Structure

```
VisionTrack-YOLO/
│
├── visiontrack.py           # Main detection application
├── streamlit_app.py         # Web interface demo
├── config.py                # Configuration settings
├── utils.py                 # Utility functions
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── LICENSE                 # MIT License
└── screenshots             # Demo images
   
```

---

## 🔍 How It Works

### Detection Pipeline

```
Input Source (Webcam/Video/Image)
            ↓
    Frame Capture & Preprocessing
            ↓
    YOLOv8 Model Inference
            ↓
    Object Detection & Classification
            ↓
    ├── Bounding Box Drawing
    ├── Object Counting
    ├── Zone Alert Check
    └── FPS Calculation
            ↓
    Display & Save Output
```

### Zone Alert System

1. **Define Zone**: Set polygon coordinates in `config.py`
2. **Track Objects**: System monitors object centers
3. **Alert Trigger**: When specified class enters zone
4. **Visual Feedback**: Red alert overlay on frame

### Object Counting

- Counts objects per frame by class
- Cumulative counting across video
- Configurable target classes
- Real-time overlay display

---

## 🎓 Detected Object Classes

YOLOv8 is trained on COCO dataset with **80 classes**:

```
person, bicycle, car, motorcycle, airplane, bus, train, truck, boat,
traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat,
dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella,
handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite,
baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle,
wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange,
broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant,
bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone,
microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors,
teddy bear, hair drier, toothbrush
```

---

## 📊 Performance Benchmarks

### On NVIDIA GTX 1660 Ti

| Model | FPS (Webcam) | FPS (Video) | Detection Time |
|-------|--------------|-------------|----------------|
| YOLOv8n | ~60 FPS | ~70 FPS | ~16ms |
| YOLOv8s | ~45 FPS | ~50 FPS | ~22ms |
| YOLOv8m | ~30 FPS | ~35 FPS | ~33ms |

### On CPU (Intel i5)
| Model | FPS (Webcam) | FPS (Video) | Detection Time |
|-------|--------------|-------------|----------------|
| YOLOv8n | ~10 FPS | ~12 FPS | ~100ms |
| YOLOv8s | ~6 FPS | ~8 FPS | ~166ms |

*Your performance may vary based on hardware*

---

## 🚀 Deployment

### Streamlit Cloud (FREE)

1. **Push to GitHub**
```bash
git add .
git commit -m "Deploy VisionTrack-YOLO"
git push origin main
```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect GitHub repository: `Ishika-guptaa25/VisionTrack-YOLO`
   - Main file: `streamlit_app.py`
   - Click "Deploy"

3. **Live in 2-3 minutes!** 🎉

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "visiontrack.py", "--source", "0", "--show"]
```

Build and run:
```bash
docker build -t visiontrack-yolo .
docker run -it --rm --device=/dev/video0 visiontrack-yolo
```

---

## 🧪 Use Cases

### 🏢 Security & Surveillance
- Monitor restricted areas
- Count people entering/exiting
- Alert on unauthorized access

### 🚗 Traffic Monitoring
- Vehicle counting by type
- Speed estimation (with calibration)
- Parking lot occupancy

### 🏭 Industrial Safety
- PPE (Personal Protective Equipment) detection
- Worker safety zone monitoring
- Equipment tracking

### 🛒 Retail Analytics
- Customer counting
- Queue length monitoring
- Product interaction tracking

### 🐾 Wildlife Monitoring
- Animal species counting
- Migration pattern tracking
- Conservation efforts

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Camera Not Opening
```bash
# Check available cameras
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"

# Try different camera index
python visiontrack.py --source 1
```

#### 2. CUDA Not Available
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU usage
python visiontrack.py --device cpu
```

#### 3. Model Download Issues
```bash
# Manually download model
mkdir models
cd models
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

#### 4. Low FPS
- Use smaller model (yolov8n.pt)
- Reduce input resolution
- Enable GPU acceleration
- Close other applications

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. 🍴 Fork the repository
2. 🌿 Create feature branch
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. 💾 Commit changes
   ```bash
   git commit -m 'Add AmazingFeature'
   ```
4. 📤 Push to branch
   ```bash
   git push origin feature/AmazingFeature
   ```
5. 🔃 Open Pull Request

### Ideas for Contributions

- [ ] Add track ID persistence across frames
- [ ] Implement multi-zone alerts
- [ ] Add email/SMS notification system
- [ ] Create analytics dashboard
- [ ] Add pose estimation features
- [ ] Implement object tracking (DeepSORT)
- [ ] Add custom model training pipeline
- [ ] Create mobile app version

---

## 🎯 Future Enhancements

- [ ] **Object Tracking** - Persistent ID tracking with DeepSORT
- [ ] **Analytics Dashboard** - Historical data visualization
- [ ] **Multi-Camera Support** - Process multiple streams
- [ ] **Cloud Integration** - AWS/Azure deployment
- [ ] **REST API** - Programmatic access
- [ ] **Mobile App** - iOS/Android applications
- [ ] **Email Alerts** - Automated notifications
- [ ] **Database Logging** - Detection history storage
- [ ] **Custom Training** - Fine-tune on your dataset
- [ ] **Pose Estimation** - Human pose analysis

---

## 📖 Learning Resources

### YOLOv8 & Object Detection
- 📘 [Ultralytics YOLOv8 Docs](https://docs.ultralytics.com/)
- 📙 [YOLO Paper (Original)](https://arxiv.org/abs/1506.02640)
- 📕 [Computer Vision Course](https://www.coursera.org/learn/computer-vision-basics)

### OpenCV
- 📗 [OpenCV Documentation](https://docs.opencv.org/)
- 📓 [OpenCV Python Tutorials](https://docs.opencv.org/master/d6/d00/tutorial_py_root.html)

### Deep Learning
- 📔 [PyTorch Tutorials](https://pytorch.org/tutorials/)
- 📖 [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning)

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License - Free to use, modify, and distribute
```

---

## 👤 Author

**Ishika Gupta**

🎓 BCA Student | Python Developer | AI/ML Enthusiast  
📍 India  
💼 Building computer vision applications  

### Connect with me:

- 🐙 GitHub: [@Ishika-guptaa25](https://github.com/Ishika-guptaa25)

---

## 🙏 Acknowledgments

- **Ultralytics** - For the amazing YOLOv8 framework
- **OpenCV Team** - For computer vision tools
- **PyTorch** - For deep learning backend
- **Streamlit** - For easy web interface creation
- **COCO Dataset** - For pretrained model weights

---

## 📞 Support

### Found this useful?

⭐ **Star this repository** if it helped you!

### Need Help?

- 🐛 [Report Issues](https://github.com/Ishika-guptaa25/VisionTrack-YOLO/issues)
- 💡 [Request Features](https://github.com/Ishika-guptaa25/VisionTrack-YOLO/issues)
- 💬 [Discussions](https://github.com/Ishika-guptaa25/VisionTrack-YOLO/discussions)

---

## ⚠️ Disclaimer

This project is for **educational and research purposes**. When deploying in production:

- ✅ Ensure compliance with local privacy laws
- ✅ Obtain necessary permissions for surveillance
- ✅ Respect individual privacy rights
- ✅ Secure sensitive detection data
- ✅ Follow ethical AI guidelines

The authors are not responsible for misuse of this software.

---

## 🔗 Related Projects

- [YOLOv8 Official Repository](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://github.com/opencv/opencv)
- [Awesome Object Detection](https://github.com/amusi/awesome-object-detection)

---

## 📈 Project Stats

![GitHub stars](https://img.shields.io/github/stars/Ishika-guptaa25/VisionTrack-YOLO?style=social)
![GitHub forks](https://img.shields.io/github/forks/Ishika-guptaa25/VisionTrack-YOLO?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/Ishika-guptaa25/VisionTrack-YOLO?style=social)

---

<div align="center">

### Built with ❤️ using Python & YOLOv8

**If this project helped you, please give it a ⭐!**

[⬆ Back to Top](#️-visiontrack-yolo-real-time-object-detection-system)

</div>

---

**© 2025 Ishika Gupta. All rights reserved.**
