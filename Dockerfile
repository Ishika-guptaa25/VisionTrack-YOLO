FROM python:3.10

WORKDIR /app

# Install system dependencies for OpenCV, YOLO, video, images
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    libglx-mesa0 \
    libgl1-mesa-dri \
    && apt-get clean

# Copy requirements
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . /app/

# Expose Streamlit port
EXPOSE 7860

# Run the app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
