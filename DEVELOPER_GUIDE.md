# DermalScanAI - Developer Guide

**Technical Documentation for Contributors and Developers**

This guide provides in-depth technical information about the DermalScanAI application architecture, codebase organization, and development workflows.

---

## 📋 Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Technology Stack](#technology-stack)
3. [Project Structure](#project-structure)
4. [Code Organization](#code-organization)
5. [Development Setup](#development-setup)
6. [Model Integration](#model-integration)
7. [Customization Guide](#customization-guide)
8. [Logging System](#logging-system)
9. [Testing & Validation](#testing--validation)
10. [Deployment](#deployment)
11. [Performance Optimization](#performance-optimization)
12. [Future Enhancements](#future-enhancements)

---

## 🏗️ Architecture Overview

### High-Level Design

```
┌─────────────────────────────────────────────────────────┐
│                   Streamlit UI Layer                     │
│  (User Interface, File Upload, Results Display)         │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│              Application Core Logic                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Image        │  │ Face         │  │ Prediction   │ │
│  │ Processing   │→ │ Detection    │→ │ Engine       │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│           Model & Data Processing Layer                  │
│  - EfficientNetB0 Model (Keras/TensorFlow)             │
│  - Haar Cascade Classifier (OpenCV)                     │
│  - Image Preprocessing Pipeline                         │
└─────────────────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│            Export & Logging Layer                        │
│  - CSV Generation  - Image Annotation                   │
│  - ZIP Packaging   - Session Logging                    │
└─────────────────────────────────────────────────────────┘
```

### Design Patterns

1. **Caching Strategy**: Uses Streamlit's `@st.cache_resource` to load model and Haar cascade once
2. **Modular Functions**: Separate concerns into focused helper functions
3. **Pipeline Architecture**: Sequential processing from upload → detection → prediction → export
4. **State Management**: Streamlit session state for UI reactivity

---

## 🔧 Technology Stack

### Core Technologies

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **Web Framework** | Streamlit | Latest | UI and web server |
| **Deep Learning** | TensorFlow/Keras | 2.x | Model training & inference |
| **Model Architecture** | EfficientNetB0 | - | CNN backbone |
| **Computer Vision** | OpenCV (cv2) | 4.x | Image processing, face detection |
| **Image Handling** | Pillow (PIL) | Latest | Image I/O operations |
| **Numerical Ops** | NumPy | Latest | Array operations |
| **Python** | Python | 3.8+ | Runtime environment |

### Dependencies Detail

```python
streamlit          # Web application framework
pillow            # Image loading and manipulation
numpy             # Numerical operations
opencv-python     # Computer vision algorithms
keras             # High-level neural network API
tensorflow        # ML backend and model execution
```

---

## 📁 Project Structure

```
infosys/
│
├── dermal_app.py              # Main application (843 lines)
│   ├── Configuration & Setup  # Lines 1-45
│   ├── Custom CSS Styling     # Lines 47-331
│   ├── Model Loading          # Lines 333-351
│   ├── Helper Functions       # Lines 353-428
│   ├── Export Functions       # Lines 430-511
│   ├── UI Components          # Lines 513-843
│   └── Processing Pipeline    # Lines 596-665
│
├── best_dermal_model.h5       # Trained EfficientNetB0 model (~43MB)
│
├── enable_long_paths.py       # Windows registry fix for TensorFlow
│
├── test_exports.py            # Export functionality validation
│
├── logs/                      # Application logs directory
│   └── dermalscan_*.log       # Timestamped log files
│
├── __pycache__/               # Python bytecode cache
│
├── requirements.txt           # Python dependencies
├── README.md                  # User documentation
└── DEVELOPER_GUIDE.md         # This file
```

### File Descriptions

#### `dermal_app.py` (Main Application)

**Primary Features**:
- Streamlit page configuration and custom CSS
- Model and Haar cascade loading with caching
- Image upload and preprocessing
- Face detection and cropping
- Skin condition prediction
- Results visualization
- Export functionality (PNG, CSV, ZIP)
- Comprehensive logging

#### `best_dermal_model.h5` (ML Model)

- **Architecture**: EfficientNetB0 (transfer learning)
- **Input Shape**: (224, 224, 3) - RGB images
- **Output**: 4 classes (softmax probabilities)
- **Classes**: ["clear skin", "dark spot", "puffy eyes", "wrinkles"]
- **Preprocessing**: EfficientNet-specific normalization

#### `enable_long_paths.py` (Utility)

Windows-specific utility to enable long file paths in the registry, required for TensorFlow installation.

#### `test_exports.py` (Testing)

Validation script for export functionality testing.

---

## 💻 Code Organization

### 1. Configuration & Setup (Lines 1-45)

```python
# Imports
import streamlit as st
from PIL import Image
import numpy as np
import cv2
import keras
from keras.applications.efficientnet import preprocess_input

# Logging configuration
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Streamlit page config
st.set_page_config(
    page_title="DermalScan - Face & Skin Condition Predictor", 
    layout="wide"
)

# Model and class configuration
MODEL_PATH = r"c:\Users\LENOVO\OneDrive\Desktop\infosys\best_dermal_model.h5"
DEFAULT_CLASSES = ["clear skin", "dark spot", "puffy eyes", "wrinkles"]
```

### 2. Model Loading (Lines 341-351)

**Caching Strategy**:
```python
@st.cache_resource(show_spinner=False)
def load_model(path):
    """Load Keras model once and cache it for the app lifetime."""
    model = keras.models.load_model(path)
    return model

@st.cache_resource(show_spinner=False)
def load_haar():
    """Load Haar cascade and cache."""
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    return cascade
```

**Benefits**:
- Model loaded only once per server session
- Subsequent predictions use cached model
- Reduces memory overhead and latency

### 3. Image Processing Pipeline

#### A. Image Conversion (Lines 356-366)

```python
def pil_to_cv2(image_pil):
    """Convert PIL.Image to OpenCV BGR numpy array."""
    img = np.array(image_pil)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def cv2_to_bytes(img_bgr):
    """Encode BGR image to PNG bytes for Streamlit display."""
    _, im_png = cv2.imencode(".png", img_bgr)
    return im_png.tobytes()
```

#### B. Face Detection (Lines 368-377)

```python
def detect_faces_haar(gray_img, face_cascade, scaleFactor=1.05, 
                      minNeighbors=3, minSize=(60,60)):
    """Return list of (x,y,w,h) faces detected by Haar cascade."""
    faces = face_cascade.detectMultiScale(
        gray_img,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minSize,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces
```

**Parameters**:
- `scaleFactor=1.05`: Image pyramid scale reduction (smaller = more detections, slower)
- `minNeighbors=3`: Minimum neighbors for valid detection (higher = fewer false positives)
- `minSize=(60,60)`: Minimum face size in pixels

#### C. Prediction Engine (Lines 379-391)

```python
def predict_on_face(face_bgr, model, classes_list):
    """Preprocess a face crop for EfficientNet, run model, 
       and return (label, conf, raw_probs)."""
    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    
    # Resize to model input size
    face_resized = cv2.resize(face_rgb, (224, 224))
    
    # EfficientNet-specific preprocessing
    face_pre = preprocess_input(face_resized.astype("float32"))
    face_input = np.expand_dims(face_pre, axis=0)
    
    # Model prediction
    probs = model.predict(face_input)[0]
    idx = int(np.argmax(probs))
    label = classes_list[idx]
    conf = float(probs[idx]) * 100.0
    
    return label, conf, probs
```

**Key Steps**:
1. Color space conversion (BGR → RGB)
2. Resize to 224×224 (EfficientNet input size)
3. Apply EfficientNet preprocessing (normalization)
4. Add batch dimension
5. Run inference
6. Extract prediction and confidence

### 4. Annotation & Visualization (Lines 393-428)

```python
def draw_annotations(img_bgr, boxes, results, box_color=(0,255,0), thickness=2):
    """Draw rectangles and multi-line labels under/above boxes."""
    annotated = img_bgr.copy()
    for (x, y, w, h), res in zip(boxes, results):
        label, conf, probs = res
        # Draw bounding box
        cv2.rectangle(annotated, (x, y), (x+w, y+h), box_color, thickness)
        
        # Add label text
        top_text = f"{label} ({conf:.1f}%)"
        txt_y = y - 10 if y - 10 > 10 else y + 20
        cv2.putText(annotated, top_text, (x, txt_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2, cv2.LINE_AA)
    
    return annotated
```

### 5. Export System (Lines 430-511)

#### A. Logging (Lines 433-440)

```python
def log_prediction(filename, face_idx, label, confidence, probs, classes_list):
    """Log prediction results to file."""
    logger.info(f"Prediction - File: {filename}, Face: {face_idx+1}, "
               f"Condition: {label}, Confidence: {confidence:.2f}%, "
               f"Probabilities: {', '.join([f'{c}={p*100:.1f}%' 
                                           for c, p in zip(classes_list, probs)])}")
```

#### B. CSV Export (Lines 442-469)

```python
def create_csv_export(filename, boxes, results, classes_list):
    """Create CSV export with all predictions and probabilities."""
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header: Timestamp, Image, Face#, Prediction, Confidence, Probabilities...
    header = ['Timestamp', 'Image_Filename', 'Face_Number', 
              'Predicted_Condition', 'Confidence_%']
    for class_name in classes_list:
        header.append(f'{class_name}_probability_%')
    writer.writerow(header)
    
    # Data rows
    for face_idx, (box, res) in enumerate(zip(boxes, results)):
        label, conf, probs = res
        row = [timestamp, filename, face_idx + 1, label, f'{conf:.2f}']
        for prob in probs:
            row.append(f'{prob * 100:.2f}')
        writer.writerow(row)
    
    return output.getvalue().encode('utf-8')
```

#### C. Image Export with Metadata (Lines 471-491)

```python
def create_annotated_image_with_metadata(img_bgr, boxes, results, 
                                         filename, add_metadata=True):
    """Create annotated image with optional metadata overlay."""
    annotated = draw_annotations(img_bgr, boxes, results)
    
    if add_metadata:
        # Create overlay banner
        overlay = np.zeros((60, annotated.shape[1], 3), dtype=np.uint8)
        overlay[:] = (10, 36, 66)  # Dark blue
        
        # Add timestamp and info
        cv2.putText(overlay, f"DermalScan Analysis - {timestamp}", ...)
        cv2.putText(overlay, f"File: {filename} | Faces: {len(results)}", ...)
        
        # Combine overlay with image
        annotated = np.vstack([overlay, annotated])
    
    return annotated
```

#### D. ZIP Package (Lines 493-510)

```python
def create_export_package(img_bgr, boxes, results, filename, classes_list):
    """Create a ZIP package containing both annotated image and CSV."""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add annotated image
        annotated = create_annotated_image_with_metadata(...)
        _, img_encoded = cv2.imencode('.png', annotated)
        zip_file.writestr('annotated_image.png', img_encoded.tobytes())
        
        # Add CSV
        csv_data = create_csv_export(...)
        zip_file.writestr('predictions.csv', csv_data)
    
    return zip_buffer.getvalue()
```

### 6. UI Components (Lines 513-843)

**Structure**:
1. Header with logo and title
2. Two-column layout (Upload | Results)
3. Upload area with file uploader and feature checklist
4. Results display with annotated image
5. Individual face analysis with expandable sections
6. Export options (3-column layout)
7. Footer

---

## 🛠️ Development Setup

### Environment Setup

1. **Create Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Installation**
   ```bash
   python -c "import streamlit, cv2, tensorflow; print('All imports successful')"
   ```

### Development Workflow

1. **Make Code Changes**
   - Edit `dermal_app.py`
   - Use consistent formatting (PEP 8)

2. **Test Locally**
   ```bash
   streamlit run dermal_app.py
   ```
   - Streamlit auto-reloads on file save
   - Test with various images

3. **Validate Exports**
   ```bash
   python test_exports.py
   ```

4. **Check Logs**
   - View `logs/dermalscan_*.log` for errors
   - Verify predictions are logged correctly

---

## 🧠 Model Integration

### Current Model

- **File**: `best_dermal_model.h5`
- **Architecture**: EfficientNetB0 (pre-trained on ImageNet)
- **Fine-tuned For**: 4-class skin condition classification
- **Input**: 224×224×3 RGB images
- **Output**: 4 softmax probabilities

### Replacing or Updating the Model

#### Step 1: Train Your Model

Ensure your model:
- Accepts (224, 224, 3) input shape
- Outputs 4 classes (or update `DEFAULT_CLASSES`)
- Uses `.h5` format (Keras model)
- Compatible with `keras.models.load_model()`

#### Step 2: Save Model

```python
# During training
model.save('my_new_model.h5')
```

#### Step 3: Update Application

**Option A: Update Sidebar**
- Run the app
- Change "Model path" in sidebar to new model path

**Option B: Update Code**
```python
# Line 334 in dermal_app.py
MODEL_PATH = r"c:\path\to\my_new_model.h5"
```

#### Step 4: Update Classes (if needed)

```python
# Line 336 in dermal_app.py
DEFAULT_CLASSES = ["class1", "class2", "class3", "class4"]
```

**Important**: Class order must match training `class_indices`.

### Model Training Tips

1. **Data Preparation**
   - Balanced dataset across all classes
   - Face images cropped and aligned
   - Augmentation: rotation, flip, brightness, zoom

2. **Transfer Learning**
   ```python
   from keras.applications import EfficientNetB0
   
   base_model = EfficientNetB0(weights='imagenet', include_top=False, 
                               input_shape=(224, 224, 3))
   x = base_model.output
   x = GlobalAveragePooling2D()(x)
   x = Dense(256, activation='relu')(x)
   x = Dropout(0.5)(x)
   predictions = Dense(4, activation='softmax')(x)
   
   model = Model(inputs=base_model.input, outputs=predictions)
   ```

3. **Compilation**
   ```python
   model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
   ```

4. **Training**
   - Use early stopping
   - Save best model based on validation accuracy
   - Monitor for overfitting

---

## 🎨 Customization Guide

### Adding New Skin Conditions

1. **Retrain Model** with new classes
2. **Update Classes List**:
   ```python
   DEFAULT_CLASSES = ["clear skin", "dark spot", "puffy eyes", 
                      "wrinkles", "acne", "redness"]
   ```
3. **Update UI** (optional): Modify condition descriptions in README

### Modifying UI Theme

**Colors** (Lines 48-331):
```python
# Find and replace color codes
"#00d4ff"  # Primary cyan - change to your brand color
"#0a1628"  # Dark blue background
"#00ff00"  # Success green
```

**Fonts**:
```python
# Line 51
@import url('https://fonts.googleapis.com/css2?family=YourFont:wght@300;400;600;700&display=swap');

# Line 55
font-family: 'YourFont', sans-serif;
```

### Adjusting Detection Parameters

**Face Detection Sensitivity** (Line 614):
```python
faces = detect_faces_haar(gray, face_cascade, 
                         scaleFactor=1.05,    # Lower = more sensitive
                         minNeighbors=3,      # Lower = more detections
                         minSize=(40,40))     # Smaller = detect smaller faces
```

**Confidence Threshold** (Line 544):
```python
conf_thresh = st.sidebar.slider("Confidence threshold (%)", 0.0, 100.0, 1.0, 0.1)
# Default: 1.0% (very permissive)
# Recommended: 50-70% for production
```

### Adding Export Formats

**Example: Add JSON Export**

```python
# Add after line 469
def create_json_export(filename, boxes, results, classes_list):
    """Create JSON export with predictions."""
    import json
    data = {
        "timestamp": datetime.now().isoformat(),
        "filename": filename,
        "faces": []
    }
    
    for face_idx, (box, res) in enumerate(zip(boxes, results)):
        label, conf, probs = res
        face_data = {
            "face_number": face_idx + 1,
            "prediction": label,
            "confidence": round(conf, 2),
            "probabilities": {
                class_name: round(prob * 100, 2) 
                for class_name, prob in zip(classes_list, probs)
            },
            "bounding_box": {"x": box[0], "y": box[1], 
                           "width": box[2], "height": box[3]}
        }
        data["faces"].append(face_data)
    
    return json.dumps(data, indent=2).encode('utf-8')

# Add download button in UI section (after line 797)
json_data = create_json_export(uploaded_file.name, filtered_boxes, 
                               filtered_results, classes)
st.download_button("📥 Download JSON", json_data, 
                  file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                  mime="application/json")
```

---

## 📝 Logging System

### Log Configuration (Lines 17-32)

```python
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

log_filename = os.path.join(LOGS_DIR, 
                           f"dermalscan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()  # Also print to console
    ]
)
```

### Log Structure

**Format**: `[YYYY-MM-DD HH:MM:SS] [LEVEL] - Message`

**Logged Events**:
1. Application start
2. Each prediction (file, face, condition, confidence, probabilities)
3. CSV export creation
4. ZIP package creation
5. Errors (model loading, processing failures)

**Example Log Entry**:
```
[2025-12-31 09:30:15] [INFO] - DermalScan application started
[2025-12-31 09:30:45] [INFO] - Prediction - File: test.jpg, Face: 1, Condition: clear skin, Confidence: 95.34%, Probabilities: clear skin=95.3%, dark spot=2.1%, puffy eyes=1.5%, wrinkles=1.1%
[2025-12-31 09:30:47] [INFO] - CSV export created for test.jpg with 1 face(s)
[2025-12-31 09:30:48] [INFO] - Export package created for test.jpg
```

### Viewing Logs

```bash
# View latest log
type logs\dermalscan_*.log | tail -50

# Search for errors
findstr /i "error" logs\*.log

# Filter by specific file
findstr "test.jpg" logs\*.log
```

---

## 🧪 Testing & Validation

### Manual Testing Checklist

#### Functional Tests

- [ ] **Single Face Detection**
  - Upload image with one clear face
  - Verify detection box appears
  - Check prediction accuracy
  - Validate confidence score

- [ ] **Multi-Face Detection**
  - Upload image with multiple faces
  - Verify all faces detected
  - Check individual results for each person
  - Confirm separate analysis sections

- [ ] **Edge Cases**
  - No face in image → should show warning
  - Side profile → may not detect
  - Very small face → adjust minSize
  - Poor lighting → test robustness

- [ ] **Export Functionality**
  - Download annotated PNG → verify image quality
  - Download CSV → open in Excel, check data
  - Download ZIP → extract, verify contents

#### UI Tests

- [ ] **Responsiveness**
  - Resize browser window
  - Check mobile view (optional)

- [ ] **Sidebar Controls**
  - Adjust confidence threshold → results update
  - Change padding → box size changes
  - Modify display width → image resizes

- [ ] **Visual Elements**
  - CSS loads correctly
  - Colors match theme
  - Animations work smoothly

### Automated Testing

**`test_exports.py`** - Validates export functions

```bash
python test_exports.py
```

**What it tests**:
- CSV generation
- Image annotation
- ZIP package creation
- Metadata overlay

### Performance Testing

**Target**: ≤5 seconds per image

**Measure Processing Time**:

```python
# Add timing code in dermal_app.py (around line 620)
import time
start_time = time.time()

# ... processing code ...

end_time = time.time()
st.info(f"Processing time: {end_time - start_time:.2f} seconds")
```

**Optimization Tips** (if slow):
1. Reduce image size before processing
2. Lower detection parameters
3. Use GPU for TensorFlow (if available)
4. Ensure model caching is working

---

## 🚀 Deployment

### Local Deployment

**Standard Run**:
```bash
streamlit run dermal_app.py
```

**Custom Port**:
```bash
streamlit run dermal_app.py --server.port 8080
```

**Custom Host** (access from network):
```bash
streamlit run dermal_app.py --server.address 0.0.0.0
```

### Production Deployment Options

#### Option 1: Streamlit Community Cloud

1. Push code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repository
4. Deploy

**Limitations**:
- Model file size (GitHub 100MB limit)
- Resource constraints

#### Option 2: Docker Container

**Dockerfile**:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY dermal_app.py best_dermal_model.h5 ./

# Expose port
EXPOSE 8501

# Run application
CMD ["streamlit", "run", "dermal_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Build & Run**:
```bash
docker build -t dermalscan .
docker run -p 8501:8501 dermalscan
```

#### Option 3: Cloud VM (AWS, Azure, GCP)

1. Provision VM with Python 3.8+
2. Clone repository
3. Install dependencies
4. Run with systemd service for auto-restart

**Systemd Service** (`/etc/systemd/system/dermalscan.service`):
```ini
[Unit]
Description=DermalScan Application
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/infosys
ExecStart=/usr/bin/streamlit run dermal_app.py
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## ⚡ Performance Optimization

### Current Performance Profile

| Operation | Time | Bottleneck |
|-----------|------|------------|
| Model Loading | 2-5s | First run only (cached) |
| Face Detection | 0.1-0.5s | Image size, detection params |
| Prediction | 0.2-0.8s | Model inference |
| UI Rendering | 0.1-0.3s | Streamlit overhead |
| **Total** | **≤5s** | ✅ Target met |

### Optimization Strategies

#### 1. Image Resizing

**Current** (Line 603-609):
```python
if orig_w > resize_for_speed:
    scale = resize_for_speed / orig_w
    img_display = cv2.resize(img_bgr, (int(orig_w * scale), int(orig_h * scale)))
```

**Best Practice**: Keep default at 900px for good balance.

#### 2. Model Optimization

**Quantization** (reduce model size):
```python
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
```

**Benefits**: Smaller file, faster inference (requires code changes)

#### 3. Batch Processing

For multiple faces, predictions already batched per face. Could optimize to predict all faces in one batch:

```python
# Instead of loop: for each face → predict
# Batch all faces
face_inputs = [preprocess_face(face) for face in face_crops]
batch_input = np.array(face_inputs)
all_probs = model.predict(batch_input)  # Single call
```

#### 4. GPU Acceleration

**Check GPU Availability**:
```python
import tensorflow as tf
print("GPUs Available:", tf.config.list_physical_devices('GPU'))
```

**Install GPU TensorFlow**:
```bash
pip install tensorflow-gpu
# Requires CUDA and cuDNN
```

---

## 🔮 Future Enhancements

### Planned Features

1. **Age Estimation**
   - Add age prediction branch to model
   - Display estimated age in results

2. **Severity Scoring**
   - Beyond classification, rate severity (mild/moderate/severe)
   - Visual severity scale

3. **Historical Tracking**
   - Store user analysis history
   - Track skin condition changes over time
   - Progress charts

4. **Batch Processing**
   - Upload multiple images
   - Process directory of images
   - Bulk export

5. **Advanced Visualizations**
   - Heatmap overlays showing problem areas
   - Side-by-side before/after comparisons
   - Interactive charts (Plotly)

6. **Authentication & User Accounts**
   - User login system
   - Personal dashboards
   - Secure data storage

7. **API Mode**
   - REST API endpoint for programmatic access
   - Mobile app integration
   - Third-party integrations

8. **Additional Export Formats**
   - PDF reports with charts and recommendations
   - JSON API responses
   - Excel workbooks

9. **Model Improvements**
   - More skin conditions (acne, rosacea, eczema)
   - Skin tone analysis
   - Texture analysis

10. **Recommendations Engine**
    - Based on detected conditions, suggest treatments
    - Product recommendations
    - Lifestyle tips

### Technical Debt

- [ ] Add unit tests (pytest)
- [ ] Add integration tests
- [ ] Refactor large functions into smaller ones
- [ ] Add type hints (Python typing)
- [ ] Improve error handling and user feedback
- [ ] Add configuration file (YAML/JSON)
- [ ] Implement proper MVC architecture
- [ ] Add database for persistence (SQLite/PostgreSQL)

---

## 🤝 Contributing Guidelines

### Code Style

- Follow PEP 8
- Use meaningful variable names
- Add docstrings to functions
- Comment complex logic
- Keep functions under 50 lines when possible

### Pull Request Process

1. Fork repository
2. Create feature branch
3. Make changes with tests
4. Update documentation
5. Submit PR with clear description

### Reporting Bugs

Include:
- Steps to reproduce
- Expected vs actual behavior
- Environment (OS, Python version)
- Error messages and logs
- Screenshots if UI-related

---

## 📚 Additional Resources

### TensorFlow/Keras
- [Keras Documentation](https://keras.io/api/)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)

### OpenCV
- [OpenCV Documentation](https://docs.opencv.org/)
- [Face Detection Tutorial](https://docs.opencv.org/4.x/d2/d99/tutorial_js_face_detection.html)

### Streamlit
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Gallery](https://streamlit.io/gallery)
- [Caching Guide](https://docs.streamlit.io/library/advanced-features/caching)

---

## 📞 Contact & Support

For technical questions, feature requests, or contributions:

- **Documentation**: This guide and README.md
- **Code Issues**: Check logs in `logs/` directory
- **Model Questions**: Review Model Integration section

---

**Happy Coding! 🚀**

*DermalScanAI Developer Team*
