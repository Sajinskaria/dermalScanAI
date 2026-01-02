# DermalScanAI 🧬

**Advanced AI-Powered Skin Condition Analysis**

DermalScanAI is an intelligent facial skin condition analyzer that leverages deep learning to detect and classify various skin conditions including clear skin, dark spots, puffy eyes, and wrinkles. Built with state-of-the-art EfficientNetB0 architecture and a modern Streamlit interface, it provides fast, accurate analysis with professional export capabilities.

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-MIT-orange)

---

## ✨ Features

### Core Capabilities
- **🎯 Face Detection**: Automatic face detection using Haar Cascade classifiers
- **🔬 Skin Condition Analysis**: AI-powered prediction across 4 conditions:
  - Clear Skin
  - Dark Spots
  - Puffy Eyes
  - Wrinkles
- **📊 Confidence Scoring**: Detailed probability distributions for all conditions
- **👥 Multi-Face Support**: Analyze multiple faces in a single image
- **⚡ Fast Processing**: Analysis completed in ≤5 seconds per image

### Export & Reporting
- **🖼️ Annotated Images**: Download PNG images with detection boxes and labels
- **📋 CSV Reports**: Detailed predictions with timestamps and probabilities
- **📦 Complete Package**: ZIP bundle with both image and CSV data
- **📝 Comprehensive Logging**: Automatic logging of all analyses

### User Interface
- **🎨 Modern Medical Theme**: Dark blue/teal aesthetic with DNA pattern background
- **🔄 Real-time Results**: Instant visual feedback with horizontal bar charts
- **📱 Responsive Design**: Works on various screen sizes
- **🎭 Glassmorphism UI**: Premium visual design with modern effects

---

## 🚀 Quick Start

### Prerequisites
- **Operating System**: Windows 10/11
- **Python**: Version 3.8 or higher
- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: ~500MB for dependencies and model

### Installation

1. **Clone or Download the Repository**
   ```bash
   cd c:\Users\LENOVO\OneDrive\Desktop\infosys
   ```

2. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

   Required packages:
   - `streamlit` - Web application framework
   - `pillow` - Image processing
   - `numpy` - Numerical operations
   - `opencv-python` - Computer vision operations
   - `keras` - Deep learning framework
   - `tensorflow` - ML backend

3. **Verify Model File**
   
   Ensure `best_dermal_model.h5` is present in the project directory.

### Running the Application

1. **Start the Streamlit Server**
   ```bash
   streamlit run dermal_app.py
   ```

2. **Access the Application**
   
   Your default browser will automatically open to:
   ```
   http://localhost:8501
   ```

3. **Alternative: Manual Browser Access**
   
   If the browser doesn't open automatically, copy the URL from the terminal.

---

## 📖 Usage Guide

### Step 1: Upload Image
1. Click the upload area or drag & drop an image file
2. Supported formats: JPG, JPEG, PNG
3. Best results with clear, frontal face photos

### Step 2: View Analysis
- **Main Image**: Shows detected faces with colored bounding boxes
- **Individual Results**: Expandable sections for each detected person
- **Confidence Badge**: Overall confidence score for the prediction
- **Probability Bars**: Detailed breakdown across all conditions

### Step 3: Export Results

Choose from three export options:

1. **📥 Annotated Image**
   - PNG format with detection boxes
   - Includes metadata overlay (timestamp, filename, face count)
   - Perfect for visual documentation

2. **📥 CSV Report**
   - Structured data with predictions
   - Includes all probability percentages
   - Ideal for data analysis and record keeping

3. **📥 Full Package**
   - ZIP file containing both image and CSV
   - Complete analysis bundle
   - Best for archival purposes

### Adjusting Settings (Sidebar)

- **Model Path**: Change the path to your trained model
- **Class Labels**: Customize condition names
- **Confidence Threshold**: Filter low-confidence predictions
- **Face Crop Padding**: Adjust detection box margins
- **Display Width**: Optimize image size for your screen

---

## 🎯 Milestones & Achievements

| Milestone | Focus Area | Metric | Status |
|-----------|------------|--------|--------|
| **M1** | Data Preparation | Balanced & clean dataset | ✅ Complete |
| **M2** | Model Performance | Good test accuracy | ✅ Complete |
| **M3** | UI & Backend | ≤5s per image processing | ✅ Complete |
| **M4** | Final Delivery | Complete documentation | ✅ Complete |

---

## 🛠️ Technology Stack

| Area | Technologies |
|------|--------------|
| **Image Processing** | OpenCV, NumPy, Haar Cascade |
| **Deep Learning** | TensorFlow/Keras, EfficientNetB0 |
| **Frontend** | Streamlit,  |
| **Backend** | Python, Modularized Inference |
| **Export** | CSV, Annotated PNG, ZIP |

---

## 🔧 Troubleshooting

### "streamlit: command not found"

**Solution**: Streamlit is not installed or not in PATH
```bash
pip install streamlit
# or
python -m pip install streamlit
```

### Model Loading Error

**Problem**: `FileNotFoundError: best_dermal_model.h5`

**Solution**:
1. Verify the model file exists in the project directory
2. Update the model path in the sidebar or code (line 334)
3. Use absolute path if needed

### TensorFlow Installation Issues on Windows

**Problem**: Long path errors during TensorFlow installation

**Solution**:
1. Run `enable_long_paths.py` as Administrator
2. Or manually enable long paths in Windows Registry
3. Restart terminal and reinstall

```bash
python enable_long_paths.py
pip install tensorflow
```

### No Faces Detected

**Problem**: "⚠️ No faces detected"

**Solution**:
- Use a clear, well-lit frontal face photo
- Ensure face is not too small in the image
- Avoid extreme angles or occlusions
- Adjust detection settings in sidebar (reduce minNeighbors)

### Slow Performance

**Problem**: Processing takes longer than 5 seconds

**Solution**:
- Reduce image size (use Max Display Width slider)
- Close other resource-intensive applications
- Ensure model is properly cached (first run is slower)
- Check system meets minimum requirements

### Import Errors

**Problem**: `ModuleNotFoundError`

**Solution**: Install all dependencies
```bash
pip install -r requirements.txt
```

---

## 📊 Understanding Results

### Confidence Scores
- **90-100%**: Very high confidence - reliable prediction
- **70-89%**: High confidence - good reliability
- **50-69%**: Moderate confidence - consider context
- **Below 50%**: Low confidence - manual review recommended

### Condition Descriptions

| Condition | Description |
|-----------|-------------|
| **Clear Skin** | Healthy skin without visible issues |
| **Dark Spots** | Hyperpigmentation or age spots |
| **Puffy Eyes** | Under-eye swelling or bags |
| **Wrinkles** | Fine lines or deep wrinkles |

---

## 📁 Project Structure

```
infosys/
├── dermal_app.py              # Main Streamlit application
├── best_dermal_model.h5       # Trained EfficientNet model
├── requirements.txt           # Python dependencies
├── enable_long_paths.py       # Windows path length fix
├── test_exports.py            # Export functionality tests
├── logs/                      # Application logs
├── README.md                  # User documentation (this file)
└── DEVELOPER_GUIDE.md         # Developer documentation
```

---

## 🔐 Privacy & Data

- **Local Processing**: All analysis happens on your local machine
- **No Data Upload**: Images are not sent to external servers
- **Logging**: Session logs stored locally in `logs/` directory
- **Export Control**: You decide what to save and where

---

## 🤝 Support & Contribution

### Reporting Issues
If you encounter bugs or have feature requests, please document:
1. Steps to reproduce the issue
2. Expected vs actual behavior
3. System information (OS, Python version)
4. Error messages or screenshots

### Feature Requests
We welcome suggestions for improvements! Consider:
- Additional skin conditions to detect
- UI/UX enhancements
- Export format options
- Performance optimizations

---

## 📝 License

This project is provided as-is for educational and research purposes.

---

## 🙏 Acknowledgments

- **EfficientNet**: Original architecture by Google Research
- **OpenCV**: Computer vision library
- **Streamlit**: Web application framework
- **TensorFlow/Keras**: Deep learning frameworks

---

## 📞 Getting Help

For technical questions and developer documentation, see [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)

---

**DermalScanAI** - Empowering skin health through AI technology 🧬✨
