# DermalScan - Google Colab Deployment Guide

## 🚀 Quick Start (Recommended - No Installation Required!)

Since TensorFlow couldn't be installed locally due to Windows Long Path limitations, use Google Colab instead:

### Steps to Run in Google Colab:

1. **Open the Notebook**
   - Go to [Google Colab](https://colab.research.google.com/)
   - Click `File` → `Upload notebook`
   - Upload `DermalScan_Colab.ipynb` from this folder

2. **Upload Your Model**
   - Run the cells in order
   - When prompted, upload your `best_dermal_model.h5` file

3. **Access Your App**
   - The notebook will generate a public URL
   - Click the URL to open your Streamlit app in a new tab
   - Upload face images and get predictions!

## 📁 Files in This Directory

- `dermal_app.py` - Original Streamlit app (requires TensorFlow locally)
- `DermalScan_Colab.ipynb` - ✅ **USE THIS** - Google Colab notebook (TensorFlow pre-installed)
- `best_dermal_model.h5` - Your trained model
- `enable_long_paths.py` - Script to fix Windows Long Path issue (requires admin)

## 🔧 Alternative: Fix Local Installation

If you prefer to run locally:

1. **Enable Windows Long Paths (requires restart)**
   - Open Command Prompt as Administrator
   - Run: `python enable_long_paths.py`
   - Restart your computer

2. **Install TensorFlow**
   ```bash
   pip install tensorflow-cpu==2.16.2
   ```

3. **Run the app**
   ```bash
   python -m streamlit run dermal_app.py
   ```

## 🎯 Features

- Face detection using Haar Cascade
- Skin condition classification: clear skin, dark spots, puffy eyes, wrinkles
- Real-time predictions with confidence scores
- Downloadable annotated images

## ⚠️ Notes

- Google Colab free tier may have session limits
- Keep the Colab notebook running while using the app
- For production use, consider deploying to Streamlit Cloud or similar service
