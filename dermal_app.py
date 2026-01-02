import streamlit as st
from PIL import Image
import numpy as np
import cv2
import io
import os
import csv
import logging
from datetime import datetime
import zipfile
import keras
from keras.applications.efficientnet import preprocess_input

# -----------------------
# LOGGING CONFIGURATION
# -----------------------
# Create logs directory if it doesn't exist
LOGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Configure logging
log_filename = os.path.join(LOGS_DIR, f"dermalscan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("DermalScan application started")

# -----------------------
# CONFIG
# -----------------------
st.set_page_config(
    page_title="DermalScan - Face & Skin Condition Predictor", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "DermalScan - AI-Powered Skin Condition Analysis"
    }
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    /* Import modern font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Orbitron:wght@500;700;900&display=swap');
    
    /* Global styling */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container - Dark blue medical theme */
    .main {
        background: linear-gradient(135deg, #0a1628 0%, #0f2847 50%, #1a3a5c 100%);
        background-attachment: fixed;
    }
    
    /* DNA Background Pattern */
    .main::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="400" height="400"><defs><pattern id="dna" x="0" y="0" width="200" height="200" patternUnits="userSpaceOnUse"><circle cx="50" cy="50" r="3" fill="%2300d4ff" opacity="0.3"/><circle cx="150" cy="150" r="3" fill="%2300d4ff" opacity="0.3"/><line x1="50" y1="50" x2="150" y2="150" stroke="%2300d4ff" stroke-width="1" opacity="0.2"/></pattern></defs><rect width="400" height="400" fill="url(%23dna)"/></svg>');
        opacity: 0.4;
        z-index: 0;
        pointer-events: none;
    }
    
    /* Content wrapper with glassmorphism */
    .block-container {
        background: rgba(10, 35, 66, 0.85);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 2rem 3rem !important;
        box-shadow: 0 8px 32px 0 rgba(0, 212, 255, 0.2), 
                    inset 0 0 60px rgba(0, 212, 255, 0.05);
        border: 1px solid rgba(0, 212, 255, 0.2);
        margin-top: 2rem;
        margin-bottom: 2rem;
        position: relative;
        z-index: 1;
    }
    
    /* Header styling - DermalScanAI */
    h1 {
        font-family: 'Orbitron', sans-serif;
        background: linear-gradient(90deg, #ffffff 0%, #00d4ff 50%, #ffffff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 900;
        font-size: 3.5rem !important;
        margin-bottom: 0.3rem !important;
        text-align: center;
        letter-spacing: 2px;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from {
            filter: drop-shadow(0 0 5px rgba(0, 212, 255, 0.4));
        }
        to {
            filter: drop-shadow(0 0 15px rgba(0, 212, 255, 0.8));
        }
    }
    
    /* Subtitle styling */
    .subtitle {
        text-align: center;
        color: #00d4ff;
        font-size: 1.3rem;
        margin-bottom: 2rem;
        font-weight: 300;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a1628 0%, #0f2847 100%);
        border-right: 1px solid rgba(0, 212, 255, 0.3);
    }
    
    section[data-testid="stSidebar"] > div {
        background: rgba(10, 35, 66, 0.6);
        backdrop-filter: blur(10px);
    }
    
    section[data-testid="stSidebar"] * {
        color: #00d4ff !important;
    }
    
    section[data-testid="stSidebar"] .stTextInput input,
    section[data-testid="stSidebar"] .stTextArea textarea {
        background: rgba(0, 212, 255, 0.1) !important;
        border: 1px solid rgba(0, 212, 255, 0.3) !important;
        color: #ffffff !important;
    }
    
    /* File uploader styling */
    .stFileUploader {
        background: rgba(0, 212, 255, 0.05);
        border: 2px dashed #00d4ff;
        border-radius: 15px;
        padding: 3rem 2rem;
        transition: all 0.3s ease;
        text-align: center;
    }
    
    .stFileUploader:hover {
        background: rgba(0, 212, 255, 0.1);
        border-color: #00ffff;
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.3);
        transform: translateY(-2px);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #0088cc 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 212, 255, 0.6);
        background: linear-gradient(135deg, #00ffff 0%, #00aadd 100%);
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #0088cc 100%);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 1rem 3rem;
        font-weight: 700;
        transition: all 0.3s ease;
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.5);
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 1.1rem;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(0, 212, 255, 0.7);
        background: linear-gradient(135deg, #00ffff 0%, #00aadd 100%);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #00d4ff 0%, #00ffff 100%);
        border-radius: 10px;
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    
    /* Info/Warning boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid #00d4ff;
        background: rgba(0, 212, 255, 0.1);
        color: #00d4ff;
    }
    
    /* Image containers */
    .stImage {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 212, 255, 0.2);
        transition: all 0.3s ease;
        border: 2px solid rgba(0, 212, 255, 0.3);
    }
    
    .stImage:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 30px rgba(0, 212, 255, 0.4);
        border-color: #00d4ff;
    }
    
    /* Feature checklist card */
    .feature-card {
        background: rgba(0, 212, 255, 0.05);
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
        border: 1px solid rgba(0, 212, 255, 0.2);
    }
    
    .feature-item {
        display: flex;
        align-items: center;
        padding: 0.5rem 0;
        color: #00d4ff;
        font-size: 1rem;
    }
    
    .feature-item::before {
        content: '✓';
        display: inline-block;
        margin-right: 0.75rem;
        color: #00ff00;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    /* Result card styling */
    .result-card {
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.1) 0%, rgba(0, 136, 204, 0.1) 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(0, 212, 255, 0.3);
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateX(5px);
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.3);
    }
    
    /* Confidence badge */
    .confidence-badge {
        display: inline-block;
        background: linear-gradient(135deg, #00ff00 0%, #00cc00 100%);
        color: #000;
        padding: 0.5rem 1.5rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
        box-shadow: 0 4px 15px rgba(0, 255, 0, 0.4);
    }
    
    /* Subheader styling */
    h2, h3 {
        font-family: 'Orbitron', sans-serif;
        color: #00d4ff;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #00d4ff 0%, #00ffff 100%);
    }
    
    /* Section headers */
    .section-header {
        font-family: 'Orbitron', sans-serif;
        font-size: 1.8rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(0, 212, 255, 0.3);
    }
    
    /* Custom text colors */
    p, span, div {
        color: #b8d4e8;
    }
    
    /* Upload icon styling */
    .upload-icon {
        font-size: 4rem;
        color: #00d4ff;
        text-align: center;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Path to your trained model file (update if necessary)
MODEL_PATH = r"c:\Users\LENOVO\OneDrive\Desktop\infosys\best_dermal_model.h5"
# Default classes order - please set to match train_generator.class_indices order.
DEFAULT_CLASSES = ["clear skin", "dark spot", "puffy eyes", "wrinkles"]

# -----------------------
# CACHING / LOADERS
# -----------------------
@st.cache_resource(show_spinner=False)
def load_model(path):
    """Load Keras model once and cache it for the app lifetime."""
    model = keras.models.load_model(path)
    return model

@st.cache_resource(show_spinner=False)
def load_haar():
    """Load Haar cascade and cache."""
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    return cascade

# -----------------------
# HELPERS
# -----------------------
def pil_to_cv2(image_pil):
    """Convert PIL.Image to OpenCV BGR numpy array."""
    img = np.array(image_pil)
    # PIL uses RGB, convert to BGR
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def cv2_to_bytes(img_bgr):
    """Encode BGR image to PNG bytes for Streamlit display."""
    _, im_png = cv2.imencode(".png", img_bgr)
    return im_png.tobytes()

def detect_faces_haar(gray_img, face_cascade, scaleFactor=1.05, minNeighbors=3, minSize=(60,60)):
    """Return list of (x,y,w,h) faces detected by Haar cascade."""
    faces = face_cascade.detectMultiScale(
        gray_img,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=minSize,
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    return faces

def predict_on_face(face_bgr, model, classes_list):
    """Preprocess a face crop for EfficientNet, run model, and return (label, conf, raw_probs)."""
    # Resize to model input size
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, (224, 224))
    # EfficientNet preprocessing
    face_pre = preprocess_input(face_resized.astype("float32"))
    face_input = np.expand_dims(face_pre, axis=0)
    probs = model.predict(face_input)[0]
    idx = int(np.argmax(probs))
    label = classes_list[idx]
    conf = float(probs[idx]) * 100.0
    return label, conf, probs

def draw_annotations(img_bgr, boxes, results, box_color=(0,255,0), thickness=2):
    """Draw rectangles and multi-line labels under/above boxes. Returns annotated image."""
    annotated = img_bgr.copy()
    for (x, y, w, h), res in zip(boxes, results):
        label, conf, probs = res
        # Draw box
        cv2.rectangle(annotated, (x, y), (x+w, y+h), box_color, thickness)

        # Create multi-line label (top) and probabilities (bottom)
        top_text = f"{label} ({conf:.1f}%)"
        # determine where to put top text (avoid going off-image)
        txt_y = y - 10 if y - 10 > 10 else y + 20
        cv2.putText(annotated, top_text, (x, txt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2, cv2.LINE_AA)

    return annotated

def build_probs_overlay(img_bgr, boxes, results, classes_list):
    """Return a small image (numpy array) summarizing class probabilities per detected face to show in sidebar."""
    n = len(results)
    # create white canvas
    w, h = 360, max(100, 40 * n)
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255
    y = 20
    for i, ((x,yb,wb,hb), res) in enumerate(zip(boxes, results)):
        label, conf, probs = res
        header = f"Face {i+1}: {label} ({conf:.1f}%)"
        cv2.putText(canvas, header, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 1, cv2.LINE_AA)
        y += 22
        # per-class probabilities
        for c_idx, cname in enumerate(classes_list):
            p = probs[c_idx] * 100.0
            line = f"  {cname[:18]:18s}: {p:5.1f}%"
            cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,50,50), 1, cv2.LINE_AA)
            y += 18
        y += 6
    return canvas

# -----------------------
# EXPORT HELPERS
# -----------------------
def log_prediction(filename, face_idx, label, confidence, probs, classes_list):
    """Log prediction results to file."""
    try:
        logger.info(f"Prediction - File: {filename}, Face: {face_idx+1}, "
                   f"Condition: {label}, Confidence: {confidence:.2f}%, "
                   f"Probabilities: {', '.join([f'{c}={p*100:.1f}%' for c, p in zip(classes_list, probs)])}")
    except Exception as e:
        logger.error(f"Error logging prediction: {e}")

def create_csv_export(filename, boxes, results, classes_list):
    """Create CSV export with all predictions and probabilities."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write header
    header = ['Timestamp', 'Image_Filename', 'Face_Number', 'Predicted_Condition', 'Confidence_%']
    for class_name in classes_list:
        header.append(f'{class_name}_probability_%')
    writer.writerow(header)
    
    # Write data for each face
    for face_idx, (box, res) in enumerate(zip(boxes, results)):
        label, conf, probs = res
        row = [timestamp, filename, face_idx + 1, label, f'{conf:.2f}']
        for prob in probs:
            row.append(f'{prob * 100:.2f}')
        writer.writerow(row)
    
    # Get CSV bytes
    csv_bytes = output.getvalue().encode('utf-8')
    output.close()
    
    logger.info(f"CSV export created for {filename} with {len(results)} face(s)")
    return csv_bytes

def create_annotated_image_with_metadata(img_bgr, boxes, results, filename, add_metadata=True):
    """Create annotated image with optional metadata overlay."""
    annotated = draw_annotations(img_bgr, boxes, results, box_color=(0,255,0), thickness=2)
    
    if add_metadata:
        # Add timestamp and summary overlay
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        overlay_height = 60
        overlay = np.zeros((overlay_height, annotated.shape[1], 3), dtype=np.uint8)
        overlay[:] = (10, 36, 66)  # Dark blue background
        
        # Add text to overlay
        cv2.putText(overlay, f"DermalScan Analysis - {timestamp}", 
                   (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 212, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, f"File: {filename} | Faces Detected: {len(results)}", 
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (184, 212, 232), 1, cv2.LINE_AA)
        
        # Combine overlay with annotated image
        annotated = np.vstack([overlay, annotated])
    
    return annotated

def create_export_package(img_bgr, boxes, results, filename, classes_list):
    """Create a ZIP package containing both annotated image and CSV."""
    # Create ZIP in memory
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add annotated image
        annotated = create_annotated_image_with_metadata(img_bgr, boxes, results, filename, add_metadata=True)
        _, img_encoded = cv2.imencode('.png', cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        zip_file.writestr('annotated_image.png', img_encoded.tobytes())
        
        # Add CSV
        csv_data = create_csv_export(filename, boxes, results, classes_list)
        zip_file.writestr('predictions.csv', csv_data)
    
    zip_buffer.seek(0)
    logger.info(f"Export package created for {filename}")
    return zip_buffer.getvalue()

# -----------------------
# UI
# -----------------------
# Header - DermalScanAI Logo
st.markdown("""
    <div style='text-align: center; margin-bottom: 1rem;'>
        <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>🧬</div>
    </div>
""", unsafe_allow_html=True)

st.title("DermalScanAI")

st.markdown("""
    <div class='subtitle'>
        Face & Skin Condition Analyzer
    </div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Sidebar controls - Hidden but functional
st.sidebar.markdown("### ⚙️ Settings")
st.sidebar.markdown("---")
model_path_input = st.sidebar.text_input("Model path", value=MODEL_PATH)
classes_input = st.sidebar.text_area(
    "Class labels (one per line)",
    value="\n".join(DEFAULT_CLASSES),
    help="Enter the classes in the exact numeric order used while training (one per line)."
)
classes = [s.strip() for s in classes_input.splitlines() if s.strip()]

st.sidebar.markdown("---")
conf_thresh = st.sidebar.slider("Confidence threshold (%)", 0.0, 100.0, 1.0, 0.1)
padding = st.sidebar.slider("Face crop padding (px)", 0, 100, 25, 5)
resize_for_speed = st.sidebar.slider("Max display width (px)", 300, 1600, 900, 100)

# Load model and cascade (cached)
try:
    model = load_model(model_path_input)
    st.sidebar.success("✅ Model loaded")
except Exception as e:
    st.sidebar.error(f"❌ Model load failed: {e}")
    st.stop()

face_cascade = load_haar()

# Main layout - Side by side (Upload | Results)
col_upload, col_results = st.columns([1, 1])

with col_upload:
    st.markdown('<div class="section-header">Upload Your Photo</div>', unsafe_allow_html=True)
    
    # Upload area with icon
    st.markdown("""
        <div class='upload-icon'>📤</div>
        <div style='text-align: center; color: #00d4ff; margin-bottom: 1rem;'>
            <strong>Drag & Drop or Browse File</strong>
        </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose image", 
        type=["jpg","jpeg","png"], 
        accept_multiple_files=False,
        label_visibility="collapsed"
    )
    
    # Feature checklist
    st.markdown("""
        <div class='feature-card'>
            <div class='feature-item'>Face Detection</div>
            <div class='feature-item'>Skin Condition Analysis</div>
            <div class='feature-item'>Confidence Score</div>
            <div class='feature-item'>Results Preview</div>
        </div>
    """, unsafe_allow_html=True)

with col_results:
    st.markdown('<div class="section-header">Analysis Result</div>', unsafe_allow_html=True)

if uploaded_file is None:
    with col_results:
        st.info("Upload an image to see analysis results here.")
else:
    # Process the uploaded image
    with col_results:
        # Read image with PIL for reliability, then convert to OpenCV BGR
        image_pil = Image.open(uploaded_file).convert("RGB")
        img_bgr = pil_to_cv2(image_pil)

        # Resize for speed/display if too wide
        orig_h, orig_w = img_bgr.shape[:2]
        scale = 1.0
        if orig_w > resize_for_speed:
            scale = resize_for_speed / orig_w
            img_display = cv2.resize(img_bgr, (int(orig_w * scale), int(orig_h * scale)))
        else:
            img_display = img_bgr.copy()

        # Convert to gray and detect faces on the display-sized image for speed
        gray = cv2.cvtColor(img_display, cv2.COLOR_BGR2GRAY)

        faces = detect_faces_haar(gray, face_cascade, scaleFactor=1.05, minNeighbors=3, minSize=(40,40))

        if len(faces) == 0:
            st.warning("⚠️ No faces detected - Try a clearer frontal photo.")
            st.image(cv2.cvtColor(img_display, cv2.COLOR_BGR2RGB), use_column_width=True)
        else:
            # Processing status
            results = []
            boxes_display = []
            for (x, y, w, h) in faces:
                # convert box coords back to original image coordinates
                x0 = int(x / scale)
                y0 = int(y / scale)
                w0 = int(w / scale)
                h0 = int(h / scale)

                # Expand with padding in original image coordinates
                x1 = max(0, x0 - padding)
                y1 = max(0, y0 - padding)
                x2 = min(orig_w, x0 + w0 + padding)
                y2 = min(orig_h, y0 + h0 + padding)

                face_crop = img_bgr[y1:y2, x1:x2].copy()

                # Run prediction
                label, conf, probs = predict_on_face(face_crop, model, classes)
                results.append((label, conf, probs))
                # Add display-scaled box for drawing
                boxes_display.append((int(x), int(y), int(w), int(h)))
                
                # Log the prediction
                log_prediction(uploaded_file.name, len(results)-1, label, conf, probs, classes)

            # Filter boxes by confidence threshold
            filtered_results = []
            filtered_boxes = []
            for box, res in zip(boxes_display, results):
                if res[1] >= conf_thresh:
                    filtered_boxes.append(box)
                    filtered_results.append(res)

            # If none pass threshold, show all but warn
            if len(filtered_results) == 0:
                st.warning("⚠️ Low confidence detections")
                filtered_boxes = boxes_display
                filtered_results = results

            # Draw annotations on the resized display image
            annotated = draw_annotations(img_display, filtered_boxes, filtered_results, box_color=(0,255,0), thickness=2)

            # Display the main face image with detection box
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_column_width=True)
            
            # Show results for ALL detected faces
            if len(filtered_results) > 0:
                st.markdown("<div style='margin-top: 2rem;'></div>", unsafe_allow_html=True)
                st.markdown('<div class="section-header">Individual Analysis</div>', unsafe_allow_html=True)
                
                # Process each detected person
                for person_idx, (box, res) in enumerate(zip(filtered_boxes, filtered_results)):
                    lbl, conf, probs = res
                    
                    # Extract face crop for display
                    x_disp, y_disp, w_disp, h_disp = box
                    # Convert back to original image coordinates
                    x0 = int(x_disp / scale)
                    y0 = int(y_disp / scale)
                    w0 = int(w_disp / scale)
                    h0 = int(h_disp / scale)
                    x1 = max(0, x0 - padding)
                    y1 = max(0, y0 - padding)
                    x2 = min(orig_w, x0 + w0 + padding)
                    y2 = min(orig_h, y0 + h0 + padding)
                    face_crop_display = img_bgr[y1:y2, x1:x2].copy()
                    
                    # Create expandable section for each person
                    with st.expander(f"👤 Person {person_idx + 1}: {lbl.title()} ({conf:.0f}% Confidence)", expanded=True):
                        # Two columns: face image and analysis
                        face_col, analysis_col = st.columns([1, 2])
                        
                        with face_col:
                            # Display individual face crop
                            st.image(cv2.cvtColor(face_crop_display, cv2.COLOR_BGR2RGB), 
                                   caption=f"Person {person_idx + 1}", 
                                   use_column_width=True)
                            
                            # Confidence badge
                            st.markdown(f"""
                                <div style='text-align: center; margin-top: 1rem;'>
                                    <div class='confidence-badge'>{conf:.0f}% Confidence</div>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with analysis_col:
                            st.markdown("<div style='margin-top: 0.5rem;'></div>", unsafe_allow_html=True)
                            
                            # Horizontal bar charts for each condition
                            for c_idx, c_name in enumerate(classes):
                                p_val = probs[c_idx] * 100
                                
                                # Color based on value - highest gets green/lime color
                                if c_idx == np.argmax(probs):
                                    bar_color = "#7fff00"  # Lime green for prediction
                                elif p_val < 10:
                                    bar_color = "#4a5568"  # Gray for low values  
                                else:
                                    bar_color = "#00d4ff"  # Cyan for medium values
                                
                                st.markdown(f"""
                                    <div style='margin-bottom: 1rem;'>
                                        <div style='display: flex; justify-content: space-between; margin-bottom: 0.3rem;'>
                                            <span style='color: #ffffff; font-weight: 500;'>{c_name.title()}</span>
                                            <span style='color: #00d4ff; font-weight: 600;'>{p_val:.0f}%</span>
                                        </div>
                                        <div style='background: rgba(0, 212, 255, 0.1); border-radius: 10px; height: 12px; overflow: hidden;'>
                                            <div style='background: {bar_color}; width: {p_val}%; height: 100%; border-radius: 10px; 
                                                        box-shadow: 0 0 10px {bar_color};'></div>
                                        </div>
                                    </div>
                                """, unsafe_allow_html=True)

# Export Options - Enhanced download section
if uploaded_file is not None and len(faces) > 0:
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Export section header
    st.markdown("""
        <div style='text-align: center; margin-bottom: 1.5rem;'>
            <h2 style='color: #00d4ff; font-family: Orbitron, sans-serif; font-size: 2rem;'>
                📊 Export Analysis Results
            </h2>
            <p style='color: #b8d4e8; margin-top: 0.5rem;'>
                Download your analysis in multiple formats
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Three columns for different export options
    export_col1, export_col2, export_col3 = st.columns(3)
    
    with export_col1:
        # Annotated Image Export
        st.markdown("""
            <div style='text-align: center; margin-bottom: 1rem;'>
                <div style='font-size: 3rem; margin-bottom: 0.5rem;'>🖼️</div>
                <h3 style='color: #00d4ff; font-size: 1.2rem; margin-bottom: 0.5rem;'>Annotated Image</h3>
                <p style='color: #b8d4e8; font-size: 0.9rem;'>PNG with detections</p>
            </div>
        """, unsafe_allow_html=True)
        
        annotated_with_meta = create_annotated_image_with_metadata(
            img_display, filtered_boxes, filtered_results, uploaded_file.name, add_metadata=True
        )
        annotated_bytes = cv2_to_bytes(cv2.cvtColor(annotated_with_meta, cv2.COLOR_BGR2RGB))
        
        st.download_button(
            "📥 Download Image",
            annotated_bytes,
            file_name=f"dermalscan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
            mime="image/png",
            use_container_width=True,
            key="download_image"
        )
    
    with export_col2:
        # CSV Export
        st.markdown("""
            <div style='text-align: center; margin-bottom: 1rem;'>
                <div style='font-size: 3rem; margin-bottom: 0.5rem;'>📊</div>
                <h3 style='color: #00d4ff; font-size: 1.2rem; margin-bottom: 0.5rem;'>CSV Report</h3>
                <p style='color: #b8d4e8; font-size: 0.9rem;'>Detailed predictions</p>
            </div>
        """, unsafe_allow_html=True)
        
        csv_data = create_csv_export(uploaded_file.name, filtered_boxes, filtered_results, classes)
        
        st.download_button(
            "📥 Download CSV",
            csv_data,
            file_name=f"dermalscan_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
            key="download_csv"
        )
    
    with export_col3:
        # Full Package Export (ZIP)
        st.markdown("""
            <div style='text-align: center; margin-bottom: 1rem;'>
                <div style='font-size: 3rem; margin-bottom: 0.5rem;'>📦</div>
                <h3 style='color: #00d4ff; font-size: 1.2rem; margin-bottom: 0.5rem;'>Full Report</h3>
                <p style='color: #b8d4e8; font-size: 0.9rem;'>Image + CSV bundle</p>
            </div>
        """, unsafe_allow_html=True)
        
        zip_data = create_export_package(img_display, filtered_boxes, filtered_results, uploaded_file.name, classes)
        
        st.download_button(
            "📥 Download Package",
            zip_data,
            file_name=f"dermalscan_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
            mime="application/zip",
            use_container_width=True,
            key="download_zip"
        )
    
    # Export statistics
    st.markdown(f"""
        <div style='text-align: center; margin-top: 2rem; padding: 1rem; 
                    background: rgba(0, 212, 255, 0.05); border-radius: 10px;
                    border: 1px solid rgba(0, 212, 255, 0.2);'>
            <p style='color: #00d4ff; margin: 0;'>
                ✅ Analysis Complete | 
                📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
                👥 {len(filtered_results)} Face(s) Detected
            </p>
        </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
    <div style='text-align: center; margin-top: 3rem; padding: 1rem; 
                border-top: 1px solid rgba(0, 212, 255, 0.2);'>
        <p style='color: #b8d4e8; font-size: 0.9rem;'>
            DermalScanAI - Advanced Skin Condition Analysis | Powered by EfficientNet
        </p>
    </div>
""", unsafe_allow_html=True)

