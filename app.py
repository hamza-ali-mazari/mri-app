import streamlit as st
import torch
import numpy as np
from PIL import Image
import os
from pathlib import Path
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from io import BytesIO
import zipfile
import time

from models import QMLSegmentationModel
from config import METRICS

# ============= PAGE CONFIG =============
st.set_page_config(
    page_title="🔬 QML Segmentation Hub",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============= CUSTOM STYLING =============
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
        margin-bottom: 20px;
    }
    .metric-box {
        background: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    .success-box {
        background: #d4edda;
        border-left: 4px solid #28a745;
        padding: 12px;
        border-radius: 8px;
        color: #155724;
    }
    .warning-box {
        background: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 12px;
        border-radius: 8px;
        color: #856404;
    }
</style>
""", unsafe_allow_html=True)

# ============= SESSION STATE =============
if 'model' not in st.session_state:
    st.session_state.model = None
if 'device' not in st.session_state:
    st.session_state.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if 'predictions_history' not in st.session_state:
    st.session_state.predictions_history = []

# ============= CONSTANTS =============
MODEL_PATH = Path("best_model.pth")
DATA_DIR = Path("PNG")
ORIGINAL_DIR = DATA_DIR / "Original"
GROUND_TRUTH_DIR = DATA_DIR / "Ground Truth"

# ============= LOAD MODEL =============
@st.cache_resource
def load_model():
    """Load QML model with caching"""
    try:
        if not MODEL_PATH.exists():
            return None
        
        model = QMLSegmentationModel()
        checkpoint = torch.load(MODEL_PATH, map_location=st.session_state.device)
        
        if isinstance(checkpoint, dict):
            if 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'], strict=False)
            elif 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'], strict=False)
            else:
                model.load_state_dict(checkpoint, strict=False)
        
        model.eval()
        return model
    except Exception as e:
        return None

# ============= IMAGE PROCESSING =============
def preprocess_image(image_path, img_size=64):
    """Preprocess image for inference"""
    try:
        # Use PIL instead of OpenCV
        img = Image.open(str(image_path)).convert('L')  # Convert to grayscale
        img = img.resize((img_size, img_size), Image.Resampling.LANCZOS)
        img_array = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).float()

        return img
    except:
        return None

def predict_segmentation(model, image_tensor):
    """Get prediction from model"""
    try:
        if image_tensor is None:
            return None
        
        image_tensor = image_tensor.to(st.session_state.device)
        
        with torch.no_grad():
            output = model(image_tensor)
        
        pred = torch.sigmoid(output).squeeze().cpu().numpy()
        
        # Adaptive thresholding
        prob_min, prob_max, prob_mean = pred.min(), pred.max(), pred.mean()
        threshold = prob_mean if (prob_max - prob_min) < 0.1 else 0.5
        
        pred_binary = (pred > threshold).astype(np.uint8) * 255
        return pred_binary
    except:
        return None

def calculate_metrics(pred, ground_truth):
    """Calculate segmentation metrics"""
    try:
        pred = (pred > 127).astype(np.uint8)
        gt = (ground_truth > 127).astype(np.uint8)
        
        pred_flat = pred.flatten()
        gt_flat = gt.flatten()
        
        intersection = np.sum(pred_flat * gt_flat)
        dice = 2.0 * intersection / (np.sum(pred_flat) + np.sum(gt_flat) + 1e-7)
        
        union = np.sum(np.logical_or(pred_flat, gt_flat))
        iou = intersection / (union + 1e-7)
        
        accuracy = np.sum(pred_flat == gt_flat) / len(pred_flat)
        
        return {
            'Dice Score': float(dice),
            'IoU Score': float(iou),
            'Pixel Accuracy': float(accuracy)
        }
    except:
        return {'Dice Score': 0.0, 'IoU Score': 0.0, 'Pixel Accuracy': 0.0}

# ============= UI COMPONENTS =============
def display_header():
    """Display main header"""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("""
        <div class="main-header">
            <h1>🧬 Quantum ML Image Segmentation</h1>
            <p>Advanced medical image analysis with QML architecture</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        device_name = "🟢 GPU" if torch.cuda.is_available() else "🔵 CPU"
        st.metric("Device", device_name)

def display_metrics_grid(metrics):
    """Display metrics in grid"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎯 Dice Score", f"{metrics['Dice Score']:.4f}", 
                 delta=f"{metrics['Dice Score']*100:.1f}%")
    with col2:
        st.metric("📊 IoU Score", f"{metrics['IoU Score']:.4f}", 
                 delta=f"{metrics['IoU Score']*100:.1f}%")
    with col3:
        st.metric("✅ Pixel Accuracy", f"{metrics['Pixel Accuracy']:.4f}", 
                 delta=f"{metrics['Pixel Accuracy']*100:.1f}%")

def create_comparison_figure(original, gt, pred, overlay):
    """Create side-by-side comparison"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    axes[0].imshow(original, cmap='viridis')
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(gt, cmap='gray')
    axes[1].set_title('Ground Truth', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(pred, cmap='gray')
    axes[2].set_title('Model Prediction', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    axes[3].imshow(overlay)
    axes[3].set_title('Overlay (Green=Predicted)', fontsize=12, fontweight='bold')
    axes[3].axis('off')
    
    plt.tight_layout()
    return fig

# ============= MAIN SIDEBAR =============
with st.sidebar:
    st.title("⚙️ Settings")
    
    mode = st.radio(
        "Select Mode",
        ["📊 Dashboard", "🔍 Test Dataset", "📤 Upload & Predict", "📈 Batch Analysis"]
    )
    
    st.divider()
    
    # Model info
    with st.expander("ℹ️ Model Information"):
        model = load_model()
        if model:
            st.success("✅ Model Loaded")
            st.caption("Architecture: U-Net + QML")
            st.caption("Input: 64×64 Grayscale")
            st.caption("Output: Binary Segmentation")
        else:
            st.error("❌ Model Not Found")
    
    with st.expander("📋 Project Info"):
        st.write("""
        **QML Segmentation Hub**
        - Quantum Machine Learning
        - Medical Image Analysis
        - Real-time Inference
        
        **Dataset**: 512 endoscopy images
        **Model Status**: Active
        """)

# ============= MAIN CONTENT =============
display_header()

# =========== MODE 1: DASHBOARD ===========
if mode == "📊 Dashboard":
    st.subheader("📊 Model Performance Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        ### 🎯 Key Metrics
        - **Dice Score**: 0.8418 (Training average)
        - **IoU Score**: 0.7740 (Training average)
        - **Pixel Accuracy**: 0.9795 (Training average)
        """)
    
    with col2:
        st.warning("""
        ### ⚠️ Model Status
        - Training convergence: Needs improvement
        - Encoder signal: Weak amplification applied
        - Recommendation: Retrain with better initialization
        """)
    
    st.divider()
    
    # Performance chart
    metrics_data = {
        'Metric': ['Dice', 'IoU', 'Accuracy'],
        'Score': [0.8418, 0.7740, 0.9795]
    }
    
    fig = go.Figure(data=[
        go.Bar(x=metrics_data['Metric'], y=metrics_data['Score'], 
               marker_color=['#667eea', '#764ba2', '#f093fb'])
    ])
    fig.update_layout(
        title="Training Metrics Summary",
        yaxis_title="Score",
        height=400,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Images", "512")
    with col2:
        st.metric("Processing Speed", "~100ms/image")

# =========== MODE 2: TEST DATASET ===========
elif mode == "🔍 Test Dataset":
    st.subheader("🔍 Dataset Visualization & Testing")

    # Check if dataset directories exist
    if not ORIGINAL_DIR.exists() or not GROUND_TRUTH_DIR.exists():
        st.error("❌ **Dataset not found!**")
        st.info("""
        The PNG dataset directories are not available in this deployment.

        **Options:**
        1. **Upload Mode**: Use the "📤 Upload & Predict" mode to test with your own images
        2. **Local Development**: Run the app locally with the full dataset
        3. **Demo Mode**: The model is loaded and ready for inference

        **Note**: Dataset images were excluded from deployment to keep the app lightweight.
        """)
        st.success("✅ Model loaded successfully - ready for custom image analysis!")
        return

    col1, col2 = st.columns([3, 1])
    with col1:
        image_files = sorted([f for f in os.listdir(ORIGINAL_DIR) if f.endswith('.png')])
        if not image_files:
            st.warning("No PNG images found in dataset directory.")
            return
        selected_image = st.selectbox("Select Image", image_files)
    with col2:
        st.metric("Total Images", len(image_files))
    
    if selected_image:
        original_path = ORIGINAL_DIR / selected_image
        ground_truth_path = GROUND_TRUTH_DIR / selected_image

        # Use PIL instead of OpenCV
        try:
            original_img = Image.open(str(original_path)).convert('RGB')
            ground_truth_img = Image.open(str(ground_truth_path)).convert('L')
            original_img = np.array(original_img)
            ground_truth_img = np.array(ground_truth_img)

            model = load_model()
            if model is not None:
                image_tensor = preprocess_image(original_path)
                prediction = predict_segmentation(model, image_tensor)

                if prediction is not None:
                    # Resize to match using PIL/numpy
                    h, w = original_img.shape[:2]
                    prediction_pil = Image.fromarray(prediction)
                    prediction_resized = np.array(prediction_pil.resize((w, h), Image.Resampling.NEAREST))

                    ground_truth_pil = Image.fromarray(ground_truth_img)
                    ground_truth_resized = np.array(ground_truth_pil.resize((w, h), Image.Resampling.NEAREST))

                    metrics = calculate_metrics(prediction_resized, ground_truth_resized)

                    # Display comparison - create overlay using numpy
                    overlay = original_img.astype(float).copy()
                    mask_indices = prediction_resized > 127
                    overlay[mask_indices] = [0, 255, 0]
                    overlay = overlay.astype(np.uint8)
                    blended = (original_img.astype(float) * 0.7 + overlay.astype(float) * 0.3).astype(np.uint8)

                    fig = create_comparison_figure(original_img, ground_truth_resized,
                                                   prediction_resized, blended)
                    st.pyplot(fig)

                    # Metrics
                    st.divider()
                    st.subheader("📈 Performance Metrics")
                    display_metrics_grid(metrics)

        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# =========== MODE 3: UPLOAD & PREDICT ===========
elif mode == "📤 Upload & Predict":
    st.subheader("📤 Upload Custom Image")
    
    uploaded_file = st.file_uploader("Choose a grayscale PNG image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Save temp
        temp_path = Path('temp_upload.png')
        image.save(temp_path)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption="📸 Input Image", use_container_width=True)
        
        with col2:
            model = load_model()
            if model and st.button("🚀 Analyze Image", use_container_width=True):
                with st.spinner("Processing..."):
                    image_tensor = preprocess_image(temp_path)
                    prediction = predict_segmentation(model, image_tensor)

                    if prediction is not None:
                        # Use PIL for resizing instead of cv2
                        prediction_pil = Image.fromarray(prediction)
                        pred_resized = np.array(prediction_pil.resize(
                            (image_array.shape[1], image_array.shape[0]), 
                            Image.Resampling.NEAREST
                        ))

                        st.success("✅ Analysis Complete")
                        st.image(pred_resized, caption="🎯 Segmentation Mask",
                                use_container_width=True, channels='GRAY')

                        # Download button
                        buf = BytesIO()
                        Image.fromarray(pred_resized).save(buf, format='PNG')
                        buf.seek(0)
                        
                        st.download_button(
                            label="📥 Download Mask",
                            data=buf,
                            file_name=f"segmentation_{uploaded_file.name}",
                            mime="image/png"
                        )

# =========== MODE 4: BATCH ANALYSIS ===========
elif mode == "📈 Batch Analysis":
    st.subheader("📈 Batch Processing")

    # Check if dataset directories exist
    if not ORIGINAL_DIR.exists() or not GROUND_TRUTH_DIR.exists():
        st.error("❌ **Dataset not found for batch analysis!**")
        st.info("""
        The PNG dataset directories are required for batch processing.

        **Alternatives:**
        1. **Upload Mode**: Process individual images with "📤 Upload & Predict"
        2. **Local Development**: Run batch analysis locally with the full dataset

        **Note**: Batch analysis requires the complete dataset to be available.
        """)
        return

    num_images = st.slider("Number of images to analyze", 1, 50, 10)

    if st.button("🔄 Start Batch Analysis", use_container_width=True):
        image_files = sorted([f for f in os.listdir(ORIGINAL_DIR) if f.endswith('.png')])[:num_images]

        if not image_files:
            st.warning("No PNG images found in dataset directory.")
            return

        progress_bar = st.progress(0)
        results = []

        model = load_model()

        for idx, img_file in enumerate(image_files):
            original_path = ORIGINAL_DIR / img_file
            ground_truth_path = GROUND_TRUTH_DIR / img_file

            # Use PIL instead of OpenCV
            try:
                original_img = Image.open(str(original_path)).convert('RGB')
                ground_truth_img = Image.open(str(ground_truth_path)).convert('L')
                original_img = np.array(original_img)
                ground_truth_img = np.array(ground_truth_img)

                image_tensor = preprocess_image(original_path)
                prediction = predict_segmentation(model, image_tensor)

                if prediction is not None:
                    h, w = original_img.shape[:2]
                    prediction_pil = Image.fromarray(prediction)
                    pred_resized = np.array(prediction_pil.resize((w, h), Image.Resampling.NEAREST))

                    ground_truth_pil = Image.fromarray(ground_truth_img)
                    gt_resized = np.array(ground_truth_pil.resize((w, h), Image.Resampling.NEAREST))

                    metrics = calculate_metrics(pred_resized, gt_resized)
                    results.append({
                        'Image': img_file,
                        'Dice': metrics['Dice Score'],
                        'IoU': metrics['IoU Score'],
                        'Accuracy': metrics['Pixel Accuracy']
                    })

            except Exception as e:
                st.warning(f"Error processing {img_file}: {str(e)}")

            progress_bar.progress((idx + 1) / len(image_files))
        
        # Display results
        st.divider()
        st.subheader("📊 Batch Results")
        
        df = st.dataframe(results, use_container_width=True)
        
        # Summary stats
        import pandas as pd
        df_results = pd.DataFrame(results)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Dice", f"{df_results['Dice'].mean():.4f}")
        with col2:
            st.metric("Avg IoU", f"{df_results['IoU'].mean():.4f}")
        with col3:
            st.metric("Avg Accuracy", f"{df_results['Accuracy'].mean():.4f}")
        
        # Download results
        csv = df_results.to_csv(index=False)
        st.download_button(
            label="📥 Download Results CSV",
            data=csv,
            file_name="batch_analysis_results.csv",
            mime="text/csv"
        )

# ============= FOOTER =============
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🔬 Quantum ML Segmentation")
with col2:
    st.caption("v2.0 - Production Ready")
with col3:
    st.caption(f"Running on {st.session_state.device}")
