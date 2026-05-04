"""
Configuration file for QML Segmentation Application
Modify these settings to customize the app behavior
"""

# ============================================================================
# APPLICATION SETTINGS
# ============================================================================

APP_TITLE = "QML Image Segmentation"
APP_DESCRIPTION = "Advanced semantic image segmentation using deep learning"
APP_VERSION = "1.0.0"

# ============================================================================
# MODEL SETTINGS
# ============================================================================

# Model checkpoint path (relative to project root)
MODEL_PATH = "best_model.pth"

# Model parameters
MODEL_CONFIG = {
    'in_channels': 3,        # RGB image
    'out_channels': 1,       # Binary segmentation
    'bilinear': True,        # Use bilinear upsampling
    'input_size': (512, 512), # Input image size
}

# ============================================================================
# DATASET SETTINGS
# ============================================================================

# Dataset paths (relative to project root)
DATASET_ROOT = "PNG"
ORIGINAL_DIR = "PNG/Original"
GROUND_TRUTH_DIR = "PNG/Ground Truth"

# Display settings
MAX_SAMPLE_IMAGES = 50  # Limit images shown in dropdown for performance

# ============================================================================
# MODEL METRICS
# ============================================================================

# Trained model performance metrics
METRICS = {
    'Dice Score': 0.8418,
    'IoU Score': 0.7740,
    'Pixel Accuracy': 0.9795,
}

# ============================================================================
# SEGMENTATION PARAMETERS
# ============================================================================

# Prediction threshold (0-1)
PREDICTION_THRESHOLD = 0.5

# Image normalization settings
NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet mean
NORMALIZE_STD = [0.229, 0.224, 0.225]   # ImageNet std

# ============================================================================
# VISUALIZATION SETTINGS
# ============================================================================

# Overlay settings
OVERLAY_COLOR = (0, 255, 0)  # Green (BGR format: Blue, Green, Red)
OVERLAY_ALPHA = 0.3          # Transparency (0-1)

# Supported image formats
SUPPORTED_FORMATS = ['png', 'jpg', 'jpeg', 'tiff', 'bmp']

# Colormap for visualization
DEFAULT_COLORMAP = 'jet'

# ============================================================================
# PROCESSING SETTINGS
# ============================================================================

# Processing parameters
USE_GPU = True              # Use GPU if available
DEVICE_TYPE = 'cuda'       # 'cuda' or 'cpu'

# Batch processing
BATCH_SIZE = 1
MAX_WORKERS = 4

# ============================================================================
# UI/UX SETTINGS
# ============================================================================

# Streamlit configuration
STREAMLIT_CONFIG = {
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
}

# Display options
SHOW_METRICS_CARDS = True
SHOW_COMPARISON_GRID = True
SHOW_COLORMAP_OPTIONS = False  # Show colormap selector for masks

# Chart settings
CHART_HEIGHT = 400
CHART_WIDTH = 800

# ============================================================================
# TRAINING HYPERPARAMETERS (Optimized for convergence)
# ============================================================================

TRAINING_CONFIG = {
    'epochs': 50,              # Maximum training epochs
    'batch_size': 16,          # Batch size (16-32 optimal)
    'learning_rate': 0.001,    # Initial learning rate for Adam optimizer
    'weight_decay': 1e-4,      # L2 regularization
    'early_stopping_patience': 10,  # Stop if no improvement for N epochs
    'train_val_split': 0.8,    # 80% train, 20% validation
    'gradient_clip': 1.0,      # Max gradient norm for stability
}

# Loss function weights (tune these for dataset-specific performance)
LOSS_WEIGHTS = {
    'bce': 0.5,                # Binary Cross-Entropy weight
    'dice': 0.3,               # Dice Loss weight
    'iou': 0.2,                # IoU Loss weight
}

# ============================================================================
# ADVANCED SETTINGS
# ============================================================================

# Logging level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
LOG_LEVEL = 'INFO'

# Cache settings
CACHE_MODELS = True
CACHE_IMAGES = True

# Temporary files
TEMP_DIR = './temp'

# ============================================================================
# PERFORMANCE SETTINGS
# ============================================================================

# Session state settings
SESSION_CACHE_TTL = 3600  # Cache time-to-live in seconds

# Image processing
MAX_IMAGE_SIZE = 2048  # Maximum image dimension
IMAGE_QUALITY = 95    # JPEG quality for downloads

# ============================================================================
# FEATURE FLAGS
# ============================================================================

# Enable/disable features
FEATURES = {
    'dashboard': True,
    'dataset_testing': True,
    'upload_predict': True,
    'batch_processing': False,  # Coming soon
    'export_reports': False,    # Coming soon
}

# ============================================================================
# HELP MESSAGES AND TEXT
# ============================================================================

HELP_TEXTS = {
    'upload_instruction': 'Upload an image file to get real-time segmentation predictions',
    'dataset_instruction': 'Select an image from the dataset to test the model',
    'dashboard_instruction': 'View overall model performance and statistics',
}

# ============================================================================
# COLOR SCHEMES
# ============================================================================

COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'accent': '#f093fb',
    'success': '#06d6a0',
    'warning': '#ffa500',
    'error': '#ef476f',
}

# ============================================================================
# PERFORMANCE THRESHOLDS
# ============================================================================

# Metrics threshold for "good" performance
PERFORMANCE_THRESHOLDS = {
    'dice_good': 0.80,
    'iou_good': 0.75,
    'accuracy_good': 0.95,
}
