"""
Utility functions for QML Segmentation Application
"""

import torch
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SegmentationUtils:
    """Utility functions for image segmentation"""
    
    @staticmethod
    def load_image(image_path: str, color_space: str = 'RGB') -> Optional[np.ndarray]:
        """
        Load image from file
        
        Args:
            image_path: Path to image file
            color_space: 'RGB' or 'GRAY'
            
        Returns:
            Image array or None if failed
        """
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                logger.error(f"Failed to load image: {image_path}")
                return None
            
            if color_space == 'RGB':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif color_space == 'GRAY':
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
            return img
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return None

    @staticmethod
    def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        """Resize image to specified size"""
        return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range"""
        return image.astype(np.float32) / 255.0

    @staticmethod
    def denormalize_image(image: np.ndarray) -> np.ndarray:
        """Denormalize image from [0, 1] to [0, 255] range"""
        return (image * 255).astype(np.uint8)

    @staticmethod
    def create_overlay(image: np.ndarray, mask: np.ndarray, 
                      color: Tuple[int, int, int] = (0, 255, 0),
                      alpha: float = 0.3) -> np.ndarray:
        """
        Create overlay of segmentation mask on image
        
        Args:
            image: Original image (RGB)
            mask: Binary segmentation mask
            color: RGB color for overlay
            alpha: Transparency of overlay
            
        Returns:
            Overlaid image
        """
        try:
            overlay = image.copy().astype(np.float32)
            mask_indices = mask > 127
            
            for c in range(3):
                overlay[mask_indices, c] = image[mask_indices, c] * (1 - alpha) + color[c] * alpha
                
            return overlay.astype(np.uint8)
        except Exception as e:
            logger.error(f"Error creating overlay: {str(e)}")
            return image

    @staticmethod
    def create_comparison_image(original: np.ndarray, prediction: np.ndarray,
                               ground_truth: Optional[np.ndarray] = None) -> np.ndarray:
        """Create comparison image showing original, prediction, and optionally ground truth"""
        try:
            h, w = original.shape[:2]
            
            if ground_truth is not None:
                # Create 2x2 grid
                comparison = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
                comparison[0:h, 0:w] = original
                comparison[0:h, w:w*2] = SegmentationUtils.create_overlay(original, prediction)
                
                # Add ground truth
                gt_rgb = cv2.cvtColor(ground_truth, cv2.COLOR_GRAY2RGB)
                comparison[h:h*2, 0:w] = gt_rgb
                comparison[h:h*2, w:w*2] = SegmentationUtils.create_overlay(original, ground_truth)
            else:
                # Create 1x2 grid
                comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
                comparison[0:h, 0:w] = original
                comparison[0:h, w:w*2] = SegmentationUtils.create_overlay(original, prediction)
                
            return comparison
        except Exception as e:
            logger.error(f"Error creating comparison: {str(e)}")
            return original

class MetricsCalculator:
    """Calculate segmentation metrics"""
    
    @staticmethod
    def dice_score(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Calculate Dice Score (F1-Score)
        
        Formula: 2 * (A ∩ B) / (|A| + |B|)
        """
        pred = (prediction > 127).astype(np.uint8)
        gt = (ground_truth > 127).astype(np.uint8)
        
        intersection = np.sum(pred * gt)
        dice = 2.0 * intersection / (np.sum(pred) + np.sum(gt) + 1e-7)
        
        return float(dice)

    @staticmethod
    def iou_score(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Calculate IoU Score (Jaccard Index)
        
        Formula: (A ∩ B) / (A ∪ B)
        """
        pred = (prediction > 127).astype(np.uint8)
        gt = (ground_truth > 127).astype(np.uint8)
        
        intersection = np.sum(pred * gt)
        union = np.sum(np.logical_or(pred, gt))
        iou = intersection / (union + 1e-7)
        
        return float(iou)

    @staticmethod
    def pixel_accuracy(prediction: np.ndarray, ground_truth: np.ndarray) -> float:
        """
        Calculate Pixel Accuracy
        
        Formula: Correctly classified pixels / Total pixels
        """
        pred = (prediction > 127).astype(np.uint8)
        gt = (ground_truth > 127).astype(np.uint8)
        
        accuracy = np.sum(pred == gt) / pred.size
        
        return float(accuracy)

    @staticmethod
    def calculate_all_metrics(prediction: np.ndarray, 
                             ground_truth: np.ndarray) -> Dict[str, float]:
        """Calculate all metrics at once"""
        return {
            'Dice Score': MetricsCalculator.dice_score(prediction, ground_truth),
            'IoU Score': MetricsCalculator.iou_score(prediction, ground_truth),
            'Pixel Accuracy': MetricsCalculator.pixel_accuracy(prediction, ground_truth),
        }


class ImageProcessor:
    """Complete image processing pipeline"""
    
    def __init__(self, input_size: Tuple[int, int] = (512, 512)):
        self.input_size = input_size
        
    def preprocess(self, image_path: str) -> Optional[torch.Tensor]:
        """Complete preprocessing pipeline"""
        # Load image
        img = SegmentationUtils.load_image(image_path, color_space='RGB')
        if img is None:
            return None
        
        # Resize
        img = SegmentationUtils.resize_image(img, self.input_size)
        
        # Normalize
        img = SegmentationUtils.normalize_image(img)
        
        # Convert to tensor
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
        
        return img
    
    def postprocess(self, output: torch.Tensor) -> np.ndarray:
        """Convert model output to segmentation mask"""
        # Convert to numpy
        output = output.squeeze().cpu().numpy()
        
        # Apply threshold
        mask = (output > 0.5).astype(np.uint8) * 255
        
        return mask


def get_image_list(directory: Path, extensions: List[str] = None) -> List[Path]:
    """Get list of images in directory"""
    if extensions is None:
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
    
    images = []
    for ext in extensions:
        images.extend(directory.glob(ext.lower()))
        images.extend(directory.glob(ext.upper()))
    
    return sorted(images)


def batch_process_images(image_dir: Path, model, processor: ImageProcessor,
                        device: torch.device) -> Dict:
    """Process all images in a directory"""
    results = {
        'processed': 0,
        'failed': 0,
        'metrics': []
    }
    
    images = get_image_list(image_dir)
    
    for image_path in images:
        try:
            # Preprocess
            img_tensor = processor.preprocess(str(image_path))
            if img_tensor is None:
                results['failed'] += 1
                continue
            
            # Predict
            with torch.no_grad():
                output = model(img_tensor.to(device))
            
            # Postprocess
            mask = processor.postprocess(output)
            results['metrics'].append({
                'image': image_path.name,
                'mask': mask
            })
            results['processed'] += 1
            
        except Exception as e:
            logger.error(f"Failed to process {image_path}: {str(e)}")
            results['failed'] += 1
    
    return results


# Color maps for visualization
COLORMAPS = {
    'jet': cv2.COLORMAP_JET,
    'hot': cv2.COLORMAP_HOT,
    'cool': cv2.COLORMAP_COOL,
    'spring': cv2.COLORMAP_SPRING,
    'summer': cv2.COLORMAP_SUMMER,
    'autumn': cv2.COLORMAP_AUTUMN,
    'winter': cv2.COLORMAP_WINTER,
    'rainbow': cv2.COLORMAP_RAINBOW,
}


def apply_colormap(mask: np.ndarray, colormap_name: str = 'jet') -> np.ndarray:
    """Apply colormap to grayscale mask"""
    colormap = COLORMAPS.get(colormap_name, cv2.COLORMAP_JET)
    return cv2.applyColorMap(mask, colormap)
