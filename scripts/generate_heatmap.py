"""
Script to generate and save Grad-CAM heatmaps with analysis for test images.
Usage: python scripts/generate_heatmap.py [image_path]

Grad-CAM (Gradient-weighted Class Activation Mapping) visualizes which regions of the input
image are most important for the CNN's classification decision, helping localize tumors.
"""

import sys
sys.path.insert(0, 'D:\\stremlit_brain')

from tensorflow import keras
from PIL import Image
import numpy as np
import tensorflow as tf
from classifier.gradcam import make_gradcam_heatmap, overlay_heatmap_on_image, analyze_heatmap, find_last_conv_layer, get_focused_mask
from classifier.classify import preprocessing
import os
from pathlib import Path

# Create output directory
heatmap_dir = Path('scripts/heatmaps')
heatmap_dir.mkdir(exist_ok=True, parents=True)

# Load model
model = keras.models.load_model('Model/model.keras')
class_names = {0: 'No Brain Tumor', 1: 'Brain Tumor'}

# Use first test image if no argument provided
if len(sys.argv) > 1:
    img_path = sys.argv[1]
else:
    img_path = 'Brain_Tumor_Datasets/Brain_Tumor_Datasets/test/YES/yes_1892.jpg'

print(f"Loading image: {img_path}")
img = Image.open(img_path).convert('RGB')

# Preprocess using existing preprocessing to get correct input shape
preprocessed = preprocessing(img)

# Get prediction
pred = model.predict(preprocessed)[0]
pred_index = np.argmax(pred)
confidence = float(np.max(pred)) * 100
print(f"Prediction: {class_names[pred_index]} ({confidence:.2f}%)")

# Generate heatmap using Grad-CAM
try:
    heatmap = make_gradcam_heatmap(preprocessed, model, pred_index=pred_index)
except ValueError as e:
    print(f"Warning: {e}, using fallback layer 'conv2d_4'")
    heatmap = make_gradcam_heatmap(preprocessed, model, last_conv_layer_name='conv2d_4', pred_index=pred_index)
print(f"Heatmap shape: {heatmap.shape}")

# Resize heatmap to match original image size for overlay
heatmap_resized = tf.image.resize(heatmap[..., tf.newaxis], (224, 224)).numpy().squeeze()

# Create overlay visualization with jet colormap (whole-image)
overlay = overlay_heatmap_on_image(img, heatmap_resized, alpha=0.5, colormap='jet')

# Compute a focused mask to highlight only the strongest activation regions
mask, used_threshold = get_focused_mask(heatmap_resized, method='percentile', percentile=90, sigma=1.5, min_area=50)
overlay_focused = overlay_heatmap_on_image(img, heatmap_resized, alpha=0.6, colormap='jet', mask=mask)

# Analyze heatmap for tumor localization using adaptive mask
analysis = analyze_heatmap(heatmap_resized, threshold=None, method='percentile', percentile=90, sigma=1.5, min_area=50)

print(f"\n=== Grad-CAM Analysis ===")
print(f"Tumor Region Found: {analysis['found']}")
if analysis['found']:
    area_pct = analysis['area_percent']
    print(f"Activation Area: {area_pct:.2f}% of image")
    print(f"Mean Activation Intensity: {analysis['mean_intensity']:.4f}")
    print(f"Max Activation Intensity: {analysis['max_intensity']:.4f}")
    print(f"Confidence Level: {analysis['confidence']}")
    bbox = analysis['bbox']
    print(f"Bounding Box (pixels): Y=[{bbox[0]}, {bbox[2]}], X=[{bbox[1]}, {bbox[3]}]")
    centroid = analysis['centroid']
    print(f"Tumor Centroid (pixels): ({centroid[0]:.1f}, {centroid[1]:.1f})")
    print("\nInterpretation: The heatmap shows regions the model considers important.")
    print("Blue -> Green -> Yellow -> Red: Low to High activation (likely tumor regions).")
else:
    print("No significant activation regions detected above threshold.")

# Save outputs
image_name = Path(img_path).stem
heatmap_img_path = heatmap_dir / f'{image_name}_heatmap.png'
overlay_img_path = heatmap_dir / f'{image_name}_overlay.png'
analysis_txt_path = heatmap_dir / f'{image_name}_analysis.txt'

# Save heatmap as grayscale
heatmap_uint8 = (255 * heatmap_resized).astype('uint8')
heatmap_pil = Image.fromarray(heatmap_uint8)
heatmap_pil.save(str(heatmap_img_path))
print(f"\nSaved heatmap to: {heatmap_img_path}")

# Save overlay
overlay.save(str(overlay_img_path))

# Save focused overlay and mask
focused_overlay_path = heatmap_dir / f'{image_name}_overlay_focused.png'
mask_path = heatmap_dir / f'{image_name}_mask.png'
overlay_focused.save(str(focused_overlay_path))
Image.fromarray((mask.astype('uint8') * 255)).save(str(mask_path))
print(f"Saved focused overlay to: {focused_overlay_path}")
print(f"Saved mask to: {mask_path}")
print(f"Saved overlay to: {overlay_img_path}")

# Save analysis as text report
with open(str(analysis_txt_path), 'w') as f:
    f.write(f"Image: {img_path}\n")
    f.write(f"Prediction: {class_names[pred_index]}\n")
    f.write(f"Confidence: {confidence:.2f}%\n\n")
    f.write("=== Grad-CAM Analysis ===\n")
    f.write(f"Tumor Region Found: {analysis['found']}\n")
    if analysis['found']:
        f.write(f"Activation Area: {analysis['area_percent']:.2f}% of image\n")
        f.write(f"Mean Activation Intensity: {analysis['mean_intensity']:.4f}\n")
        f.write(f"Max Activation Intensity: {analysis['max_intensity']:.4f}\n")
        f.write(f"Confidence Level: {analysis['confidence']}\n")
        bbox = analysis['bbox']
        f.write(f"Bounding Box (pixels): Y=[{bbox[0]}, {bbox[2]}], X=[{bbox[1]}, {bbox[3]}]\n")
        centroid = analysis['centroid']
        f.write(f"Tumor Centroid (pixels): ({centroid[0]:.1f}, {centroid[1]:.1f})\n")
        f.write("\nInterpretation:\n")
        f.write("The heatmap shows regions the model considers important for its classification.\n")
        f.write("Blue -> Green -> Yellow -> Red: Low to High activation (likely tumor regions).\n")
        f.write("The bounding box and centroid show the spatial location of detected activation.\n")
    else:
        f.write("No significant activation regions detected above threshold.\n")
        
print(f"Saved analysis to: {analysis_txt_path}")
