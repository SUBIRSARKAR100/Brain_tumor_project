import numpy as np
import tensorflow as tf
from tensorflow import keras


def find_last_conv_layer(model):
    """Find the last convolutional layer in the model."""
    conv_layers = []
    for layer in model.layers:
        # Check if it's a Conv2D layer by class name
        if 'Conv2D' in str(type(layer)):
            conv_layers.append(layer.name)
    
    if conv_layers:
        return conv_layers[-1]  # Return last conv layer
    
    raise ValueError("Could not find any Conv2D layer in model.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    Generate a Grad-CAM heatmap for a given image and model.
    """
    if last_conv_layer_name is None:
        last_conv_layer_name = find_last_conv_layer(model)

    last_conv_layer = model.get_layer(last_conv_layer_name)
    
    # Create a model to extract conv layer output
    grad_model = tf.keras.models.Model(
        inputs=model.inputs[0], 
        outputs=[last_conv_layer.output, model.outputs[0]]
    )

    # Convert input to tensor and watch it
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    
    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        conv_outputs, predictions = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Compute gradients of predictions w.r.t. input
    grads = tape.gradient(class_channel, img_tensor)
    
    if grads is None:
        raise ValueError(f"Could not compute gradients for layer {last_conv_layer_name}.")

    # Now get conv outputs and compute Grad-CAM
    conv_outputs, _ = grad_model(img_tensor)
    conv_outputs = conv_outputs[0]  # (H, W, C)
    
    # Compute gradients of class output w.r.t. conv layer
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    
    grads = tape.gradient(class_channel, conv_outputs)
    
    if grads is not None:
        # Pool gradients across spatial dimensions
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs_val = conv_outputs[0]
        heatmap = conv_outputs_val * pooled_grads[..., tf.newaxis]
        heatmap = tf.reduce_mean(heatmap, axis=-1)
    else:
        # Fallback: just use conv activation directly
        heatmap = tf.reduce_mean(conv_outputs[0], axis=-1)
    
    # ReLU and normalize
    heatmap = np.maximum(heatmap, 0)
    max_val = np.max(heatmap) if np.max(heatmap) != 0 else 1e-8
    heatmap = heatmap / max_val

    return heatmap if isinstance(heatmap, np.ndarray) else heatmap.numpy()


def overlay_heatmap_on_image(orig_img, heatmap, alpha=0.5, colormap='jet', mask=None):
    """
    Overlay a heatmap (H, W) on top of an original RGB image (H, W, 3).
    Uses a proper colormap (jet, viridis, hot, cool, etc.) for better visualization.
    Returns an RGB image (uint8) with the heatmap blended.
    """
    import PIL.Image as Image
    import matplotlib.cm as cm
    
    # Normalize heatmap to [0, 1]
    heatmap_norm = np.clip(heatmap, 0, 1)
    
    # Apply colormap (jet, viridis, hot, cool, etc.)
    cmap = cm.get_cmap(colormap)
    heatmap_colored = cmap(heatmap_norm)  # Returns (H, W, 4) RGBA
    heatmap_rgb = (heatmap_colored[..., :3] * 255).astype(np.uint8)  # (H, W, 3) RGB
    
    heatmap_img = Image.fromarray(heatmap_rgb)

    if orig_img.mode != 'RGB':
        orig = orig_img.convert('RGB')
    else:
        orig = orig_img.copy()

    # Resize heatmap to match original image
    heatmap_img = heatmap_img.resize(orig.size, resample=Image.BILINEAR)

    # If mask is provided we only color masked regions and keep other pixels unchanged
    if mask is None:
        overlay = Image.blend(orig, heatmap_img, alpha=alpha)
        return overlay

    # Resize mask to original image size and ensure boolean
    mask_resized = Image.fromarray((mask.astype('uint8') * 255)).resize(orig.size, resample=Image.NEAREST)
    mask_arr = np.array(mask_resized) > 128

    # Convert images to arrays
    orig_arr = np.array(orig).astype('float32')
    heatmap_arr = np.array(heatmap_img).astype('float32')

    # Blend only on masked pixels
    alpha_val = float(alpha)
    mask_3 = mask_arr[..., None]
    # Blend only where mask True, keep original elsewhere
    out_arr = np.where(mask_3, (1 - alpha_val) * orig_arr + alpha_val * heatmap_arr, orig_arr)

    out_img = Image.fromarray(np.clip(out_arr, 0, 255).astype('uint8'))
    return out_img


def get_focused_mask(heatmap, method='percentile', percentile=90, sigma=1.0, min_area=50):
    """
    Return a boolean mask focusing on the most relevant activation areas.
    - method: 'percentile' uses percentile threshold. 'relative' uses a fraction of max.
    - percentile: used for 'percentile' method (0-100) or relative fraction (0-1) if method=='relative'
    - sigma: gaussian smoothing sigma (reduce noise)
    - min_area: remove connected components smaller than this (in heatmap pixels)
    Returns (mask, threshold_used)
    """
    from scipy import ndimage as ndi

    h = np.array(heatmap, dtype=np.float32)
    if sigma and sigma > 0:
        h = ndi.gaussian_filter(h, sigma=sigma)

    if method == 'relative':
        thr = float(np.max(h) * float(percentile))
    else:
        thr = float(np.percentile(h, percentile))

    mask = h >= thr

    # Remove small objects
    labeled, nlabels = ndi.label(mask)
    if nlabels > 0:
        sizes = ndi.sum(mask, labeled, range(1, nlabels + 1))
        keep = [i + 1 for i, s in enumerate(sizes) if s >= min_area]
        if keep:
            newmask = np.isin(labeled, keep)
            mask = newmask

    # If mask is empty, relax threshold iteratively until we have something
    if not np.any(mask):
        for p in [80, 70, 60, 50, 40, 30, 20, 10]:
            thr = float(np.percentile(h, p))
            mask = h >= thr
            if np.any(mask):
                break

    return mask, thr


def analyze_heatmap(heatmap, threshold=None, method='percentile', percentile=90, sigma=1.0, min_area=50):
    """
    Analyze heatmap to localize tumor region and return detailed metrics.
    - threshold: normalized threshold to binarize heatmap (lower = more sensitive)
    Returns a dict with centroid, bbox, area percent, intensity stats.
    """
    h = np.array(heatmap, dtype=np.float32)

    if threshold is None:
        mask, thr = get_focused_mask(h, method=method, percentile=percentile, sigma=sigma, min_area=min_area)
        threshold = thr
    else:
        mask = h >= threshold

    result = {
        'found': bool(np.any(mask)),
        'threshold': float(threshold),
        'area_percent': 0.0,
        'bbox': None,
        'centroid': None,
        'mean_intensity': float(np.mean(h[mask])) if np.any(mask) else 0.0,
        'max_intensity': float(np.max(h[mask])) if np.any(mask) else 0.0,
            'confidence': 'Low' if np.max(h) < 0.3 else ('Medium' if np.max(h) < 0.6 else 'High'),
    }
    
    if not result['found']:
        return result

    coords = np.argwhere(mask)
    ymin, xmin = coords.min(axis=0)
    ymax, xmax = coords.max(axis=0)

    area = mask.sum()
    total = mask.size
    area_percent = float(area) / float(total) * 100.0
    result['area_percent'] = area_percent

    # Weighted centroid using heatmap intensities
    total_weight = h[mask].sum()
    if total_weight == 0:
        centroid = (float((ymin + ymax) / 2.0), float((xmin + xmax) / 2.0))
    else:
        ys, xs = np.where(mask)
        weights = h[ys, xs]
        centroid_y = np.average(ys, weights=weights)
        centroid_x = np.average(xs, weights=weights)
        centroid = (float(centroid_y), float(centroid_x))

    result['bbox'] = (int(ymin), int(xmin), int(ymax), int(xmax))
    result['centroid'] = centroid
    
    return result
