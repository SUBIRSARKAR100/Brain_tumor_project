
from PIL import ImageOps
import numpy as np


def preprocessing(img):
    """Preprocesses an image for classification."""
    # Ensure grayscale, resize to model's expected input, add channel and batch dims
    img = ImageOps.grayscale(img)
    img = img.resize((224, 224))
    img_pred = np.array(img).astype('float32')
    # add channel dimension (H, W, 1)
    img_pred = np.expand_dims(img_pred, axis=-1)
    # add batch dimension (1, H, W, 1)
    img_pred = np.expand_dims(img_pred, axis=0)
    img_pred = img_pred / 255.0
    return img_pred


def classifier(image, model, class_names):    
    """Classifies an image as Brain Tumor or No Brain Tumor."""
    preprocessed_img = preprocessing(image)
    
    pred = model.predict(preprocessed_img)[0]
    pred_index = np.argmax(pred, axis=-1)
    score = int(np.round(pred[pred_index], 4) * 100) 
    label = class_names[pred_index]
    
    return label,score
