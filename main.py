
import streamlit as st
from utils.util import set_background
from utils.report_renderer import render_tumor_report
from PIL import Image
from tensorflow import keras
from classifier.classify import classifier, preprocessing
from classifier.gradcam import make_gradcam_heatmap, overlay_heatmap_on_image, analyze_heatmap, get_focused_mask
from io import BytesIO
import base64
import numpy as np
import tensorflow as tf

st.set_page_config(
    page_title='Brain Tumor Classification',
    layout='wide'
)

set_background('utils/bg.jpg')

st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 60px;
        color: #f0f0f0;
        font-weight: bold;        
        margin-top: -75px;
    }
    .header {
        display: flex;
        justify-content: center;  /* Center horizontally */
        align-items: center;  /* Center vertically (if needed) */
        text-align: center;  
        font-size: 30px;
        color: #87cefa;
        white-space: nowrap;
        margin-top: -20px;
        width: 100%;  /* Ensures full width */
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="title">Brain Tumor Classification</div>', unsafe_allow_html=True)
st.markdown('<div class="header">Upload an image to classify it as Brain Tumor or No Brain Tumor.</div>', unsafe_allow_html=True)

file = st.file_uploader('Upload MRI Image', type=['jpg','jpeg','png','jfif'])

model = keras.models.load_model("Model/model.keras" )

class_names = {0:'No Brain Tumor', 1:'Brain Tumor'}

if file is not None:
    
    image = Image.open(file).convert('RGB')

    prediction, score = classifier(image, model, class_names)
    
    # Prepare preprocessed image for Grad-CAM
    preprocessed = preprocessing(image)
    pred = model.predict(preprocessed, verbose=0)[0]
    pred_index = np.argmax(pred)
    
    # Generate Grad-CAM heatmap
    try:
        heatmap = make_gradcam_heatmap(preprocessed, model, pred_index=pred_index)
    except ValueError:
        heatmap = make_gradcam_heatmap(preprocessed, model, last_conv_layer_name='conv2d_4', pred_index=pred_index)
    heatmap_resized = tf.image.resize(heatmap[..., tf.newaxis], (512, 512)).numpy().squeeze()
    
    # Focused mask is always used now to highlight strongest activation regions
    # Use fixed defaults for the focused mask (advanced tuning removed)
    percentile = 90
    sigma = 1.5
    min_area = 50

    # Choose colormap based on prediction: hot colors for tumor, cool colors for no tumor
    colormap = 'jet' if prediction == 'Brain Tumor' else 'winter'
    
    # Full overlay (for inspection) and focused overlay + adaptive analysis
    overlay_full = overlay_heatmap_on_image(image, heatmap_resized, alpha=0.5, colormap=colormap)

    mask, threshold_used = get_focused_mask(heatmap_resized, method='percentile', percentile=percentile, sigma=sigma, min_area=min_area)
    overlay_focused = overlay_heatmap_on_image(image, heatmap_resized, alpha=0.6, colormap=colormap, mask=mask)

    # Analysis is always based on the focused mask (adaptive)
    analysis = analyze_heatmap(heatmap_resized, threshold=None, method='percentile', percentile=percentile, sigma=sigma, min_area=min_area)
   
    bufferd = BytesIO()
    image.save(bufferd, format='PNG')
    img_base64 = base64.b64encode(bufferd.getvalue()).decode()
    
    # prepare focused overlay base64

    # Display results in columns
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown('<div style="text-align:center;"><h3>Classification Result</h3></div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="align-items: center;">        
            <img src="data:image/png;base64,{img_base64}" style="width: 100%; height: auto; object-fit: cover;"/>
            <div style="font-size:28px; font-weight:bold; margin-top: 20px; color:#FFFFFF; text-align:center;">
                <p><strong>{prediction}</strong></p>
                <p style="margin-top:5px; font-size:24px;"> Confidence: {score}% </p>
            </div>        
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown('<div style="text-align:center;"><h3>Grad-CAM Heatmap & Analysis</h3></div>', unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Overlay (full)", "Overlay (focused)"])

        # prepare full overlay base64
        overlay_full_bufferd = BytesIO()
        overlay_full.save(overlay_full_bufferd, format='PNG')
        overlay_full_b64 = base64.b64encode(overlay_full_bufferd.getvalue()).decode()

        with tab1:
            st.markdown(
                f'<div style="text-align: center;"><img src="data:image/png;base64,{overlay_full_b64}" style="width: 120%; max-width: 600px; height: auto; display: inline-block;"/></div>',
                unsafe_allow_html=True
            )

        # prepare focused overlay base64
        overlay_focused_buffer = BytesIO()
        overlay_focused.save(overlay_focused_buffer, format='PNG')
        overlay_focused_base64 = base64.b64encode(overlay_focused_buffer.getvalue()).decode()

        with tab2:
            st.markdown(
                f'<div style="text-align: center;"><img src="data:image/png;base64,{overlay_focused_base64}" style="width: 120%; max-width: 600px; height: auto; display: inline-block;"/></div>',
                unsafe_allow_html=True
            )
        


        st.markdown('---')
        
        if analysis['found']:
            render_tumor_report(
                pred_class=prediction,
                confidence=score,
                area=analysis['area_percent'],
                peak=analysis['max_intensity'],
                mean_intensity=analysis['mean_intensity'],
                centroid=analysis['centroid'],
                bbox=analysis['bbox'],
                original_mri_path=file,
                heatmap_path=None # Heatmap is generated in memory
            )
        else:
            # Show report even when no activation found, with default values
            render_tumor_report(
                pred_class=prediction,
                confidence=score,
                area=0.0,
                peak=0.0,
                mean_intensity=0.0,
                centroid=(0, 0),
                bbox=(0, 0, 0, 0),
                original_mri_path=file,
                heatmap_path=None
            )

