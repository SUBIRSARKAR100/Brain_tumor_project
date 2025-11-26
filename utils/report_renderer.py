import streamlit as st

def render_tumor_report(pred_class, confidence, area, peak, mean_intensity, centroid, bbox, original_mri_path=None, heatmap_path=None):
    """
    Renders a clean, modern, doctor-friendly diagnostic dashboard for Tumor Localization Analysis.
    
    Args:
        pred_class (str): The predicted class (e.g., 'Brain Tumor', 'No Brain Tumor').
        confidence (float): The confidence score of the prediction (0-100).
        area (float): Activation area percentage.
        peak (float): Peak intensity value.
        mean_intensity (float): Mean intensity value.
        centroid (tuple): (y, x) coordinates of the tumor centroid.
        bbox (tuple): (y_min, x_min, y_max, x_max) bounding box coordinates.
        original_mri_path (str, optional): Path to the original MRI image.
        heatmap_path (str, optional): Path to the heatmap image.
    """
    
    # Custom CSS for the report
    st.markdown("""
        <style>
        .report-container {
            background-color: #ffffff;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            margin-bottom: 20px;
            color: #333333;
        }
        .report-title {
            font-size: 24px;
            font-weight: 700;
            color: #1c7ed6;
            margin-bottom: 20px;
            border-bottom: 2px solid #f0f2f6;
            padding-bottom: 10px;
        }
        /* Make diagnosis card simple and white per request */
        .diagnosis-card {
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            margin-bottom: 20px;
            background-color: #ffffff; /* simple white card */
            color: #212529; /* dark text for contrast */
            border: 1px solid #e9ecef; /* subtle border */
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }
        /* Keep the semantic class but make it visually identical to the base card
           so both 'success' and 'neutral' results display as a simple white box */
        .diagnosis-success {
            background-color: #ffffff;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
            border: 1px solid #e9ecef;
            color: #212529;
        }
        .diagnosis-neutral {
            background: linear-gradient(135deg, #6c757d 0%, #adb5bd 100%);
            box-shadow: 0 4px 15px rgba(108, 117, 125, 0.3);
        }
        .diagnosis-text {
            font-size: 22px;
            font-weight: 600;
            margin-bottom: 5px;
        }
        .confidence-text {
            font-size: 36px;
            font-weight: 800;
            margin: 0;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .confidence-label {
            font-size: 14px;
            opacity: 0.9;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .section-title {
            font-size: 18px;
            font-weight: 600;
            color: #0066cc !important;
            margin-top: 20px;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #e9ecef;
            text-align: center;
            transition: transform 0.2s;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
            border-color: #dee2e6;
        }
        .metric-value {
            font-size: 20px;
            font-weight: 700;
            color: #212529;
        }
        .metric-label {
            font-size: 12px;
            color: #6c757d;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .clinical-box {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            padding: 12px;
            border-radius: 4px;
            color: #333333;
            font-size: 14px;
            line-height: 1.6;
            margin-top: 15px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="report-title">Tumor Detection Report</div>', unsafe_allow_html=True)

    # 2. Diagnosis Card - Check if it's exactly "Brain Tumor" (not "No Brain Tumor")
    # Debug: Print the prediction to verify
    print(f"DEBUG: pred_class = '{pred_class}', type = {type(pred_class)}")
    
    is_tumor = pred_class == "Brain Tumor"
    
    if is_tumor:
        diagnosis_text = "🧠 Brain Tumor Detected"
        card_style = "diagnosis-success" # Green
    else:
        diagnosis_text = "✔ No Tumor Detected"
        card_style = "diagnosis-success" # Green for healthy result too
    
    st.markdown(f"""
        <div class="diagnosis-card {card_style}">
            <div class="diagnosis-text">{diagnosis_text}</div>
            <div class="confidence-text">{confidence:.1f}%</div>
            <div class="confidence-label">Confidence Score</div>
        </div>
    """, unsafe_allow_html=True)

    # 3. Localization Summary
    st.markdown('<div style="font-size: 18px; font-weight: 600; color: #0066cc; margin-top: 20px; margin-bottom: 15px;">Localization Summary</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Activation Area", f"{area:.2f}%")
        st.metric("Peak Intensity", f"{peak:.4f}")
        
    with col2:
        st.metric("Mean Intensity", f"{mean_intensity:.4f}")
        st.metric("Tumor Coordinates", f"({centroid[1]:.0f}, {centroid[0]:.0f})") # x, y

    # 4. Clinical Suggestion
    if is_tumor:
        st.markdown("""
            <div style="font-size: 18px; font-weight: 600; color: #0066cc; margin-top: 20px; margin-bottom: 15px;">Clinical Suggestion</div>
            <div class="clinical-box">
                <strong>Recommendation:</strong> Activation patterns indicate a strong likelihood of tumor presence. 
                A radiologist review and further MRI evaluation are recommended.
            </div>
        """, unsafe_allow_html=True)
    else:
         st.markdown("""
            <div style="font-size: 18px; font-weight: 600; color: #0066cc; margin-top: 20px; margin-bottom: 15px;">Clinical Suggestion</div>
            <div class="clinical-box">
                <strong>Note:</strong> No significant tumor activation patterns detected. 
                Standard routine check-up is advised if symptoms persist.
            </div>
        """, unsafe_allow_html=True)
