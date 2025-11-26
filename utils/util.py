
import streamlit as st
import base64


def set_background(img_path):
    with open(img_path,'rb') as f:
        img_data = f.read()
        
    b64_encoded = base64.b64encode(img_data).decode()
    
    style = f"""
            <style>
            .stApp{{
                background-image: url(data:image/png;base64,{b64_encoded});
                background-size: cover;
            }}
            </style>
            """
    st.markdown(style, unsafe_allow_html=True)
