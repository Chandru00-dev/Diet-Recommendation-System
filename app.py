import streamlit as st
import traceback

st.set_page_config(page_title="AI Nutritionist", page_icon="🤖", layout="centered")

try:
    import joblib
    import numpy as np
    import pandas as pd
    import re
    import easyocr
    import torch
    from transformers import pipeline
    import tensorflow as tf
    from PIL import Image
    import io

    @st.cache_resource
    def load_resources():
        model = tf.keras.models.load_model('saved_model/ai_nutrition_model.keras')
        le_diet = joblib.load('saved_model/le_diet.pkl')
        scaler = joblib.load('saved_model/scaler.pkl')
        feature_cols = joblib.load('saved_model/feature_cols.pkl')
        df_food = pd.read_csv('Indian_Food_Nutrition_Processed.csv')
        return model, le_diet, scaler, feature_cols, df_food

    @st.cache_resource
    def load_ocr_bert():
        reader = easyocr.Reader(['en'], gpu=False)
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
        return reader, classifier

    model, le_diet, scaler, feature_cols, df_food = load_resources()
    reader, classifier = load_ocr_bert()

    st.success("✅ All models and resources loaded successfully!")

except Exception as e:
    st.error("🚨 Failed to load resources")
    st.error(str(e))
    st.code(traceback.format_exc())
    st.stop()

# Rest of your recommend_diet function and chat UI (same as before)
def recommend_diet(age=30, user_query="", image_bytes=None):
    # ... (keep your existing recommend_diet function exactly as it was)
    pass   # ← Replace this line with your full recommend_diet function from previous version

# ====================== CHAT UI ======================
st.title("AI Nutrition Chat")
st.caption("Talk naturally • Upload medical reports • Get instant diet plans")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask me anything about your diet...")
uploaded_file = st.file_uploader("Upload Medical Report (optional)", type=["png", "jpg", "jpeg"])

if prompt or uploaded_file is not None:
    # ... (keep the rest of your chat code exactly the same)
    pass