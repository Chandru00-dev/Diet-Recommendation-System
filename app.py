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

def recommend_diet(age=30, user_query="", image_bytes=None):
    extracted_text = ""
    health_condition = "General Healthy"
    
    # 1. Process Image (OCR on Medical Reports)
    if image_bytes is not None:
        try:
            results = reader.readtext(image_bytes)
            extracted_text = " ".join([text for (_, text, _) in results])
        except Exception as e:
            return f"Error reading medical report: {str(e)}"
            
    text_to_analyze = extracted_text + " " + user_query
    
    # 2. NLP intent / condition parsing
    if text_to_analyze.strip():
        candidate_labels = ["diabetes", "heart disease", "weight loss", "weight gain", "general healthy", "kidney disease", "hypertension"]
        result = classifier(text_to_analyze, candidate_labels)
        health_condition = result['labels'][0]
        
    # 3. Model Inference (Predicting Diet Class)
    try:
        # Create a dummy payload array for the model features
        input_data = pd.DataFrame(columns=feature_cols)
        input_data.loc[0] = 0  # Fill missing features with 0s for now
        if 'Age' in input_data.columns:
            input_data['Age'] = age
            
        scaled_input = scaler.transform(input_data)
        pred = model.predict(scaled_input, verbose=0)
        diet_class_idx = np.argmax(pred, axis=1)[0]
        predicted_diet = le_diet.inverse_transform([diet_class_idx])[0]
    except Exception as e:
        predicted_diet = "Balanced Diet (Fallback)"
        
    # 4. Recommend Foods from Dataset
    try:
        recommendations = df_food.sample(5)['Dish Name'].tolist()
    except Exception:
        recommendations = ["Oats", "Salad", "Grilled Chicken", "Brown Rice"]

    # 5. Build Response
    response = f"**Detected Health Goal / Condition:** {health_condition.title()}\n\n"
    if extracted_text:
        response += f"*(Extracted text from report: {extracted_text[:60]}...)*\n\n"
    response += f"**AI Recommended Diet Plan:** {predicted_diet}\n\n"
    response += "**Suggested Foods for you:**\n"
    for food in recommendations:
        response += f"- {food}\n"
        
    return response

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
    user_message = prompt if prompt else "Uploaded Medical Report"
    
    # 1. Add and show user message
    st.session_state.messages.append({"role": "user", "content": user_message})
    with st.chat_message("user"):
        st.markdown(user_message)
        
    img_bytes = None
    if uploaded_file is not None:
        img_bytes = uploaded_file.read()
        
    # 2. Get AI Response
    with st.chat_message("assistant"):
        with st.spinner("AI is analyzing..."):
            response = recommend_diet(user_query=prompt if prompt else "", image_bytes=img_bytes)
            st.markdown(response)
            
    # 3. Add to history
    st.session_state.messages.append({"role": "assistant", "content": response})