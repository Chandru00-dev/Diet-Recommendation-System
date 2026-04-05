import streamlit as st
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

st.set_page_config(page_title="AI Nutritionist", page_icon="🤖", layout="centered")

# Load model & data (cached)
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model('saved_model/ai_nutrition_model.keras')
    le_diet = joblib.load('saved_model/le_diet.pkl')
    scaler = joblib.load('saved_model/scaler.pkl')
    feature_cols = joblib.load('saved_model/feature_cols.pkl')
    df_food = pd.read_csv('Indian_Food_Nutrition_Processed.csv')
    return model, le_diet, scaler, feature_cols, df_food

model, le_diet, scaler, feature_cols, df_food = load_resources()

@st.cache_resource
def load_ocr_bert():
    reader = easyocr.Reader(['en'], gpu=False)  # CPU for cloud
    classifier = pipeline("zero-shot-classification", 
                         model="facebook/bart-large-mnli", 
                         device=-1)  # CPU
    return reader, classifier

reader, classifier = load_ocr_bert()

def recommend_diet(age=30, user_query="", image_bytes=None):
    extracted_text = ""
    if image_bytes:
        image = Image.open(io.BytesIO(image_bytes))
        result = reader.readtext(np.array(image))
        extracted_text = " ".join([text[1] for text in result])
        st.info(f"📄 Extracted text: {extracted_text[:250]}...")

    full_context = f"Age: {age}. Query: {user_query}. Report: {extracted_text}"
    
    candidate_labels = ["diabetes", "hypertension", "high cholesterol", "obesity", "heart disease", "normal"]
    parsed = classifier(full_context, candidate_labels)
    condition = parsed['labels'][0]
    confidence = parsed['scores'][0]

    diet_plan = "Balanced"
    if "diabetes" in condition or re.search(r'glucose|blood sugar|HbA1c', extracted_text, re.I):
        diet_plan = "Low Carb Diabetic"
    elif "hypertension" in condition or re.search(r'blood pressure|BP', extracted_text, re.I):
        diet_plan = "Low Sodium DASH"
    elif "cholesterol" in condition:
        diet_plan = "Low Fat Heart Healthy"
    elif "obesity" in condition:
        diet_plan = "Calorie Deficit High Protein"

    # Model fallback
    try:
        sample = np.zeros((1, len(feature_cols)))
        sample_scaled = scaler.transform(sample)
        pred = model.predict(sample_scaled, verbose=0)
        model_diet = le_diet.inverse_transform([np.argmax(pred)])[0]
        if confidence < 0.6:
            diet_plan = model_diet
    except:
        pass

    foods = df_food.sample(5)['Dish Name'].tolist() if 'Dish Name' in df_food.columns else df_food.sample(5).iloc[:,0].tolist()

    return f"""
🎯 **Your AI Nutritionist says:**

• **Condition detected**: {condition} (confidence: {confidence:.2f})
• **Recommended Diet**: **{diet_plan}**
• **Suggested Indian Meals**:
{chr(10).join(['• ' + f for f in foods])}

💡 **Tip**: {user_query if user_query else 'Follow this plan daily'}
📌 *This is AI-generated. Consult your doctor.*
"""

# ====================== CHAT UI ======================
st.title("🤖 AI Nutrition Chat")
st.caption("Talk naturally • Upload medical reports • Get instant diet plans")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask me anything about your diet (e.g. I have diabetes and high BP)")
uploaded_file = st.file_uploader("📸 Upload Medical Report (optional)", type=["png", "jpg", "jpeg"])

if prompt or uploaded_file is not None:
    # Show user message
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

    image_bytes = uploaded_file.getvalue() if uploaded_file else None
    if image_bytes:
        with st.chat_message("user"):
            st.image(image_bytes, caption="Uploaded Report", use_column_width=True)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing report & generating plan..."):
            response = recommend_diet(
                age=30,
                user_query=prompt or "Medical report uploaded",
                image_bytes=image_bytes
            )
            st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})