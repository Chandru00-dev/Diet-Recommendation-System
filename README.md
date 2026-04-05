# AI Nutrition System

Intelligent Diet Recommendation Chat Application
Powered by Medical Report Image Analysis and Machine Learning

---

## Project Overview

The AI Nutrition System is a conversational diet recommendation platform. It provides personalized Indian diet plans based on:

- Uploaded medical report images (blood tests and lab reports)
- Natural language text queries (age, medical conditions, dietary goals)
- Combination of image and text input

The system extracts text from medical reports, detects health conditions, and generates accurate diet recommendations using a trained machine learning model.

---

## Key Features

- Conversational chat interface for natural interaction
- Medical report image processing with optical character recognition
- Automatic detection of medical conditions using natural language processing
- TensorFlow-based personalized diet recommendation model
- Culturally relevant Indian meal suggestions
- Full conversation history within each session
- Mobile-friendly responsive design

---

## Tech Stack

| Component                    | Technology Used                  |
|------------------------------|----------------------------------|
| Machine Learning Model       | TensorFlow and Keras             |
| Image Text Extraction        | EasyOCR (CNN-based)              |
| Natural Language Processing  | BERT (Zero-shot classification)  |
| User Interface               | Streamlit                        |
| Data Processing              | Pandas, NumPy, Joblib            |
| Deployment                   | Streamlit Community Cloud        |

---

## Datasets Used

- Personalized Medical Diet Recommendations Dataset (Kaggle)
- Indian Food Nutrition Processed Dataset (Kaggle)

---

## System Workflow

1. User sends a message and optionally uploads a medical report image
2. EasyOCR extracts text from the uploaded image
3. BERT analyzes the combined input to detect medical conditions
4. TensorFlow model combined with rule-based logic generates the diet recommendation
5. System returns the condition detected, recommended diet type, and specific Indian meal suggestions

---

## How to Run Locally

### Prerequisites
- Python 3.10 or higher
- Git

### Installation Steps

1. Clone the repository
   ```bash
   git clone https://github.com/YOUR-USERNAME/ai-nutrition-system.git
   cd ai-nutrition-system