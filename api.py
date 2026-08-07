from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import pandas as pd
import numpy as np
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.config import PROCESSED_DATA_DIR, RANDOM_STATE, TEST_SIZE

# Initialize FastAPI App
app = FastAPI(
    title="AI-Powered Fake Product Review Detection API",
    description="Production-grade REST API for real-time review verification, trust scoring, and explainable AI insights.",
    version="1.0.0"
)

# Request Body Schema
class ReviewRequest(BaseModel):
    review_text: str

# Global variables for caching model components
model = None
vectorizer = None
explainer = None

@app.on_event("startup")
def load_artifacts():
    global model, vectorizer, explainer
    print("🔄 Loading dataset and training pipeline for API backend...")
    
    preprocessed_path = os.path.join(PROCESSED_DATA_DIR, "final_preprocessed_reviews.csv")
    if not os.path.exists(preprocessed_path):
        raise FileNotFoundError(f"❌ Preprocessed file not found at {preprocessed_path}.")
        
    df = pd.read_csv(preprocessed_path)
    df['lemmatized_text'] = df['lemmatized_text'].fillna("")
    
    X = df['lemmatized_text'].astype(str).values
    y = df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_vec, y_train)
    
    explainer = shap.LinearExplainer(model, X_train_vec, feature_names=vectorizer.get_feature_names_out())
    print("✅ API Backend successfully initialized and models loaded!")

@app.get("/")
def home():
    return {
        "status": "Online",
        "message": "Welcome to the AI-Powered Fake Review Detection API. Use /predict endpoint to analyze reviews."
    }

@app.post("/predict")
def predict_review(payload: ReviewRequest):
    text = payload.review_text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Review text cannot be empty.")
        
    # Vectorize and Predict
    review_vec = vectorizer.transform([text])
    prediction = int(model.predict(review_vec)[0])
    probabilities = model.predict_proba(review_vec)[0] # [Genuine, Fake]
    
    genuine_prob = float(probabilities[0])
    fake_prob = float(probabilities[1])
    
    confidence = float(max(probabilities) * 100)
    trust_score = float(round(genuine_prob * 100, 2))
    spam_score = float(round(fake_prob * 100, 2))
    
    status = "FAKE REVIEW" if prediction == 1 else "GENUINE REVIEW"
    
    # SHAP Insights
    shap_vals = explainer(review_vec)
    feature_names = vectorizer.get_feature_names_out()
    row_shap = shap_vals.values[0]
    top_indices = np.argsort(np.abs(row_shap))[::-1][:3]
    
    key_factors = []
    for idx in top_indices:
        word = feature_names[idx]
        impact = float(row_shap[idx])
        effect = "Pushed towards Fake" if impact > 0 else "Pushed towards Genuine"
        key_factors.append({"word": word, "impact": round(impact, 4), "effect": effect})
        
    return {
        "review": text,
        "prediction_label": status,
        "prediction_code": prediction,
        "confidence_score_percent": round(confidence, 2),
        "trust_score": trust_score,
        "spam_risk_score": spam_score,
        "key_factors": key_factors
    }