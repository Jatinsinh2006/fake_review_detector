import os
import joblib
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import train_test_split
from src.config import PROCESSED_DATA_DIR, RANDOM_STATE, TEST_SIZE

def run_explainability():
    print("🔍 Starting Phase 8: Explainable AI (SHAP) Pipeline...")
    
    # 1. Load preprocessed data
    preprocessed_path = os.path.join(PROCESSED_DATA_DIR, "final_preprocessed_reviews.csv")
    if not os.path.exists(preprocessed_path):
        raise FileNotFoundError(f"❌ Preprocessed file not found at {preprocessed_path}.")
        
    df = pd.read_csv(preprocessed_path)
    df['lemmatized_text'] = df['lemmatized_text'].fillna("")
    
    X = df['lemmatized_text'].astype(str).values
    y = df['label'].values
    
    # Train-test split (same random state to align with training)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # 2. Load the best saved ML model and its vectorizer/pipeline if saved together, 
    # Or let's load our TF-IDF + Logistic Regression components. 
    # Since our logistic regression pipeline can be wrapped, let's load the vectorizer and model.
    # Alternatively, let's check if we saved a pipeline. If not, let's quickly train a small explainer object or load vectorizer.
    
    # Let's write a robust check for vectorizer and model:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    
    print("🔄 Re-instantiating and fitting Logistic Regression pipeline for SHAP explanation...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_vec, y_train)
    
    print("📊 Initializing SHAP LinearExplainer...")
    # For linear models like Logistic Regression, shap.LinearExplainer is extremely fast and accurate
    explainer = shap.LinearExplainer(model, X_train_vec, feature_names=vectorizer.get_feature_names_out())
    
    # Calculate SHAP values for a subset of test data (e.g., first 100 samples for quick explanation demonstration)
    subset_size = 100
    print(f"⚙️ Calculating SHAP values for {subset_size} test samples...")
    shap_values = explainer(X_test_vec[:subset_size])
    
    print("\n✅ SHAP Explainer successfully initialized and values computed!")
    print(f"💡 SHAP Output Shape: {shap_values.values.shape}")
    print("🎯 You can now integrate these SHAP values into the Streamlit dashboard for word-level explanations!")

if __name__ == "__main__":
    run_explainability()