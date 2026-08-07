import os
import sys

# Ensure root directory is in path so 'src' can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

print("🔥 SCRIPT STARTED SUCCESSFULLY!")

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from src.config import PROCESSED_DATA_DIR, RANDOM_STATE, TEST_SIZE
from src.features import extract_tfidf_features

def run_pipeline():
    print("🚀 Running ML Training Pipeline...")
    
    preprocessed_path = os.path.join(PROCESSED_DATA_DIR, "final_preprocessed_reviews.csv")
    print(f"📁 Target data file: {preprocessed_path}")

    if not os.path.exists(preprocessed_path):
        print(f"❌ Error: File not found at {preprocessed_path}")
        return

    print("🔄 Loading dataset...")
    df = pd.read_csv(preprocessed_path)
    df['lemmatized_text'] = df['lemmatized_text'].fillna("")
    print(f"📊 Dataset loaded. Shape: {df.shape}")

    print("✂️ Splitting data...")
    X = df['lemmatized_text']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    X_train_vec, X_test_vec = extract_tfidf_features(X_train, X_test)

    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE, n_jobs=-1),
        "LightGBM": LGBMClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        "CatBoost": CatBoostClassifier(verbose=0, random_state=RANDOM_STATE)
    }

    results = []
    best_model_name = None
    best_f1 = 0
    best_model_obj = None

    print("\n🏋️ Training Models Now...")
    for name, model in models.items():
        print(f"----------------------------------------")
        print(f"🔹 Training {name}...")
        
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print(f"📊 {name} -> Acc: {acc:.4f} | Prec: {precision:.4f} | Rec: {recall:.4f} | F1: {f1:.4f}")
        
        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        })

        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model_obj = model

    results_df = pd.DataFrame(results)
    print("\n🏆 --- MODEL PERFORMANCE SUMMARY ---")
    print(results_df.to_string(index=False))

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "ml_model_comparison_results.csv"), index=False)
    joblib.dump(best_model_obj, os.path.join(PROCESSED_DATA_DIR, "best_ml_model.pkl"))

    print(f"\n🥇 Best Model: {best_model_name} with F1-Score: {best_f1:.4f}")
    print("💾 Saved successfully!")

if __name__ == "__main__":
    run_pipeline()