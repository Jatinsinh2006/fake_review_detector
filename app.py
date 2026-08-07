import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from src.config import PROCESSED_DATA_DIR, RANDOM_STATE, TEST_SIZE

# --- Page Configuration ---
st.set_page_config(
    page_title="AI-Powered Fake Product Review Detection System",
    page_icon="🛡️",
    layout="wide"
)

# --- Custom CSS for High-End UI ---
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        font-weight: 700;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 40px;
    }
    .metric-card {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Model & Data Pipeline Cached ---
@st.cache_resource
def load_trained_model_and_explainer():
    preprocessed_path = os.path.join(PROCESSED_DATA_DIR, "final_preprocessed_reviews.csv")
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
    
    return model, vectorizer, explainer

with st.spinner("🔄 Initializing AI Models and Explainability Pipeline... Please wait."):
    model, vectorizer, explainer = load_trained_model_and_explainer()

# --- App Header ---
st.markdown('<div class="main-header">🛡️ AI-Powered Fake Product Review Detection & Trust Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Analyze product reviews in real-time, detect fraud patterns, calculate trust scores, and uncover explainable insights.</div>', unsafe_allow_html=True)

# --- Sidebar Inputs ---
st.sidebar.header("🔍 Review Input Panel")
user_review = st.sidebar.text_area("Enter Product Review Text:", placeholder="Type or paste a product review here...")

analysis_button = st.sidebar.button("Analyze Review", type="primary")

if analysis_button:
    if not user_review.strip():
        st.sidebar.warning("⚠️ Please enter some review text to analyze.")
    else:
        # Vectorize input
        review_vec = vectorizer.transform([user_review])
        
        # Prediction & Probability
        prediction = model.predict(review_vec)[0]
        probabilities = model.predict_proba(review_vec)[0] # [Prob_Genuine, Prob_Fake]
        
        fake_prob = probabilities[1]
        genuine_prob = probabilities[0]
        
        # Calculate Advanced Metrics (Phase 9)
        confidence = max(probabilities) * 100
        trust_score = round(genuine_prob * 100, 2)
        spam_score = round(fake_prob * 100, 2)
        
        # Layout Results
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status = "🚨 FAKE REVIEW" if prediction == 1 else "✅ GENUINE REVIEW"
            st.metric(label="Prediction Result", value=status)
        with col2:
            st.metric(label="Confidence Score", value=f"{confidence:.2f}%")
        with col3:
            st.metric(label="Trust Score (0-100)", value=f"{trust_score}/100")
        with col4:
            st.metric(label="Spam Risk Score", value=f"{spam_score}%")
            
        st.markdown("---")
        
        # Detailed Analysis Section
        st.subheader("📊 Detailed Trust & Probability Breakdown")
        chart_data = pd.DataFrame({
            "Category": ["Genuine Probability", "Fake Probability"],
            "Score": [genuine_prob, fake_prob]
        })
        st.bar_chart(chart_data.set_index("Category"))
        
        # Explainable AI Section (SHAP)
        st.subheader("🔍 Explainable AI (SHAP Word Impact Analysis)")
        st.info("The chart below displays how specific words in the review pushed the prediction towards Fake or Genuine.")
        
        shap_vals = explainer(review_vec)
        
        # Display top impacting words table
        feature_names = vectorizer.get_feature_names_out()
        row_shap = shap_vals.values[0]
        top_indices = np.argsort(np.abs(row_shap))[::-1][:5]
        
        impact_data = []
        for idx in top_indices:
            word = feature_names[idx]
            impact = row_shap[idx]
            effect = "Pushed towards Fake" if impact > 0 else "Pushed towards Genuine"
            impact_data.append({"Word": word, "SHAP Impact Value": round(float(impact), 4), "Effect": effect})
            
        st.table(pd.DataFrame(impact_data))

else:
    st.info("💡 Enter a review in the sidebar and click **Analyze Review** to test the system.")