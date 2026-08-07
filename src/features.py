import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer
from src.config import PROCESSED_DATA_DIR

def extract_bow_features(train_texts, test_texts, max_features=5000):
    """
    Extracts Bag of Words (CountVectorizer) features.
    """
    print("Generating Bag of Words features...")
    vectorizer = CountVectorizer(max_features=max_features)
    X_train_bow = vectorizer.fit_transform(train_texts)
    X_test_bow = vectorizer.transform(test_texts)

    joblib.dump(vectorizer, os.path.join(PROCESSED_DATA_DIR, "bow_vectorizer.pkl"))
    return X_train_bow, X_test_bow

def extract_tfidf_features(train_texts, test_texts, max_features=5000):
    """
    Extracts TF-IDF features.
    """

    print("Generating TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(train_texts)
    X_test_tfidf = vectorizer.transform(test_texts)

    joblib.dump(vectorizer, os.path.join(PROCESSED_DATA_DIR, "tfidf_vectorizer.pkl"))
    return X_train_tfidf, X_test_tfidf

def extract_transformer_embeddings(texts):
    """
    Extracts dense semantic embeddings using Sentence Transformers.
    """
    print("Generating Sentence Transformer embedings (MiniLM)... This may take a moment...")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts.tolist(), show_progress_bar=True)
    return embeddings

if __name__ == "__main__":

    preprocessed_path = os.path.join(PROCESSED_DATA_DIR, "final_preprocessed_reviews.csv")

    if not os.path.exists(preprocessed_path):
        raise FileNotFoundError(f"❌ Preprocessed file not found at {preprocessed_path}. Run preprocessing.py first!")

    print(f"🔄 Loading preprocessed data from {preprocessed_path}...")
    df = pd.read_csv(preprocessed_path)

    df['lemmatized_text'] = df['lemmatized_text'].fillna("")

    train_df = df.sample(frac=0.8, random_state=42)
    test_df = df.drop(train_df.index)

    X_train_tfidf, X_test_tfidf = extract_tfidf_features(train_df['lemmatized_text'], test_df['lemmatized_text'])
    print(f"TF-IDF Train Shape: {X_train_tfidf.shape}, Test Shape: {X_test_tfidf.shape}")


    

