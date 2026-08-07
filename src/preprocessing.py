import os
import re
import string
import pandas as pd
import spacy
from src.config import PROCESSED_FILE_PATH

try:
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except OSError:
    print("📥 Downloading spaCy language model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def lemmatize_text(text: str) -> str:
    if not text:
        return ""
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_space]
    return " ".join(tokens)

def process_reviews_pipeline(input_file_path: str = PROCESSED_FILE_PATH) -> pd.DataFrame:
    print(f"🔄 Loading data from {input_file_path} for NLP preprocessing...")
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"❌ Processed file not found at {input_file_path}. Run data_pipeline.py first!")
        
    df = pd.read_csv(input_file_path)
    
    print("🧹 Step 1/2: Cleaning text...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    print("🌿 Step 2/2: Lemmatizing text...")
    df['lemmatized_text'] = df['cleaned_text'].apply(lemmatize_text)
    
    df = df[df['lemmatized_text'].str.strip() != ""]
    
    output_path = input_file_path.replace("cleaned_reviews.csv", "final_preprocessed_reviews.csv")
    df.to_csv(output_path, index=False)
    
    print(f"\n🎉 NLP Preprocessing Pipeline Completed Successfully!")
    print(f"📊 Final Cleaned Records: {df.shape[0]}")
    print(f"💾 Saved preprocessed data to: {output_path}")
    return df

if __name__ == "__main__":
    process_reviews_pipeline()