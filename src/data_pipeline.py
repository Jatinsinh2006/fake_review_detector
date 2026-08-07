import os
import pandas as pd
from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, PROCESSED_FILE_PATH

def load_and_preprocess_deceptive_spam() -> pd.DataFrame:
    path = os.path.join(RAW_DATA_DIR, "deceptive_spam.csv")
    if not os.path.exists(path):
        print(f"⚠️ Warning: {path} not found. Skipping.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    processed_df = pd.DataFrame()
    processed_df['text'] = df['text']
    processed_df['rating'] = None
    processed_df['label'] = df['deceptive'].apply(lambda x: 1 if str(x).lower() == 'deceptive' else 0)
    processed_df['source'] = 'deceptive_spam'
    return processed_df

def load_and_preprocess_fake_reviews() -> pd.DataFrame:
    path = os.path.join(RAW_DATA_DIR, "fake_reviews.csv")
    if not os.path.exists(path):
        print(f"⚠️ Warning: {path} not found. Skipping.")
        return pd.DataFrame()
        
    df = pd.read_csv(path)
    processed_df = pd.DataFrame()
    processed_df['text'] = df['text_'] if 'text_' in df.columns else df['text']
    processed_df['rating'] = df['rating']
    processed_df['label'] = df['label'].apply(lambda x: 1 if str(x).strip().upper() == 'CG' else 0)
    processed_df['source'] = 'fake_reviews_dataset'
    return processed_df

def load_and_preprocess_amazon_json() -> pd.DataFrame:
    path = os.path.join(RAW_DATA_DIR, "amazon_electronics.json")
    if not os.path.exists(path):
        print(f"⚠️ Warning: {path} not found. Skipping.")
        return pd.DataFrame()

    print("🔄 Reading Amazon JSON file (this may take a moment)...")
    df = pd.read_json(path, lines=True, nrows=50000)

    processed_df = pd.DataFrame()
    processed_df['text'] = df['reviewText'] if 'reviewText' in df.columns else df.get('text', '')
    processed_df['rating'] = df['overall'] if 'overall' in df.columns else None
    processed_df['label'] = 0
    processed_df['source'] = 'amazon_electronics'
    return processed_df

def build_master_dataset():
    print("🚀 Starting Multi-Dataset Ingestion Pipeline...")

    df_deceptive = load_and_preprocess_deceptive_spam()
    df_fake = load_and_preprocess_fake_reviews()
    df_amazon = load_and_preprocess_amazon_json()

    frames = [df for df in [df_deceptive, df_fake, df_amazon] if not df.empty]

    if not frames:
        raise ValueError("❌ No datasets found in 'data/raw/'!")

    master_df = pd.concat(frames, ignore_index=True)
    master_df = master_df.dropna(subset=['text'])
    master_df['text'] = master_df['text'].astype(str)
    master_df = master_df.drop_duplicates(subset=['text'])

    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    master_df.to_csv(PROCESSED_FILE_PATH, index=False)

    print(f"\n🎉 Master Dataset Built Successfully!")
    print(f"📊 Total Combined Records: {master_df.shape[0]}")
    print(f"📈 Label Distribution:\n{master_df['label'].value_counts()}")
    print(f"💾 Saved to: {PROCESSED_FILE_PATH}")

if __name__ == "__main__":
    build_master_dataset()