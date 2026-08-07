import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, GRU
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.config import PROCESSED_DATA_DIR, RANDOM_STATE, TEST_SIZE

def train_deep_learning_models():
    print("🚀 Starting Deep Learning Training Pipeline (LSTM / Bi-LSTM / GRU)...")
    
    # 1. Load preprocessed data
    preprocessed_path = os.path.join(PROCESSED_DATA_DIR, "final_preprocessed_reviews.csv")
    if not os.path.exists(preprocessed_path):
        raise FileNotFoundError(f"❌ Preprocessed file not found at {preprocessed_path}.")
        
    print(f"🔄 Loading data from {preprocessed_path}...")
    df = pd.read_csv(preprocessed_path)
    df['lemmatized_text'] = df['lemmatized_text'].fillna("")
    
    # 2. Train-Test Split
    print("✂️ Splitting data into Train and Test sets...")
    X = df['lemmatized_text'].astype(str).values
    y = df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    # 3. Tokenization & Padding hyperparameters
    vocab_size = 15000
    max_length = 150
    embedding_dim = 64
    
    print("🔢 Tokenizing and padding text sequences...")
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post', truncating='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post', truncating='post')
    
    # 4. Define Architectures to Compare
    def build_model(model_type="bilstm"):
        model = Sequential()
        model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))
        
        if model_type == "lstm":
            model.add(LSTM(64, return_sequences=False))
        elif model_type == "bilstm":
            model.add(Bidirectional(LSTM(64, return_sequences=False)))
        elif model_type == "gru":
            model.add(GRU(64, return_sequences=False))
            
        model.add(Dropout(0.5))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    architectures = {
        "LSTM": "lstm",
        "Bi-LSTM": "bilstm",
        "GRU": "gru"
    }
    
    dl_results = []
    
    # 5. Train and Evaluate each architecture
    for name, arch_type in architectures.items():
        print(f"\n----------------------------------------")
        print(f"🔹 Training Deep Learning Model: {name}...")
        
        model = build_model(arch_type)
        
        # Early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=2, restore_best_weights=True
        )
        
        history = model.fit(
            X_train_pad, y_train,
            epochs=5,
            batch_size=64,
            validation_split=0.1,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Predictions
        y_pred_probs = model.predict(X_test_pad)
        y_pred = (y_pred_probs >= 0.5).astype(int).flatten()
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        print(f"📊 {name} Performance -> Accuracy: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")
        
        dl_results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        })
        
    # 6. Summary Report
    results_df = pd.DataFrame(dl_results)
    print("\n🏆 --- DEEP LEARNING MODEL PERFORMANCE SUMMARY ---")
    print(results_df.to_string(index=False))
    
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    results_df.to_csv(os.path.join(PROCESSED_DATA_DIR, "dl_model_comparison_results.csv"), index=False)
    print("\n💾 Deep Learning evaluation results saved successfully!")

if __name__ == "__main__":
    train_deep_learning_models()

