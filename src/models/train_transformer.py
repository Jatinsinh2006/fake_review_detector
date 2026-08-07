import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from src.config import PROCESSED_DATA_DIR, RANDOM_STATE, TEST_SIZE

def train_transformer_model():
    print("🚀 Starting Transformer (DistilBERT) Fine-Tuning Pipeline...")
    
    # 1. Load preprocessed data
    preprocessed_path = os.path.join(PROCESSED_DATA_DIR, "final_preprocessed_reviews.csv")
    if not os.path.exists(preprocessed_path):
        raise FileNotFoundError(f"❌ Preprocessed file not found at {preprocessed_path}.")
        
    print(f"🔄 Loading data from {preprocessed_path}...")
    df = pd.read_csv(preprocessed_path)
    df['lemmatized_text'] = df['lemmatized_text'].fillna("")
    
    # Extract text and labels first
    X = df['lemmatized_text'].astype(str).tolist()
    y = df['label'].astype(int).tolist()
    
    # Take an optimized subset for fast CPU transformer fine-tuning (10,000 samples)
    if len(X) > 10000:
        X, _, y, _ = train_test_split(X, y, train_size=10000, random_state=RANDOM_STATE, stratify=y)
        print(f"⚡ Using optimized subset of {len(X)} samples for fast training.")
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print("🤖 Loading DistilBERT Tokenizer and Model...")
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)
    
    # Tokenize data
    print("✂️ Tokenizing datasets for Hugging Face Trainer...")
    train_encodings = tokenizer(X_train, truncation=True, padding=True, max_length=128)
    test_encodings = tokenizer(X_test, truncation=True, padding=True, max_length=128)
    
    # Create Hugging Face Dataset objects
    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'label': y_train
    })
    
    test_dataset = Dataset.from_dict({
        'input_ids': test_encodings['input_ids'],
        'attention_mask': test_encodings['attention_mask'],
        'label': y_test
    })
    
    model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    # Metrics computation function for Trainer
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        acc = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )
    
    print("\n🏋️ Fine-Tuning DistilBERT Model on 10k subset...")
    trainer.train()
    
    print("\n📊 Evaluating DistilBERT on Test Set...")
    eval_results = trainer.evaluate()
    print("Results:", eval_results)
    
    # Save the fine-tuned model and tokenizer
    model_save_path = os.path.join(PROCESSED_DATA_DIR, "best_transformer_model")
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    print(f"\n💾 Transformer model saved successfully to: {model_save_path}")

if __name__ == "__main__":
    train_transformer_model()