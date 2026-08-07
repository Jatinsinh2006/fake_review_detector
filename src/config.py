import os 
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = os.path.join(ROOT_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

RAW_FILE_PATH = os.path.join(RAW_DATA_DIR, "raw_reviews.csv")
PROCESSED_FILE_PATH = os.path.join(PROCESSED_DATA_DIR, "cleaned_reviews.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES = 5000