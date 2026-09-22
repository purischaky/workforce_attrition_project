"""
src/make_dataset.py
Reproducible train/test split from the raw synthetic dataset.
Run with: python src/make_dataset.py
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/employees.csv"
PROCESSED_DIR = "data/processed"
TEST_SIZE = 0.20
RANDOM_STATE = 42

def build_train_test_split():
    print("=" * 60)
    print("BUILDING TRAIN/TEST SPLIT FROM RAW DATA")
    print("=" * 60)

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    df = pd.read_csv(RAW_PATH)
    print(f"✓ Loaded raw dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["attrition"],
    )

    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    test_path = os.path.join(PROCESSED_DIR, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"✓ Train set: {train_df.shape[0]} rows → {train_path}")
    print(f"✓ Test set:  {test_df.shape[0]} rows → {test_path}")
    print(f"✓ Stratified split on 'attrition' (test_size={TEST_SIZE}, random_state={RANDOM_STATE})")

if __name__ == "__main__":
    build_train_test_split()