# 1_process_data.py
import pandas as pd
import re
from underthesea import word_tokenize
import config


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = word_tokenize(text, format="text")
    words = text.split()
    words = [w for w in words if w not in config.STOPWORDS]
    return " ".join(words)


def run():
    print("STEP 1: PROCESSING DATA...")
    try:
        df = pd.read_csv(config.DATA_PATH, encoding="utf-8")
        print(f"   - Loaded {len(df)} rows of raw data.")
    except Exception as e:
        print(f"Error: {e}")
        return

    # Apply cleaning
    print("   - Tokenizing and removing stopwords...")
    df["clean_text"] = df[config.COL_TEXT].apply(clean_text)

    # Save processed data for the next step
    df.to_csv(config.PROCESSED_DATA_PATH, index=False, encoding="utf-8")


if __name__ == "__main__":
    run()
