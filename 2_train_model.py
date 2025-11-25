# 2_train_model.py
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import config


def run():
    print("STEP 2: TRAINING MODEL...")

    # Load processed data from step 1
    try:
        df = pd.read_csv(config.PROCESSED_DATA_PATH)
        # Drop rows with null values in 'clean_text' column (if any)
        df = df.dropna(subset=["clean_text"])
    except:
        print("Processed data file not found. Please run 1_process_data.py first!")
        return

    # Split dataset
    print("   - Splitting Train/Test set (80-20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df[config.COL_LABEL],
        test_size=0.2,
        random_state=42,
        stratify=df[config.COL_LABEL],
    )

    # Vectorize
    print("   - Creating TF-IDF Vector...")
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)

    # Train Model
    print("   - Training SVM (Machine Learning)...")
    model = SVC(kernel="linear")
    model.fit(X_train_tfidf, y_train)

    # Save Model and Vectorizer to files (for step 3)
    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(vectorizer, config.VECTORIZER_PATH)

    # Save test set for fair evaluation in step 3
    test_data = pd.DataFrame({"clean_text": X_test, "label": y_test})
    test_data.to_csv("test_data_temp.csv", index=False)

    print(f"Training completed! Model saved to {config.MODEL_PATH}")


if __name__ == "__main__":
    run()
