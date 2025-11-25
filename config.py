import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

DATA_PATH = os.path.join(DATA_DIR, "data.csv")
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_data.csv")

# Output Models
MODEL_PATH = os.path.join(MODEL_DIR, "svm_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

# Output Results
CONFUSION_MATRIX_IMG = os.path.join(RESULT_DIR, "confusion_matrix.png")
LATEX_REPORT = os.path.join(RESULT_DIR, "report_results.tex")
TEST_DATA_TEMP = os.path.join(DATA_DIR, "test_data_temp.csv")

# --- NLP CONFIG ---
COL_TEXT = "comment"
COL_LABEL = "label"
STOPWORDS = {
    "thì",
    "là",
    "mà",
    "bị",
    "của",
    "những",
    "cái",
    "việc",
    "ạ",
    "nhé",
    "vâng",
    "dạ",
}
