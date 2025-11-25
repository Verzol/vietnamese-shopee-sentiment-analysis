# 3_evaluate.py
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import config


def run():
    print("STEP 3: EVALUATION & REPORT GENERATION...")

    # Load everything again
    try:
        model = joblib.load(config.MODEL_PATH)
        vectorizer = joblib.load(config.VECTORIZER_PATH)
        test_data = pd.read_csv("test_data_temp.csv")
        test_data = test_data.dropna(subset=["clean_text"])
    except:
        print("Missing model or test data files. Please run file 2 first!")
        return

    X_test = test_data["clean_text"]
    y_test = test_data["label"]

    # Predict
    print("   - Scoring on Test set...")
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)

    # 1. Draw Confusion Matrix and save image
    conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        conf_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=model.classes_,
        yticklabels=model.classes_,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.savefig("report_confusion_matrix.png")
    print("Saved image: report_confusion_matrix.png")

    # 2. Create report as DataFrame
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    # Format numbers nicely (round to 2 decimal places)
    report_df = report_df.round(2)

    # Print preview to screen
    print("\n--- RESULTS ---")
    print(report_df)

    # 3. EXPORT TO LATEX FILE
    latex_code = report_df.to_latex(
        caption="Experimental results of SVM model on Test set",
        label="tab:svm_results",
        column_format="|l|c|c|c|c|",
        position="h",
    )

    # Save to .tex file
    with open("report_results.tex", "w", encoding="utf-8") as f:
        f.write(latex_code)

    print("\n✅ LaTeX file exported: report_results.tex")
    print("   (You can open this file and copy it into Overleaf/Word)")


if __name__ == "__main__":
    run()
