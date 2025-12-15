# Giải Thích Chi Tiết Dự Án: Vietnamese Shopee Sentiment Analysis

## 📌 Tổng Quan

Dự án này xây dựng một hệ thống **Phân tích Cảm xúc (Sentiment Analysis)** cho các bình luận tiếng Việt trên sàn thương mại điện tử Shopee. Mục tiêu là tự động phân loại bình luận của khách hàng thành các nhãn cảm xúc: **Tích cực (POS)**, **Tiêu cực (NEG)**, hoặc **Trung tính (NEU)**.

**Công nghệ sử dụng:**

- **Ngôn ngữ:** Python
- **Thư viện NLP:** `underthesea` (tách từ tiếng Việt)
- **Machine Learning:** `scikit-learn` (SVM - Support Vector Machine)
- **Trực quan hóa:** `matplotlib`, `seaborn`

---

## 📁 Cấu Trúc Thư Mục

```
vietnamese-shopee-sentiment-analysis/
├── config.py                 # Cấu hình đường dẫn và tham số
├── 1_process_data.py         # Bước 1: Tiền xử lý dữ liệu
├── 2_train_model.py          # Bước 2: Huấn luyện mô hình
├── 3_evaluate.py             # Bước 3: Đánh giá và xuất báo cáo
├── dataset/
│   ├── data.csv              # Dữ liệu thô (comment + label)
│   └── processed_data.csv    # Dữ liệu đã xử lý (clean_text + label)
├── models/
│   ├── svm_model.pkl         # Mô hình SVM đã huấn luyện
│   └── tfidf_vectorizer.pkl  # Bộ vector hóa TF-IDF
├── results/
│   ├── confusion_matrix.png  # Hình ảnh ma trận nhầm lẫn
│   └── report_results.tex    # Báo cáo kết quả dạng LaTeX
├── docs/                     # Tài liệu
├── requirements.txt          # Danh sách thư viện cần cài
└── README.md                 # Hướng dẫn sử dụng
```

---

## 🔧 Giải Thích Từng File

### 1. `config.py` - File Cấu Hình

File này chứa tất cả các **đường dẫn** và **tham số** được sử dụng xuyên suốt dự án.

```python
# Các thư mục chính
DATA_DIR = "dataset/"
MODEL_DIR = "models/"
RESULT_DIR = "results/"

# Đường dẫn file
DATA_PATH = "dataset/data.csv"              # Dữ liệu gốc
PROCESSED_DATA_PATH = "dataset/processed_data.csv"  # Dữ liệu đã xử lý
MODEL_PATH = "models/svm_model.pkl"         # Mô hình đã train
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"     # Bộ TF-IDF

# Cấu hình NLP
COL_TEXT = "comment"    # Tên cột chứa bình luận
COL_LABEL = "label"     # Tên cột chứa nhãn (POS/NEG/NEU)

# Danh sách Stopwords (từ dừng - không mang ý nghĩa)
STOPWORDS = {"thì", "là", "mà", "bị", "của", "những", "cái", "việc", "ạ", "nhé", "vâng", "dạ"}
```

**Tại sao cần file này?**

- Dễ dàng thay đổi đường dẫn hoặc tham số mà không cần sửa nhiều file.
- Code sạch hơn, dễ bảo trì hơn.

---

### 2. `1_process_data.py` - Tiền Xử Lý Dữ Liệu

**Mục đích:** Làm sạch và chuẩn hóa văn bản tiếng Việt trước khi đưa vào mô hình.

#### Quy trình xử lý:

```
Dữ liệu thô (data.csv)
        ↓
    Chữ thường (lowercase)
        ↓
    Xóa ký tự đặc biệt (!@#$%...)
        ↓
    Tách từ tiếng Việt (underthesea)
        ↓
    Loại bỏ stopwords
        ↓
Dữ liệu sạch (processed_data.csv)
```

#### Code quan trọng:

```python
def clean_text(text):
    # 1. Chuyển thành chữ thường
    text = text.lower()

    # 2. Xóa ký tự đặc biệt, chỉ giữ chữ và số
    text = re.sub(r"[^\w\s]", "", text)

    # 3. Tách từ tiếng Việt (VD: "chất lượng" → "chất_lượng")
    text = word_tokenize(text, format="text")

    # 4. Loại bỏ stopwords
    words = text.split()
    words = [w for w in words if w not in config.STOPWORDS]

    return " ".join(words)
```

**Ví dụ:**
| Input (comment) | Output (clean_text) |
|-----------------|---------------------|
| "Hàng đẹp lắm ạ, giao nhanh nhé!" | "hàng đẹp lắm giao nhanh" |
| "Chất lượng kém, thất vọng quá!" | "chất_lượng kém thất_vọng quá" |

**Tại sao cần tách từ tiếng Việt?**

- Tiếng Việt có nhiều từ ghép (VD: "chất lượng", "giao hàng"). Nếu không tách từ, mô hình sẽ hiểu sai.
- Thư viện `underthesea` giúp tách từ chính xác, nối các từ ghép bằng dấu `_`.

---

### 3. `2_train_model.py` - Huấn Luyện Mô Hình

**Mục đích:** Sử dụng dữ liệu đã xử lý để huấn luyện mô hình Machine Learning.

#### Quy trình:

```
Dữ liệu sạch (processed_data.csv)
        ↓
    Chia Train/Test (80% - 20%)
        ↓
    Vector hóa bằng TF-IDF
        ↓
    Huấn luyện mô hình SVM
        ↓
    Lưu mô hình (svm_model.pkl)
```

#### Code quan trọng:

```python
# 1. Chia dữ liệu thành tập Train và Test
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"],
    df["label"],
    test_size=0.2,      # 20% dữ liệu dùng để test
    random_state=42,    # Đảm bảo kết quả tái lập được
    stratify=df["label"]  # Giữ tỷ lệ nhãn giống nhau ở train và test
)

# 2. Chuyển văn bản thành vector số (TF-IDF)
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)

# 3. Huấn luyện mô hình SVM
model = SVC(kernel="linear")
model.fit(X_train_tfidf, y_train)

# 4. Lưu mô hình để sử dụng sau
joblib.dump(model, "models/svm_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
```

**Giải thích TF-IDF:**

- **TF (Term Frequency):** Từ xuất hiện nhiều trong 1 văn bản → điểm cao.
- **IDF (Inverse Document Frequency):** Từ xuất hiện ở ít văn bản → điểm cao (từ đặc trưng).
- Kết hợp: Từ "đẹp" xuất hiện nhiều trong 1 review nhưng không phổ biến ở tất cả review → điểm cao.

**Tại sao chọn SVM?**

- SVM hoạt động tốt với dữ liệu văn bản nhiều chiều.
- Phù hợp với bài toán phân loại nhị phân/đa lớp.
- Tốc độ nhanh, dễ triển khai.

---

### 4. `3_evaluate.py` - Đánh Giá Mô Hình

**Mục đích:** Kiểm tra độ chính xác của mô hình và xuất báo cáo.

#### Quy trình:

```
Load mô hình đã train
        ↓
    Dự đoán trên tập Test
        ↓
    Tính các chỉ số đánh giá
        ↓
    Vẽ Confusion Matrix
        ↓
    Xuất báo cáo LaTeX
```

#### Các chỉ số đánh giá:

| Chỉ số        | Ý nghĩa                                                          |
| ------------- | ---------------------------------------------------------------- |
| **Accuracy**  | Tỷ lệ dự đoán đúng trên tổng số mẫu                              |
| **Precision** | Trong những mẫu dự đoán là X, bao nhiêu % thực sự là X?          |
| **Recall**    | Trong tổng số mẫu thực sự là X, mô hình tìm ra được bao nhiêu %? |
| **F1-Score**  | Trung bình điều hòa của Precision và Recall                      |

#### Code quan trọng:

```python
# 1. Dự đoán
y_pred = model.predict(X_test_tfidf)

# 2. In báo cáo chi tiết
print(classification_report(y_test, y_pred))

# 3. Vẽ Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues")
plt.savefig("report_confusion_matrix.png")

# 4. Xuất báo cáo LaTeX (để đưa vào báo cáo/luận văn)
report_df.to_latex("report_results.tex")
```

**Confusion Matrix là gì?**

- Ma trận thể hiện số lượng dự đoán đúng/sai cho từng nhãn.
- Giúp xác định mô hình hay nhầm lẫn giữa những nhãn nào.

---

## 🚀 Cách Chạy Dự Án

### Bước 1: Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### Bước 2: Chạy lần lượt các script

```bash
python 1_process_data.py   # Tiền xử lý dữ liệu
python 2_train_model.py    # Huấn luyện mô hình
python 3_evaluate.py       # Đánh giá và xuất báo cáo
```

### Kết quả đầu ra:

- `models/svm_model.pkl`: Mô hình đã huấn luyện
- `results/confusion_matrix.png`: Hình ảnh ma trận nhầm lẫn
- `results/report_results.tex`: Bảng kết quả dạng LaTeX

---

## 📊 Luồng Dữ Liệu Tổng Quan

```
┌─────────────────┐
│   data.csv      │  ← Dữ liệu thô (comment, label)
│  (Raw Data)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 1_process_data  │  ← Làm sạch, tách từ, bỏ stopwords
│   .py           │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ processed_data  │  ← Dữ liệu sạch (clean_text, label)
│     .csv        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2_train_model   │  ← TF-IDF + SVM Training
│     .py         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  svm_model.pkl  │  ← Mô hình đã huấn luyện
│ tfidf_vec.pkl   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  3_evaluate.py  │  ← Dự đoán + Đánh giá
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ confusion_matrix│  ← Kết quả trực quan
│ report.tex      │  ← Bảng số liệu LaTeX
└─────────────────┘
```

---

## 💡 Điểm Nổi Bật Của Dự Án

1. **Pipeline rõ ràng:** Chia thành 3 bước riêng biệt, dễ debug và mở rộng.
2. **Hỗ trợ tiếng Việt:** Sử dụng `underthesea` để tách từ chính xác.
3. **Tái sử dụng được:** Mô hình và vectorizer được lưu lại, có thể dùng để dự đoán bình luận mới.
4. **Xuất báo cáo LaTeX:** Tiện lợi cho việc viết báo cáo/luận văn.

---

## 🔮 Hướng Phát Triển

- [ ] Thêm xử lý teencode (VD: "đc" → "được", "ko" → "không")
- [ ] Thử nghiệm các mô hình khác (Naive Bayes, Random Forest)
- [ ] Áp dụng Deep Learning (LSTM, PhoBERT)
- [ ] Xây dựng API để dự đoán realtime
- [ ] Tăng cường dữ liệu training

---
