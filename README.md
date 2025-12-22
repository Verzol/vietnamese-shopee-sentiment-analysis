# Vietnamese Shopee Sentiment Analysis

Dự án phân tích cảm xúc bình luận Shopee tiếng Việt.

## Cấu trúc dự án

- `1_process_data.py`: Xử lý dữ liệu thô, chuẩn hóa văn bản tiếng Việt.
- `2_train_model.py`: Huấn luyện mô hình phân loại cảm xúc.
- `3_evaluate.py`: Đánh giá hiệu suất mô hình.
- `config.py`: Các cấu hình và tham số chung.
- `dataset/`: Thư mục chứa dữ liệu.
- `models/`: Thư mục chứa mô hình đã huấn luyện.
- `results/`: Kết quả đánh giá và báo cáo.

## Cài đặt

Cài đặt các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

## Sử dụng

Chạy lần lượt các bước:

1. Xử lý dữ liệu:
   ```bash
   python 1_process_data.py
   ```
2. Huấn luyện mô hình:
   ```bash
   python 2_train_model.py
   ```
3. Đánh giá:
   ```bash
   python 3_evaluate.py
   ```

## Notes

Repo này được tạo ra với mục đích thực nghiệm cho môn **Khai phá dữ liệu**.
Vì đây là dự án học thuật, những góp ý và đóng góp qua [Issues](../../issues) hoặc Pull Request đều rất đáng quý và được hoan nghênh!
