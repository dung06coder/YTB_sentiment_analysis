# HƯỚNG DẪN TRIỂN KHAI PHOBERT CHO BÀI TOÁN PHÂN TÍCH CẢM XÚC BÌNH LUẬN YOUTUBE
> **Thành viên phụ trách:** Bạn Hùng (Deep Learning & PhoBERT)  
> **Dự án:** `YTB_sentiment_analysis`  
> **Nhánh Git:** `feature/hung-phobert-dl`  

---

## 📌 1. Tổng quan nhiệm vụ của Hùng

1. **Tiếp nhận dữ liệu sạch từ bạn Vitou:**
   - Dữ liệu `train.csv`, `test.csv` (và `clean.csv`) được chia sẻ qua Google Drive chung.
   - Áp dụng hàm tiền xử lý `clean_text(text, segment=True)` từ `machine_learning/preprocessing.py` (loại bỏ URL, chuẩn hóa teencode, chuyển emoji thành từ tiếng Việt, và phân đoạn từ ghép với dấu gạch dưới `_` để tối ưu cho PhoBERT).
2. **Cấu hình PhoBERT Tokenizer & Fine-tune PhoBERT:**
   - Sử dụng mô hình nền tảng `vinai/phobert-base` từ VinAI Research.
   - Tokenizer: PhoBERT Tokenizer với `max_length=128` (hoặc 256).
   - Fine-tuning bằng PyTorch và Hugging Face Transformers (`train_phobert.py`).
3. **Đánh giá & Vẽ biểu đồ so sánh với Baseline SVM của Vitou:**
   - Chạy `evaluate_and_compare.py` để xuất các biểu đồ trực quan và mã nguồn bảng LaTeX (`latex_comparison_table.tex`) phục vụ viết báo cáo đồ án.
4. **Đóng gói và bàn giao cho Backend:**
   - Cung cấp class `PhoBertSentimentPredictor` (`predict_phobert.py`) để Backend nạp và suy luận nhanh chóng cho 500 comment YouTube crawl về.
5. **Tuân thủ quy tắc Git:**
   - **TUYỆT ĐỐI KHÔNG ĐẨY FILE TRỌNG SỐ (.pt, .bin, .safetensors) LÊN GITHUB.**
   - Đẩy code lên nhánh riêng `feature/hung-phobert-dl` và lưu file weights lên Google Drive chung.

---

## 📁 2. Cấu trúc thư mục mã nguồn PhoBERT

```
machine_learning/
├── preprocessing.py               # Tiền xử lý văn bản tiếng Việt của Vitou
├── requirements.txt               # Danh sách thư viện (PyTorch, Transformers, Underthesea, Pyvi...)
├── phobert/
│   ├── __init__.py
│   ├── dataset.py                 # PyTorch Dataset & DataLoader
│   ├── model.py                   # Kiến trúc PhoBertSentimentClassifier
│   └── utils.py                   # Tiện ích metrics, seed, device, heatmap
├── train_phobert.py               # Script huấn luyện PhoBERT (hỗ trợ CUDA FP16 & CPU)
├── predict_phobert.py             # Script suy luận đơn lẻ & batch CSV cho Backend
├── evaluate_and_compare.py        # Đánh giá đối sánh hiệu năng, vẽ biểu đồ, xuất LaTeX
├── notebooks/
│   └── PhoBERT_FineTuning_Colab.ipynb # Notebook chạy trực tiếp trên GPU Google Colab
├── models/
│   └── phobert_finetuned/         # Thư mục lưu checkpoint mô hình (được .gitignore bảo vệ)
└── reports/                       # Thư mục chứa báo cáo & biểu đồ
    ├── comparison_metrics_barchart.png
    ├── f1_per_class_comparison.png
    ├── phobert_confusion_matrix.png
    ├── model_comparison_extended.csv
    ├── latex_comparison_table.tex
    └── comparison_report.md
```

---

## 🚀 3. Hướng dẫn chạy

### Cách 1: Chạy nhanh trên máy local (Windows / PowerShell)

#### Bước 3.1: Cài đặt thư viện
```powershell
pip install -r machine_learning/requirements.txt
```

#### Bước 3.2: Tải dữ liệu từ Google Drive chung
Tải 2 file `train.csv` và `test.csv` từ link Google Drive chung của nhóm vào thư mục:
`machine_learning/data/` (thư mục này đã được cấu hình trong `.gitignore`).

#### Bước 3.3: Tiến hành Fine-tune PhoBERT
```powershell
python machine_learning/train_phobert.py `
    --train machine_learning/data/train.csv `
    --test machine_learning/data/test.csv `
    --epochs 4 `
    --batch-size 16 `
    --lr 2e-5 `
    --fp16
```
*(Nếu máy chạy CPU, bỏ cờ `--fp16` và giảm `--batch-size 8`)*.

#### Bước 3.4: Đánh giá và xuất biểu đồ so sánh với SVM
```powershell
python machine_learning/evaluate_and_compare.py
```
Các file kết quả sẽ được tự động sinh ra trong `machine_learning/reports/`.

#### Bước 3.5: Kiểm thử suy luận
```powershell
# Dự đoán 1 bình luận:
python machine_learning/predict_phobert.py --text "Video giải thích thuật toán trực quan và dễ hiểu quá, cảm ơn kênh! 😍"

# Dự đoán file CSV:
python machine_learning/predict_phobert.py --csv machine_learning/data/test.csv
```

---

### Cách 2: Huấn luyện trên Google Colab GPU (Khuyến nghị cho huấn luyện nhanh)

1. Mở [Google Colab](https://colab.research.google.com).
2. Tải lên tệp notebook: `machine_learning/notebooks/PhoBERT_FineTuning_Colab.ipynb`.
3. Bật GPU: Menu **Runtime** $\to$ **Change runtime type** $\to$ Chọn **T4 GPU**.
4. Chạy tuần tự các ô trong Notebook. Quá trình huấn luyện trên GPU T4 chỉ mất khoảng 10 - 15 phút.
5. Sau khi huấn luyện xong, notebook sẽ tự động nén `phobert_finetuned.zip` và lưu vào thư mục Google Drive chung của nhóm.

---

## 🤝 4. Hướng dẫn tích hợp cho bạn Backend (Giai đoạn 3)

Bạn làm Backend có thể import và sử dụng class `PhoBertSentimentPredictor` cực kỳ đơn giản:

```python
from machine_learning.predict_phobert import PhoBertSentimentPredictor

# 1. Khởi tạo predictor (chỉ nạp 1 lần khi server backend khởi động)
predictor = PhoBertSentimentPredictor(model_dir="machine_learning/models/phobert_finetuned")

# 2. Dự đoán 1 bình luận đơn:
result = predictor.predict_comment("Video này quá hay, ad đỉnh nóc kịch trần!")
print(result)
# Output:
# {
#     'text': 'Video này quá hay, ad đỉnh nóc kịch trần!',
#     'clean_text': 'video này quá hay ad đỉnh nóc kịch trần',
#     'label': 'positive',
#     'confidence': 0.9654,
#     'probabilities': {
#         'negative': 0.0121,
#         'neutral': 0.0225,
#         'positive': 0.9654
#     }
# }

# 3. Dự đoán hàng loạt 500 bình luận vừa crawl từ YouTube:
comments = ["Cmt 1", "Cmt 2", "Cmt 3", ...]
batch_results = predictor.predict_batch(comments, batch_size=32)
```

---

## 📊 5. Kết quả so sánh hiệu năng (Đưa vào đồ án)

| Mô hình | Accuracy | Macro F1 | Weighted F1 | F1 (Tiêu cực) | F1 (Trung lập) | F1 (Tích cực) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MultinomialNB** | 68.86% | 68.60% | 68.60% | 69.90% | 61.99% | 73.90% |
| **Logistic Regression (SGD)** | 74.31% | 74.28% | 74.28% | 74.31% | 70.41% | 78.11% |
| **LinearSVC (Baseline)** | 74.42% | 74.40% | 74.40% | 74.91% | 69.88% | 78.40% |
| **PhoBERT (Fine-tuned)** | **86.53%** | **86.47%** | **86.47%** | **86.50%** | **83.56%** | **89.60%** |

- **Điểm nổi bật:** PhoBERT vượt trội hoàn toàn so với Baseline LinearSVC (Accuracy tăng **+12.11%**, Macro-F1 tăng **+12.07%**).
- Mã nguồn bảng LaTeX cho đồ án đã sẵn sàng tại: `machine_learning/reports/latex_comparison_table.tex`.

---

## 🌿 6. Quy trình Git & Đẩy code lên GitHub

```powershell
# 1. Kiểm tra trạng thái git (đảm bảo KHÔNG có file .pt, .bin, .safetensors nào được thêm)
git status

# 2. Thêm code vào git
git add machine_learning/
git add .gitignore

# 3. Tạo commit rõ ràng, chuẩn semantic commit
git commit -m "feat(phobert): them pipeline fine-tune PhoBERT, script danh gia so sanh SVM va bao cao LaTeX"

# 4. Đẩy lên nhánh riêng của Hùng
git push origin feature/hung-phobert-dl
```
Sau đó, Hùng lên giao diện GitHub tạo **Pull Request (PR)** từ `feature/hung-phobert-dl` vào `main` để nhóm trưởng duyệt và merge!
