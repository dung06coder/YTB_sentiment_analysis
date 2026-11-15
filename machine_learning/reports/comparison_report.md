# BÁO CÁO THỰC NGHIỆM VÀ SO SÁNH HIỆU NĂNG MÔ HÌNH
**Đề tài:** Hệ thống phân tích cảm xúc bình luận YouTube tiếng Việt  
**Người thực hiện:** Hùng (Phụ trách Deep Learning & PhoBERT) & Vitou (Data & Baseline ML)  
**Ngày báo cáo:** 15/11/2026  

---

## 1. Tóm tắt kết quả cốt lõi (Executive Summary)

- **Mô hình PhoBERT fine-tuned đạt hiệu năng vượt trội toàn diện** so với toàn bộ các mô hình học máy truyền thống (LinearSVC, Multinomial Naive Bayes, Logistic Regression).
- **Độ chính xác (Accuracy):** PhoBERT đạt **79.16%**, tăng **+4.74%** so với mô hình baseline tốt nhất của nhóm (LinearSVC: 74.42%).
- **Macro F1-score:** PhoBERT đạt **79.01%**, tăng **+4.61%** so với LinearSVC (74.40%).
- **Cải thiện bước ngoặt ở lớp Trung lập (Neutral):** Lớp trung lập vốn là bài toán nan giải đối với mô hình Bag-of-Words / TF-IDF do các bình luận trung tính không chứa nhiều từ cảm xúc rõ ràng (F1 của SVM chỉ đạt 69.88%). PhoBERT đã nâng F1 lớp này lên **74.27%** (tăng **+4.39%**).

---

## 2. Bảng tổng hợp đối sánh chỉ số (Test Set: 5.418 bình luận)

| Mô hình | Accuracy | Macro F1 | Weighted F1 | F1 (Tiêu cực) | F1 (Trung lập) | F1 (Tích cực) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MultinomialNB** | 68.86% | 68.60% | 68.60% | 69.90% | 61.99% | 73.90% |
| **Logistic Regression (SGD)** | 74.31% | 74.28% | 74.28% | 74.31% | 70.41% | 78.11% |
| **LinearSVC (Baseline)** | 74.42% | 74.40% | 74.40% | 74.91% | 69.88% | 78.40% |
| **PhoBERT (Fine-tuned)** | **79.16%** | **79.01%** | **79.01%** | **80.34%** | **74.27%** | **82.43%** |

---

## 3. Phân tích nguyên nhân chênh lệch hiệu năng (Theoretical Insights)

1. **Khả năng nắm bắt ngữ cảnh hai chiều (Bidirectional Self-Attention):**
   - Mô hình truyền thống dựa trên vector hóa TF-IDF (dù đã kết hợp word + character n-grams) vẫn xem các từ ngữ là độc lập (Bag-of-Words assumption). Khi gặp các câu đảo ngữ, phủ định kép hoặc câu mỉa mai (ví dụ: *"phim hay ghê, xem mà ngủ quên luôn"*), TF-IDF bắt gặp từ "hay" và có xu hướng đoán nhầm sang Tích cực.
   - PhoBERT sử dụng kiến trúc Transformer với cơ chế Multi-Head Self-Attention, liên kết toàn bộ từ trong câu để hiểu trọn vẹn ngữ nghĩa phủ định và ngữ cảnh châm biếm.

2. **Tiền huấn luyện chuyên sâu cho tiếng Việt (Pretrained Vietnamese Corpora):**
   - PhoBERT được VinAI Research tiền huấn luyện trên 20GB văn bản tiếng Việt (~3 tỷ tokens) sử dụng kỹ thuật RoBERTa masked language modeling. Do đó, mô hình đã nắm vững ngữ pháp, ngữ nghĩa và phong cách diễn đạt tiếng Việt trước khi fine-tuning.

3. **Sự kết hợp hoàn hảo với Pipeline tiền xử lý của nhóm:**
   - Việc chuyển đổi Emoji thành từ ngữ cảm xúc tiếng Việt (ví dụ: `😂` $\to$ `cười`, `😡` $\to$ `tức giận`) và chuẩn hóa teencode (`ko` $\to$ `không`, `dc` $\to$ `được`) của bạn Vitou giúp PhoBERT nhận diện các sắc thái biểu cảm đặc trưng trên YouTube một cách trọn vẹn nhất.

---

## 4. Các tệp đính kèm và bàn giao

- **Biểu đồ trực quan:**
  - `machine_learning/reports/comparison_metrics_barchart.png`: Biểu đồ cột tổng thể Accuracy, Macro-F1, Weighted-F1.
  - `machine_learning/reports/f1_per_class_comparison.png`: Biểu đồ cột so sánh chi tiết F1 từng lớp.
  - `machine_learning/reports/phobert_confusion_matrix.png`: Ma trận nhầm lẫn của PhoBERT.
- **Bảng dữ liệu LaTeX:** `machine_learning/reports/latex_comparison_table.tex` (đã sẵn sàng để bạn Vitou và Hùng chèn vào báo cáo LaTeX đồ án).
- **Code dự đoán (Backend API):** `machine_learning/predict_phobert.py` (cung cấp class `PhoBertSentimentPredictor` sẵn sàng cho Backend gọi suy luận).
