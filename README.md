# YouTube Comment Sentiment Analysis (Vietnamese NLP)

Hệ thống ứng dụng Học máy (Machine Learning) và Học sâu (Deep Learning) để tự động thu thập bình luận từ một video YouTube bất kỳ, phân loại sắc thái cảm xúc tiếng Việt thành 3 nhãn (**Tích cực**, **Trung lập**, **Tiêu cực**) và trực quan hóa kết quả so sánh giữa hai mô hình **TF-IDF + SVM** và **PhoBERT**.

---

## 1. Giới thiệu sản phẩm

Khi người dùng dán đường dẫn (URL) một video YouTube vào giao diện web (hỗ trợ cả link video thường, `youtu.be`, `Shorts`, `Embed`, `Live`), hệ thống sẽ tự động:

1. **Thu thập dữ liệu thời gian thực:** Kết nối YouTube Data API v3 để lấy tối đa 500 bình luận cấp cao nhất mới nhất kèm siêu dữ liệu (tác giả, thời gian đăng, lượt thích).
2. **Phân tích đa mô hình trên cùng một tập bình luận:** Người dùng có thể chạy riêng lẻ hoặc tích chọn đồng thời **TF-IDF + SVM** và **PhoBERT** để đối chiếu trực tiếp kết quả phân loại.
3. **Trực quan hóa kết quả:**
   - Thẻ tổng quan hiển thị tổng số bình luận thu thập, độ tin cậy trung bình và thời gian phản hồi thực tế.
   - Biểu đồ cột so sánh phân bổ cảm xúc giữa hai mô hình và biểu đồ tròn chi tiết theo từng mô hình.
   - Bảng đối chiếu các chỉ số đánh giá mô hình ($Accuracy$, $Precision$, $Recall$, $F_1$-score theo từng lớp).
   - Danh sách bình luận tiêu biểu có bộ lọc theo cảm xúc kèm điểm độ tin cậy (_confidence_).

---

## 2. Kiến trúc hệ thống & Luồng hoạt động

```text
┌──────────────────────┐        POST /api/analyze         ┌──────────────────────────────┐
│   Frontend (React)   │  ──── { url, models } ────────►  │     Backend API (Flask)      │
│   Vite · Port 5173   │  ◄─── JSON results & metrics ──  │         Port 5000            │
└──────────────────────┘                                  └──────────────┬───────────────┘
                                                                         │
                                      ┌──────────────────────────────────┴──────────────────────────────────┐
                                      ▼                                                                     ▼
                     ┌────────────────────────────────┐                                   ┌─────────────────────────────────┐
                     │    YouTube Data API v3         │                                   │   Inference Service (in RAM)    │
                     │  (backend/youtube_crawler.py)  │                                   │  1. Preprocessing (clean_text)  │
                     │  Cào tối đa 500 bình luận thô  │                                   │  2. TF-IDF + SVM (.pkl)         │
                     └────────────────────────────────┘                                   │  3. PhoBERT Fine-tuned (PyTorch)│
                                                                                          └─────────────────────────────────┘
```

### Quy trình xử lý một yêu cầu phân tích

1. **Kiểm tra hợp lệ (Validation):** Frontend kiểm tra định dạng URL YouTube; Backend kiểm tra mô hình yêu cầu đã được nạp sẵn vào RAM hay chưa trước khi tiêu tốn quota của YouTube API.
2. **Thu thập (Crawling):** `youtube_crawler.py` tách `video_id` và gọi phân trang (`nextPageToken`) tới `commentThreads` để lấy tối đa 500 bình luận. Các trường hợp đặc biệt như video tắt bình luận (`COMMENTS_DISABLED`), video riêng tư/không tồn tại (`VIDEO_NOT_FOUND`), video 0 bình luận (`NO_COMMENTS`) hoặc hết quota (`QUOTA_EXCEEDED`) đều được trả mã lỗi chuẩn về giao diện.
3. **Tiền xử lý & Suy luận (Preprocessing & Inference):**
   - Bình luận thô đi qua hàm `clean_text` dùng chung từ `machine_learning/preprocessing.py` (loại bỏ URL, thẻ HTML, nhiễu; chuyển đổi emoji cảm xúc sang từ tiếng Việt; chuẩn hóa teencode/từ viết tắt).
   - Dữ liệu sạch được đẩy qua mô hình **TF-IDF + SVM** và/hoặc **PhoBERT** đã nạp sẵn trong RAM ngay khi khởi động server để tối ưu độ trễ.
4. **Tổng hợp đầu ra:** Backend tính tỷ lệ phần trăm từng nhãn, độ tin cậy trung bình, chọn tập bình luận mẫu cân bằng giữa 3 sắc thái, đính kèm bộ chỉ số đánh giá từ `machine_learning/reports/` và trả về JSON cho Frontend vẽ biểu đồ.

---

## 3. Huấn luyện & Cách hoạt động của các mô hình

### 3.1. Pipeline tiền xử lý tiếng Việt (`machine_learning/preprocessing.py`)

Để cả mô hình truyền thống và học sâu học trên cùng một phân phối ngôn ngữ tự nhiên, toàn bộ dữ liệu huấn luyện và dữ liệu suy luận thực tế đều dùng chung một pipeline:

- Chuẩn hóa Unicode, chữ thường, xóa đường dẫn URL, thẻ HTML và ký tự lặp kéo dài.
- Ánh xạ **Emoji** sang từ ngữ mang cảm xúc tương ứng (ví dụ: biểu tượng cười, tức giận, buồn...).
- Chuẩn hóa **teencode / từ viết tắt** tiếng Việt phổ biến trên mạng xã hội (`ko`, `k`, `kh` $\rightarrow$ `không`, v.v.).
- Tách từ tiếng Việt (_word segmentation_) cho đầu vào của PhoBERT.

### 3.2. Hai mô hình cốt lõi

| Tiêu chí                     | Mô hình Baseline: TF-IDF + SVM                                          | Mô hình Nâng cao: PhoBERT Fine-tuned                        |
| :--------------------------- | :---------------------------------------------------------------------- | :---------------------------------------------------------- |
| **Biểu diễn văn bản**        | TF-IDF Vectorizer (n-gram từ/ký tự)                                     | PhoBERT Tokenizer (`vinai/phobert-base`)                    |
| **Thuật toán phân loại**     | Support Vector Machine (`best_model.pkl`)                               | Transformer Encoder + lớp phân loại 3 nhãn (PyTorch)        |
| **Đặc điểm vận hành**        | Dung lượng nhẹ (~230KB), tốc độ suy luận cực nhanh trên CPU             | Nắm bắt ngữ cảnh, câu phủ định và sắc thái phức tạp tốt hơn |
| **Độ tin cậy (Confidence)**  | Tính từ `predict_proba` hoặc chuẩn hóa Softmax trên `decision_function` | Tính trực tiếp từ phân phối Softmax đầu ra của mạng         |
| **Accuracy (trên tập Test)** | $74.42\%$                                                               | $79.16\%$                                                   |
| **$F_1$-score (Weighted)**   | $74.40\%$                                                               | $79.01\%$                                                   |

_(Các kết quả đánh giá chi tiết, ma trận nhầm lẫn và biểu đồ so sánh nằm trong thư mục `machine_learning/reports/`.)_

### 3.3. Cách tự huấn luyện lại mô hình (tùy chọn)

Nếu bạn muốn tự tái lập quá trình huấn luyện hoặc đánh giá trên tập dữ liệu của mình:

```bash
# 1. Cài đặt đầy đủ thư viện ML
pip install -r machine_learning/requirements.txt

# 2. Huấn luyện mô hình Baseline (TF-IDF + SVM)
python machine_learning/train_svm.py

# 3. Fine-tune PhoBERT (khuyến nghị chạy trên máy có GPU hoặc dùng notebook Colab có sẵn)
python machine_learning/train_phobert.py
# Hoặc mở: machine_learning/notebooks/PhoBERT_FineTuning_Colab.ipynb

# 4. Đánh giá và xuất bảng/biểu đồ so sánh giữa các mô hình
python machine_learning/evaluate_and_compare.py
```

---

## 4. Cấu trúc thư mục dự án

```text
YTB_sentiment_analysis/
├── backend/
│   ├── app.py                  # Flask API server (/api/health, /api/analyze)
│   ├── youtube_crawler.py      # Module gọi YouTube Data API v3
│   ├── run.py                  # Điểm khởi chạy server backend
│   └── requirements.txt        # Thư viện Python cho backend & suy luận
├── frontend/
│   ├── src/
│   │   ├── api.js              # Kết nối API backend & quản lý mã lỗi
│   │   ├── App.jsx             # Giao diện chính
│   │   └── components/         # Biểu đồ, bảng so sánh, danh sách bình luận
│   └── package.json            # Cấu hình React + Vite
├── machine_learning/
│   ├── preprocessing.py        # Module làm sạch, chuẩn hóa teencode & emoji
│   ├── train_svm.py            # Script huấn luyện SVM
│   ├── train_phobert.py        # Script fine-tune PhoBERT
│   ├── predict_phobert.py      # Wrapper suy luận cho PhoBERT
│   ├── evaluate_and_compare.py # Script đánh giá & vẽ biểu đồ so sánh
│   ├── models/
│   │   ├── best_model.pkl      # Mô hình SVM đã huấn luyện (có sẵn trên repo)
│   │   └── phobert_finetuned/  # Thư mục đặt trọng số PhoBERT (tải riêng)
│   └── reports/                # Kết quả metrics (.json, .csv, .png, .tex)
├── .env.example                # File biến môi trường mẫu
├── .gitignore
└── README.md
```

---

## 5. Yêu cầu tiên quyết & Chuẩn bị khóa API

Trước khi cài đặt trên bất kỳ hệ điều hành nào, máy cần cài sẵn:

- **Git**
- **Python 3.10** (khuyến nghị Python `3.10.x` để tương thích tốt nhất với PyTorch, Transformers và Scikit-learn)
- **Node.js 18+** và **npm**
- **YouTube Data API v3 Key (miễn phí):**
  1. Truy cập [Google Cloud Console](https://console.cloud.google.com/), tạo một Project mới.
  2. Vào **APIs & Services → Library**, tìm **YouTube Data API v3** và nhấn **Enable**.
  3. Vào **APIs & Services → Credentials → + Create Credentials → API key** để lấy chuỗi khóa bắt đầu bằng `AIzaSy...`.

---

## 6. Hướng dẫn cài đặt & chạy trên từng hệ điều hành

### Bước chung: Clone dự án và chuẩn bị trọng số PhoBERT

```bash
git clone https://github.com/dung06coder/YTB_sentiment_analysis.git
cd YTB_sentiment_analysis
```

> **Lưu ý về trọng số PhoBERT (>100MB):** Vì giới hạn dung lượng của GitHub, thư mục trọng số `machine_learning/models/phobert_finetuned/` không lưu trực tiếp trong repo.
>
> - **Cách 1 (Đầy đủ 2 mô hình):** Tải thư mục `phobert_finetuned` (chứa `config.json` và file trọng số) từ đường dẫn chia sẻ của nhóm _(cập nhật link Google Drive / GitHub Releases tại đây)_ và đặt vào `machine_learning/models/phobert_finetuned/`.
> - **Cách 2 (Chạy nhanh không cần tải trọng số PhoBERT):** Bạn vẫn có thể chạy hệ thống ngay lập tức với mô hình **TF-IDF + SVM** (đã có sẵn `best_model.pkl`). Nếu muốn bật PhoBERT ở chế độ thử nghiệm cấu trúc, đặt `ALLOW_PHOBERT_BASE_FALLBACK=true` trong file `.env`.

---

### A. Cài đặt trên Ubuntu / Linux

```bash
# 1. Cài đặt công cụ cơ bản (nếu máy chưa có)
sudo apt update
sudo apt install -y python3 python3-venv python3-pip nodejs npm git

# 2. Tạo và kích hoạt môi trường ảo tại thư mục gốc dự án
python3 -m venv .venv
source .venv/bin/activate

# 3. Cài đặt thư viện Backend & ML
pip install --upgrade pip
pip install -r backend/requirements.txt

# 4. Tạo file biến môi trường và điền YOUTUBE_API_KEY
cp .env.example .env
nano .env   # Thay YOUTUBE_API_KEY=AIzaSy... rồi lưu (Ctrl+O, Enter, Ctrl+X)

# 5. Chạy Backend (Terminal 1 - đứng tại thư mục gốc dự án)
python -m backend.run
```

Mở **Terminal 2** để chạy Frontend:

```bash
cd frontend
npm install
npm run dev
```

---

### B. Cài đặt trên Windows (PowerShell)

```powershell
# 1. Tạo và kích hoạt môi trường ảo tại thư mục gốc dự án
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# (Nếu PowerShell chặn script, chạy trước lệnh: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned)

# Nếu bạn dùng Anaconda / Miniconda, có thể tạo môi trường Conda thay thế:
# conda create -n ytb_sentiment python=3.10 -y
# conda activate ytb_sentiment

# 2. Cài đặt thư viện Backend & ML
python -m pip install --upgrade pip
pip install -r backend/requirements.txt

# 3. Tạo file .env từ file mẫu
Copy-Item .env.example .env
# Mở file .env bằng Notepad hoặc VS Code và điền YOUTUBE_API_KEY của bạn

# 4. Chạy Backend (Terminal 1 - đứng tại thư mục gốc dự án)
python -m backend.run
```

Mở **Terminal 2** để chạy Frontend:

```powershell
cd frontend
npm install
npm run dev
```

---

### C. Cài đặt trên macOS (Intel & Apple Silicon M1/M2/M3/M4)

```bash
# 1. Cài Python 3.10 và Node.js qua Homebrew (nếu chưa có)
brew install python@3.10 node

# 2. Tạo và kích hoạt môi trường ảo tại thư mục gốc dự án
python3.10 -m venv .venv
source .venv/bin/activate

# 3. Cài đặt thư viện Backend & ML
pip install --upgrade pip
pip install -r backend/requirements.txt

# 4. Tạo file biến môi trường và điền YOUTUBE_API_KEY
cp .env.example .env
open -e .env   # Điền YOUTUBE_API_KEY=AIzaSy... rồi lưu (Cmd+S)

# 5. Chạy Backend (Terminal 1 - đứng tại thư mục gốc dự án)
python -m backend.run
```

Mở **Terminal 2** để chạy Frontend:

```bash
cd frontend
npm install
npm run dev
```

---

## 7. Kiểm tra hoạt động sau khi khởi chạy

1. **Kiểm tra trạng thái Backend:** Mở trình duyệt truy cập `http://localhost:5000/api/health`. Kết quả hợp lệ sẽ có dạng:
   ```json
   {
     "status": "ok",
     "models": { "svm": true, "phobert": true },
     "errors": { "svm": null, "phobert": null }
   }
   ```
2. **Mở giao diện người dùng:** Truy cập `http://localhost:5173/`. Dán link một video YouTube bất kỳ, chọn **TF-IDF + SVM** và/hoặc **PhoBERT** rồi nhấn **Phân tích**.
3. **Kiểm thử nhanh API không cần YouTube API Key:**
   ```bash
   curl -X POST http://localhost:5000/api/analyze \
     -H "Content-Type: application/json" \
     -d '{"models": ["svm"], "comments": ["Video phân tích rất hay và dễ hiểu!", "Nội dung hơi lan man, chưa đúng trọng tâm."]}'
   ```

---

## 8. Tài liệu API ngắn gọn

| Endpoint       | Phương thức | Mô tả                                                                                                                                                                                      |
| :------------- | :---------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `/api/health`  |    `GET`    | Kiểm tra tình trạng server và trạng thái nạp từng mô hình AI trong RAM.                                                                                                                    |
| `/api/analyze` |   `POST`    | Nhận `{"url": "...", "models": ["svm", "phobert"], "max_comments": 500}` (hoặc mảng `comments`), trả về phân bổ cảm xúc, `avg_confidence`, `metrics`, `latency_ms` và danh sách `samples`. |

---

## 9. Thành viên thực hiện

- **Quản lý tích hợp & Kiến trúc hệ thống:** Dũng
- **Dữ liệu & ML Truyền thống (Preprocessing, TF-IDF + SVM):** Vitou
- **Học sâu (Deep Learning & PhoBERT):** Hùng
- **Backend & API Inference:** Công
- **Frontend & Trực quan hóa dữ liệu:** Malang
