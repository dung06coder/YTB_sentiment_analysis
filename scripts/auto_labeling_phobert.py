import pandas as pd
import torch
from transformers import pipeline
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def auto_label_dataset(input_file, output_file):
    logging.info(f"Đang đọc dữ liệu từ {input_file}...")
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        logging.error(f"Không tìm thấy file {input_file}")
        return

    # Loại bỏ các dòng rỗng (nếu có)
    df = df.dropna(subset=['text_segmented'])
    comments = df['text_segmented'].tolist()

    # Kiểm tra xem máy có hỗ trợ GPU không (RTX 4050 sẽ dùng device=0)
    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU (CUDA)" if device == 0 else "CPU"
    logging.info(f"Đang tải mô hình PhoBERT sentiment trên {device_name}...")

    # Sử dụng mô hình PhoBERT đã fine-tune cho phân tích cảm xúc tiếng Việt
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="wonrax/phobert-base-vietnamese-sentiment",
        device=device,
        truncation=True,
        max_length=256
    )

    logging.info("Bắt đầu gán nhãn cho 60.000 câu. Quá trình này có thể mất một lúc...")
    
    # Chạy dự đoán theo batch để tránh tràn RAM
    batch_size = 128
    results = []
    
    for i in tqdm(range(0, len(comments), batch_size), desc="Gán nhãn"):
        batch = comments[i:i+batch_size]
        # Xử lý các câu quá dài hoặc không hợp lệ
        batch = [str(text)[:1000] for text in batch] 
        preds = sentiment_pipeline(batch)
        results.extend(preds)

    # Bóc tách nhãn (Label) và Độ tự tin (Score)
    df['auto_label'] = [res['label'] for res in results]
    df['confidence_score'] = [res['score'] for res in results]

    # Map lại nhãn cho dễ hiểu (Tùy thuộc vào output của mô hình)
    # wonrax/phobert-base-vietnamese-sentiment thường trả ra: POS, NEG, NEU
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    logging.info(f"Đã lưu kết quả gán nhãn thành công tại {output_file}")

if __name__ == "__main__":
    INPUT_FILE = "cleaned_youtube_comments.csv"
    OUTPUT_FILE = "labeled_youtube_comments.csv"
    # Uncomment dòng dưới để chạy
    auto_label_dataset(INPUT_FILE, OUTPUT_FILE)
    print("Code gán nhãn đã sẵn sàng.")
