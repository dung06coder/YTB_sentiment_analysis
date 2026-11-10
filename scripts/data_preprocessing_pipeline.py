import re
import pandas as pd
from pyvi import ViTokenizer
import logging

# Cấu hình logging để theo dõi tiến trình
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. TỪ ĐIỂN CHUẨN HÓA TEENCODE & VIẾT TẮT (Nên mở rộng thêm trong quá trình làm YTB_sentiment_analysis)
TEENCODE_DICT = {
    'ko': 'không', 'k': 'không', 'kh': 'không', 'khg': 'không', 'kg': 'không',
    'đc': 'được', 'dc': 'được', 'ok': 'đồng ý', 'oke': 'đồng ý', 'okela': 'đồng ý',
    'vcl': 'vô cùng', 'vl': 'vô cùng', 'vkl': 'vô cùng', 'vdd': 'vô cùng',
    'r': 'rồi', 'vs': 'với', 'sp': 'sản phẩm', 'mn': 'mọi người', 'm.n': 'mọi người',
    'mik': 'mình', 'm': 'mày', 't': 'tao', 'g9': 'chúc ngủ ngon', 'kb': 'không biết',
    'thik': 'thích', 'ntn': 'như thế nào', 'dt': 'điện thoại', 'mkt': 'marketing',
    'chuẩnn': 'chuẩn', 'đẹpp': 'đẹp', 'quá đỉnhhh': 'quá đỉnh'
}

def normalize_teencode(text):
    """Thay thế các từ viết tắt, teencode bằng từ chuẩn tiếng Việt."""
    words = text.split()
    normalized_words = [TEENCODE_DICT.get(word, word) for word in words]
    return ' '.join(normalized_words)

def clean_text_pipeline(text):
    """
    Pipeline làm sạch chuỗi văn bản thô.
    Lưu ý: Chưa bao gồm bước tách từ (Word Segmentation).
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
        
    text = text.lower() # 1. Chữ thường
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # 2. Xóa link
    text = re.sub(r'<.*?>', '', text) # 3. Xóa thẻ HTML
    text = re.sub(r'\@\w+|\#\w+', '', text) # 4. Xóa Tag/Hashtag
    text = re.sub(r'[^\w\s]', ' ', text) # 5. Xóa ký tự đặc biệt, dấu câu (giữ lại chữ và số)
    text = re.sub(r'\d+', ' ', text) # 6. Xóa số (Tùy chọn: Bỏ dòng này nếu số có ý nghĩa)
    text = re.sub(r'([a-z])\1+', r'\1', text) # 7. Xóa ký tự lặp (ví dụ: đẹppp -> đẹp)
    text = re.sub(r'\s+', ' ', text).strip() # 8. Xóa khoảng trắng thừa
    text = normalize_teencode(text) # 9. Chuẩn hóa teencode
    
    return text

def process_dataset(input_path, output_path):
    """
    Hàm xử lý toàn bộ dataset 60k dòng từ đầu đến cuối.
    """
    logging.info(f"Đang đọc dữ liệu thô từ: {input_path}")
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        logging.error(f"Lỗi khi đọc file: {e}")
        return

    original_count = len(df)
    logging.info(f"Tổng số dòng ban đầu: {original_count}")

    # BƯỚC 1: LÀM SẠCH CẤU TRÚC (Basic Cleaning)
    logging.info("Bắt đầu lọc cột, xóa giá trị rỗng và trùng lặp...")
    # Chỉ giữ lại các cột cần thiết (Giả sử data có comment_id, text_original)
    if 'text_original' not in df.columns:
        logging.error("Không tìm thấy cột 'text_original' trong dataset.")
        return
        
    df = df[['comment_id', 'text_original']] if 'comment_id' in df.columns else df[['text_original']]
    
    df = df.dropna(subset=['text_original'])
    df = df.drop_duplicates(subset=['text_original'])
    
    cleaned_count = len(df)
    logging.info(f"Số dòng sau khi xóa rỗng và trùng lặp: {cleaned_count} (Đã xóa {original_count - cleaned_count} dòng rác)")

    # BƯỚC 2: TIỀN XỬ LÝ VĂN BẢN & CHUẨN HÓA
    logging.info("Đang thực hiện làm sạch văn bản và chuẩn hóa teencode (Quá trình này mất vài phút cho 60k dòng)...")
    df['text_cleaned'] = df['text_original'].apply(clean_text_pipeline)
    
    # Xóa những dòng mà sau khi làm sạch chỉ còn chuỗi rỗng
    df = df[df['text_cleaned'].str.strip() != '']

    # BƯỚC 3: TÁCH TỪ TIẾNG VIỆT (Word Segmentation với Pyvi)
    logging.info("Đang thực hiện tách từ (Word Segmentation) bằng Pyvi...")
    df['text_segmented'] = df['text_cleaned'].apply(lambda x: ViTokenizer.tokenize(x))

    # BƯỚC 4: LƯU KẾT QUẢ
    logging.info(f"Đang lưu dữ liệu đã tiền xử lý ra file: {output_path}")
    df.to_csv(output_path, index=False, encoding='utf-8-sig ')
    logging.info("Hoàn tất! Pipeline xử lý dữ liệu đã chạy xong.")

if __name__ == "__main__":
    # Thay đổi tên file cho phù hợp với máy của bạn
    INPUT_FILE = "raw_comments.csv" 
    OUTPUT_FILE = "cleaned_youtube_comments1.csv"
    
    process_dataset(INPUT_FILE, OUTPUT_FILE)
    print("Mã nguồn đã sẵn sàng. Hãy uncomment dòng process_dataset để chạy thực tế.")
