
import os
import re
import pandas as pd
from googleapiclient.discovery import build

API_KEY = "AIzaSyA3rdnrkFNVApOFbvIhVlqCOAAXIXL2dW8"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(SCRIPT_DIR, "raw_comments.csv")
def extract_video_id(url):
    """Trích xuất Video ID từ các định dạng URL YouTube khác nhau"""
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1) if match else None

def append_youtube_comments(url, api_key, file_path, max_results=500):
    # 1. Kiểm tra trạng thái file hiện tại
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        print(f"[Info] File hiện đang lưu trữ {len(existing_df)} bình luận.")
        write_header = False
        mode = 'a' # Chế độ Append (Ghi nối)
    else:
        print("[Info] File chưa tồn tại. Hệ thống sẽ tạo file mới.")
        write_header = True
        mode = 'w' # Chế độ Write (Tạo mới)

    # 2. Xử lý URL
    video_id = extract_video_id(url)
    if not video_id:
        print("[Error] URL không hợp lệ. Vui lòng kiểm tra lại link YouTube.")
        return

    print(f"[Process] Đang tiến hành cào dữ liệu từ Video ID: {video_id}...")

    # 3. Kết nối API và thu thập dữ liệu
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments_list = []
    
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100
        )
        
        while request and len(comments_list) < max_results:
            response = request.execute()
            
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comments_list.append({
                    "Comment_ID": item['id'],
                    "Author": comment['authorDisplayName'],
                    "Published_At": comment['publishedAt'],
                    "Like_Count": comment['likeCount'],
                    "Text_Original": comment['textDisplay']
                })
                
            request = youtube.commentThreads().list_next(request, response)
            
    except Exception as e:
        print(f"[Error] Lỗi API hoặc video bị tắt bình luận: {e}")
        return

    # 4. Ghi dữ liệu và thống kê
    if comments_list:
        new_df = pd.DataFrame(comments_list)
        
        # Ghi nối vào CSV. Nếu mode='a', pandas sẽ tự động thêm vào cuối file.
        new_df.to_csv(file_path, mode=mode, header=write_header, index=False, encoding='utf-8')
        print(f"[Success] Đã thêm thành công {len(new_df)} bình luận mới.")
        
        # Đọc lại tổng số dòng để xác nhận
        total_df = pd.read_csv(file_path)
        print(f"[Result] Tổng dung lượng tập dữ liệu thô hiện tại: {len(total_df)} bình luận.\n")
    else:
        print("[Warning] Không tìm thấy bình luận nào cho video này.\n")

# Chạy thử nghiệm
if __name__ == "__main__":
    while True:
        target_url = input("Nhập URL YouTube (hoặc gõ 'q' để thoát): ")
        if target_url.lower() == 'q':
            break
        append_youtube_comments(target_url, API_KEY, FILE_PATH, max_results=5000)