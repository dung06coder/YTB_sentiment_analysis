import os
import re
from typing import List, Optional
import requests
from dotenv import load_dotenv

load_dotenv()

YOUTUBE_COMMENTS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"


def extract_video_id(url: str) -> Optional[str]:
    """Tách video_id từ các định dạng URL YouTube phổ biến."""
    if not url or not isinstance(url, str):
        return None

    url = url.strip()
    patterns = [
        r"(?:v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([A-Za-z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def fetch_video_comments(
    url: str,
    max_comments: int = 500,
    api_key: Optional[str] = None,
) -> List[str]:
    """Cào tối đa max_comments (mặc định 500) bình luận cấp cao nhất mới nhất."""
    key = api_key or os.getenv("YOUTUBE_API_KEY", "").strip()
    if not key or key == "YOUR_YOUTUBE_API_KEY":
        raise ValueError("Thiếu YOUTUBE_API_KEY. Hãy cấu hình trong file .env.")

    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("URL YouTube không hợp lệ hoặc không tìm thấy video_id.")

    comments: List[str] = []
    next_page_token: Optional[str] = None

    while len(comments) < max_comments:
        page_size = min(100, max_comments - len(comments))
        params = {
            "part": "snippet",
            "videoId": video_id,
            "maxResults": page_size,
            "order": "time",
            "textFormat": "plainText",
            "key": key,
        }
        if next_page_token:
            params["pageToken"] = next_page_token

        response = requests.get(YOUTUBE_COMMENTS_URL, params=params, timeout=15)
        data = response.json()

        if response.status_code != 200:
            error_msg = data.get("error", {}).get("message", "Lỗi gọi YouTube Data API.")
            raise RuntimeError(f"YouTube API lỗi ({response.status_code}): {error_msg}")

        items = data.get("items", [])
        if not items:
            break

        for item in items:
            snippet = item.get("snippet", {}).get("topLevelComment", {}).get("snippet", {})
            text = snippet.get("textDisplay", "").strip()
            if text:
                comments.append(text)
            if len(comments) >= max_comments:
                break

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

    return comments


if __name__ == "__main__":
    test_url = input("Nhập link video YouTube để test: ").strip()
    try:
        result = fetch_video_comments(test_url, max_comments=500)
        print(f"Lấy được {len(result)} bình luận.")
        for i, c in enumerate(result[:5], 1):
            print(f"{i}. {c}")
    except Exception as exc:
        print("Lỗi:", exc)
