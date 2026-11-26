from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()

YOUTUBE_COMMENTS_URL = "https://www.googleapis.com/youtube/v3/commentThreads"


class YouTubeCrawlerError(Exception):
    """Lỗi có mã code chuẩn để trả thẳng về Frontend."""

    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


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
        r"(?:youtube\.com/live/)([A-Za-z0-9_-]{11})",
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
) -> Dict[str, Any]:
    """
    Cào tối đa max_comments (mặc định 500) bình luận cấp cao nhất mới nhất.
    Trả về dict gồm video_id và danh sách bình luận kèm metadata.
    """
    key = (api_key or os.getenv("YOUTUBE_API_KEY", "")).strip()
    if not key or key == "YOUR_YOUTUBE_API_KEY":
        raise YouTubeCrawlerError(
            code="CONFIG_ERROR",
            message="Thiếu YOUTUBE_API_KEY. Hãy cấu hình trong file .env.",
            status_code=500,
        )

    video_id = extract_video_id(url)
    if not video_id:
        raise YouTubeCrawlerError(
            code="INVALID_URL",
            message="Link YouTube không hợp lệ hoặc không tìm thấy video_id.",
            status_code=400,
        )

    safe_max = max(1, min(int(max_comments), 1000))
    comments: List[Dict[str, Any]] = []
    next_page_token: Optional[str] = None

    while len(comments) < safe_max:
        page_size = min(100, safe_max - len(comments))
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

        try:
            response = requests.get(YOUTUBE_COMMENTS_URL, params=params, timeout=15)
        except requests.exceptions.RequestException as exc:
            raise YouTubeCrawlerError(
                code="NETWORK",
                message=f"Không thể kết nối tới YouTube Data API: {exc}",
                status_code=502,
            ) from exc

        try:
            data = response.json()
        except ValueError as exc:
            raise YouTubeCrawlerError(
                code="YOUTUBE_API_ERROR",
                message=f"YouTube API trả về dữ liệu không phải JSON (HTTP {response.status_code}).",
                status_code=502,
            ) from exc

        if response.status_code != 200:
            err_obj = data.get("error", {})
            err_msg = err_obj.get("message", "Lỗi gọi YouTube Data API.")
            err_list = err_obj.get("errors", [])
            reason = err_list[0].get("reason", "") if err_list else ""

            if reason == "commentsDisabled":
                raise YouTubeCrawlerError(
                    code="COMMENTS_DISABLED",
                    message="Video này đã tắt bình luận nên không thể phân tích.",
                    status_code=403,
                )
            if response.status_code == 404 or reason == "videoNotFound":
                raise YouTubeCrawlerError(
                    code="VIDEO_NOT_FOUND",
                    message="Không tìm thấy video. Video có thể đã bị xóa hoặc để riêng tư.",
                    status_code=404,
                )
            if reason in ("quotaExceeded", "dailyLimitExceeded"):
                raise YouTubeCrawlerError(
                    code="QUOTA_EXCEEDED",
                    message="YouTube API đã hết hạn mức (quota) trong ngày.",
                    status_code=429,
                )

            raise YouTubeCrawlerError(
                code="YOUTUBE_API_ERROR",
                message=f"YouTube API lỗi ({response.status_code}): {err_msg}",
                status_code=502,
            )

        items = data.get("items", [])
        if not items:
            break

        for item in items:
            snippet = (
                item.get("snippet", {})
                .get("topLevelComment", {})
                .get("snippet", {})
            )
            text = str(snippet.get("textDisplay", "")).strip()
            if not text:
                continue

            comments.append(
                {
                    "text": text,
                    "author": str(snippet.get("authorDisplayName", "Ẩn danh")).lstrip("@"),
                    "published_at": str(snippet.get("publishedAt", "")),
                    "like_count": int(snippet.get("likeCount", 0) or 0),
                }
            )
            if len(comments) >= safe_max:
                break

        next_page_token = data.get("nextPageToken")
        if not next_page_token:
            break

    if not comments:
        raise YouTubeCrawlerError(
            code="NO_COMMENTS",
            message="Video chưa có bình luận nào để phân tích.",
            status_code=404,
        )

    return {
        "video_id": video_id,
        "comments": comments,
    }


if __name__ == "__main__":
    test_url = input("Nhập link video YouTube để test: ").strip()
    try:
        result = fetch_video_comments(test_url, max_comments=20)
        print(f"Video ID: {result['video_id']} | Lấy được {len(result['comments'])} bình luận.")
        for idx, c in enumerate(result["comments"][:5], 1):
            print(f"{idx}. [{c['author']} | {c['like_count']} likes] {c['text']}")
    except YouTubeCrawlerError as exc:
        print(f"Lỗi [{exc.code}]: {exc.message}")