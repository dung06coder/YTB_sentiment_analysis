import mockResult from "./mock/mockResult.json";

// Đổi thành true nếu muốn test giao diện bằng dữ liệu giả không cần bật backend
export const USE_MOCK = false;
const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:5000";
const TIMEOUT_MS = 120000;

export const ERROR_MESSAGES = {
  INVALID_URL: "Link YouTube không hợp lệ. Hãy kiểm tra lại link.",
  COMMENTS_DISABLED: "Video này đã tắt bình luận nên không thể phân tích.",
  VIDEO_NOT_FOUND:
    "Không tìm thấy video. Video có thể đã bị xóa hoặc để riêng tư.",
  NO_COMMENTS: "Video chưa có bình luận nào để phân tích.",
  QUOTA_EXCEEDED:
    "YouTube API đã hết lượt gọi hôm nay. Hãy thử lại sau hoặc đổi API Key.",
  MODEL_NOT_READY:
    "Mô hình AI bạn chọn chưa sẵn sàng trên server. Hãy kiểm tra trạng thái backend.",
  INVALID_MODEL: "Mô hình bạn chọn không hợp lệ.",
  NETWORK:
    "Không kết nối được tới server. Kiểm tra mạng hoặc xem backend đã chạy chưa.",
  TIMEOUT:
    "Server phản hồi quá lâu (hơn 120 giây). Hãy thử lại hoặc chọn ít mô hình hơn.",
  CANCELLED: "Bạn đã hủy phân tích.",
  UNKNOWN: "Có lỗi xảy ra. Hãy thử lại sau.",
};

const fail = (code, fallback) => ({
  code,
  message: ERROR_MESSAGES[code] || fallback || ERROR_MESSAGES.UNKNOWN,
});

const delay = (ms, signal) =>
  new Promise((resolve, reject) => {
    if (signal?.aborted)
      return reject(new DOMException("Aborted", "AbortError"));
    const t = setTimeout(resolve, ms);
    signal?.addEventListener("abort", () => {
      clearTimeout(t);
      reject(new DOMException("Aborted", "AbortError"));
    });
  });

const isYoutubeUrl = (url) =>
  /^(https?:\/\/)?(www\.|m\.)?(youtube\.com\/(watch\?v=|shorts\/|embed\/|live\/)|youtu\.be\/)[\w-]{6,}/.test(
    url.trim(),
  );

export async function analyze(url, models, userSignal) {
  if (!isYoutubeUrl(url)) throw fail("INVALID_URL");

  const ctrl = new AbortController();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    ctrl.abort();
  }, TIMEOUT_MS);
  userSignal?.addEventListener("abort", () => ctrl.abort());

  try {
    if (USE_MOCK) {
      await delay(4000, ctrl.signal);
      if (url.includes("error")) throw fail("COMMENTS_DISABLED");
      if (url.includes("notfound")) throw fail("VIDEO_NOT_FOUND");
      const results = {};
      models.forEach((m) => {
        if (mockResult.results[m]) results[m] = mockResult.results[m];
      });
      return { ...mockResult, results };
    }

    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url, models }),
      signal: ctrl.signal,
    });

    let data = null;
    try {
      data = await res.json();
    } catch {
      /* phản hồi không phải JSON */
    }

    if (!res.ok || data?.error)
      throw fail(data?.error || "UNKNOWN", data?.message);
    return data;
  } catch (err) {
    if (err?.code) throw err;
    if (err?.name === "AbortError")
      throw fail(timedOut ? "TIMEOUT" : "CANCELLED");
    throw fail("NETWORK");
  } finally {
    clearTimeout(timer);
  }
}
