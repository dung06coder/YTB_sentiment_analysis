import mockResult from './mock/mockResult.json'

// Đổi thành false khi backend (TV4) chạy xong
export const USE_MOCK = true
const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:5000'
const TIMEOUT_MS = 60000

export const ERROR_MESSAGES = {
  INVALID_URL: 'Link YouTube không hợp lệ. Hãy kiểm tra lại link.',
  COMMENTS_DISABLED: 'Video này đã tắt bình luận nên không thể phân tích.',
  VIDEO_NOT_FOUND: 'Không tìm thấy video. Video có thể đã bị xóa hoặc để riêng tư.',
  NO_COMMENTS: 'Video chưa có bình luận nào để phân tích.',
  NETWORK: 'Không kết nối được tới server. Kiểm tra mạng hoặc xem backend đã chạy chưa.',
  TIMEOUT: 'Server phản hồi quá lâu (hơn 60 giây). Hãy thử lại hoặc chọn video khác.',
  CANCELLED: 'Bạn đã hủy phân tích.',
  UNKNOWN: 'Có lỗi xảy ra. Hãy thử lại sau.',
}
const fail = (code, fallback) => ({ code, message: ERROR_MESSAGES[code] || fallback || ERROR_MESSAGES.UNKNOWN })

const delay = (ms, signal) =>
  new Promise((resolve, reject) => {
    if (signal?.aborted) return reject(new DOMException('Aborted', 'AbortError'))
    const t = setTimeout(resolve, ms)
    signal?.addEventListener('abort', () => {
      clearTimeout(t)
      reject(new DOMException('Aborted', 'AbortError'))
    })
  })

const isYoutubeUrl = (url) =>
  /^(https?:\/\/)?(www\.|m\.)?(youtube\.com\/watch\?v=|youtu\.be\/)[\w-]{6,}/.test(url.trim())

export async function analyze(url, models, userSignal) {
  if (!isYoutubeUrl(url)) throw fail('INVALID_URL')

  const ctrl = new AbortController()
  let timedOut = false
  const timer = setTimeout(() => {
    timedOut = true
    ctrl.abort()
  }, TIMEOUT_MS)
  userSignal?.addEventListener('abort', () => ctrl.abort())

  try {
    if (USE_MOCK) {
      await delay(4000, ctrl.signal)
      // Gõ link có chữ "error" hoặc "notfound" để thử giao diện lỗi
      if (url.includes('error')) throw fail('COMMENTS_DISABLED')
      if (url.includes('notfound')) throw fail('VIDEO_NOT_FOUND')
      const results = {}
      models.forEach((m) => {
        if (mockResult.results[m]) results[m] = mockResult.results[m]
      })
      return { ...mockResult, results }
    }

    const res = await fetch(`${API_BASE}/api/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, models }),
      signal: ctrl.signal,
    })
    let data = null
    try {
      data = await res.json()
    } catch {
      /* phản hồi không phải JSON */
    }
    if (!res.ok || data?.error) throw fail(data?.error || 'UNKNOWN', data?.message)
    return data
  } catch (err) {
    if (err?.code) throw err
    if (err?.name === 'AbortError') throw fail(timedOut ? 'TIMEOUT' : 'CANCELLED')
    throw fail('NETWORK')
  } finally {
    clearTimeout(timer)
  }
}