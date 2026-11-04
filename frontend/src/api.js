import mockResult from './mock/mockResult.json'

// Đổi thành false khi backend (TV4) chạy xong
const USE_MOCK = true
const API_BASE = 'http://localhost:5000'

const delay = (ms) => new Promise((r) => setTimeout(r, ms))

const isYoutubeUrl = (url) =>
  /^(https?:\/\/)?(www\.|m\.)?(youtube\.com\/watch\?v=|youtu\.be\/)[\w-]{6,}/.test(url.trim())

export async function analyze(url, models) {
  if (!isYoutubeUrl(url)) {
    throw { code: 'INVALID_URL', message: 'Link YouTube không hợp lệ.' }
  }

  if (USE_MOCK) {
    await delay(4000)
    // Gõ link có chữ "error" để thử giao diện lỗi
    if (url.includes('error')) {
      throw { code: 'COMMENTS_DISABLED', message: 'Video này đã tắt bình luận.' }
    }
    const results = {}
    models.forEach((m) => {
      if (mockResult.results[m]) results[m] = mockResult.results[m]
    })
    return { ...mockResult, results }
  }

  let res
  try {
    res = await fetch(`${API_BASE}/api/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, models }),
    })
  } catch {
    throw { code: 'NETWORK', message: 'Không kết nối được tới server.' }
  }
  const data = await res.json()
  if (!res.ok || data.error) {
    throw { code: data.error || 'UNKNOWN', message: data.message || 'Có lỗi xảy ra.' }
  }
  return data
}