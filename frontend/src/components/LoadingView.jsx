import { useEffect, useState } from 'react'

const STEPS = [
  'Lấy bình luận từ YouTube',
  'Làm sạch văn bản, đổi emoji',
  'Chạy các mô hình dự đoán',
  'Tổng hợp kết quả',
]

export default function LoadingView() {
  const [progress, setProgress] = useState(5)

  useEffect(() => {
    // Giả lập tiến độ, dừng ở 90% cho tới khi backend trả kết quả
    const timer = setInterval(() => setProgress((p) => (p < 90 ? p + 2 : p)), 300)
    return () => clearInterval(timer)
  }, [])

  const now = Math.min(Math.floor(progress / 25), STEPS.length - 1)

  return (
    <div className="card center">
      <div className="spinner" />
      <div className="progress">
        <div className="progress-bar" style={{ width: `${progress}%` }} />
      </div>
      <ul className="steps">
        {STEPS.map((s, i) => (
          <li key={s} className={i < now ? 'done' : i === now ? 'now' : ''}>
            {i < now ? '✓' : i === now ? '⏳' : '○'} {s}
          </li>
        ))}
      </ul>
      <p className="hint" style={{ marginTop: 14 }}>Việc này có thể mất 10-15 giây, bạn đợi chút nhé.</p>
    </div>
  )
}