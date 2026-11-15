import { useEffect, useState } from 'react'

const STEPS = [
  'Lấy bình luận từ YouTube',
  'Làm sạch văn bản, đổi emoji',
  'Chạy các mô hình dự đoán',
  'Tổng hợp kết quả',
]

export default function LoadingView({ onCancel }) {
  const [progress, setProgress] = useState(4)
  const [seconds, setSeconds] = useState(0)

  useEffect(() => {
    // Chạy chậm dần, tiến sát 92% và đứng đợi tới khi backend trả kết quả
    const p = setInterval(() => setProgress((v) => v + (92 - v) * 0.04), 200)
    const s = setInterval(() => setSeconds((v) => v + 1), 1000)
    return () => {
      clearInterval(p)
      clearInterval(s)
    }
  }, [])

  const now = Math.min(Math.floor(progress / 25), STEPS.length - 1)

  return (
    <div className="card center">
      <div className="spinner" />
      <div className="progress">
        <div className="progress-bar" style={{ width: `${progress}%` }} />
      </div>
      <p className="hint">{Math.round(progress)}% · đã chờ {seconds}s</p>
      <ul className="steps">
        {STEPS.map((s, i) => (
          <li key={s} className={i < now ? 'done' : i === now ? 'now' : ''}>
            {i < now ? '✓' : i === now ? '⏳' : '○'} {s}
          </li>
        ))}
      </ul>
      <p className="hint" style={{ marginTop: 14 }}>Việc này có thể mất 10-15 giây, bạn đợi chút nhé.</p>
      {onCancel && <button className="btn-ghost" onClick={onCancel}>Hủy</button>}
    </div>
  )
}