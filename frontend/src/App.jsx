import { useState } from 'react'
import './App.css'
import UrlForm from './components/UrlForm'
import LoadingView from './components/LoadingView'
import ResultChart, { LABELS } from './components/ResultChart'
import CommentList from './components/CommentList'
import { analyze } from './api'

function dominant(results) {
  const ids = Object.keys(results)
  const avg = { positive: 0, negative: 0, neutral: 0 }
  ids.forEach((m) => Object.keys(avg).forEach((k) => (avg[k] += results[m][k] / ids.length)))
  const key = Object.keys(avg).reduce((a, b) => (avg[a] >= avg[b] ? a : b))
  return { key, value: avg[key] }
}

export default function App() {
  const [status, setStatus] = useState('idle') // idle | loading | success | error
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)

  const handleSubmit = async (url, models) => {
    setStatus('loading')
    setError(null)
    try {
      const result = await analyze(url, models)
      setData(result)
      setStatus('success')
    } catch (err) {
      setError(err)
      setStatus('error')
    }
  }

  const reset = () => {
    setStatus('idle')
    setData(null)
    setError(null)
  }

  const top = data ? dominant(data.results) : null

  return (
    <div className="container">
      <header className="hero">
        <div className="logo" />
        <h1>Phân tích cảm xúc bình luận YouTube</h1>
        <p className="hint">Dán link video, chọn mô hình và xem khoảng 500 bình luận mới nhất đang nói gì.</p>
      </header>

      <UrlForm onSubmit={handleSubmit} disabled={status === 'loading'} />

      {status === 'loading' && <LoadingView />}

      {status === 'error' && (
        <div className="card error-card">
          <strong>Có lỗi: </strong>
          {error?.message}
          <div>
            <button className="btn" onClick={reset}>Thử lại</button>
          </div>
        </div>
      )}

      {status === 'success' && data && (
        <>
          <div className="stats">
            <div className="stat"><span>Bình luận đã phân tích</span><b>{data.total_comments}</b></div>
            <div className="stat"><span>Thời gian xử lý</span><b>{(data.latency_ms / 1000).toFixed(1)}s</b></div>
            <div className="stat">
              <span>Cảm xúc chủ đạo</span>
              <b style={{ color: LABELS[top.key].color }}>
                {LABELS[top.key].emoji} {LABELS[top.key].text} · {top.value.toFixed(1)}%
              </b>
            </div>
          </div>
          <ResultChart results={data.results} />
          <CommentList results={data.results} />
        </>
      )}
    </div>
  )
}