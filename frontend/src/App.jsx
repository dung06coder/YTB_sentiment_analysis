import { useEffect, useRef, useState } from 'react'
import './App.css'
import UrlForm from './components/UrlForm'
import LoadingView from './components/LoadingView'
import Summary from './components/Summary'
import PieCard from './components/PieCard'
import ResultChart from './components/ResultChart'
import CommentList from './components/CommentList'
import ModelTables from './components/ModelTables'
import { analyze, USE_MOCK, ERROR_MESSAGES } from './api'

export default function App() {
  const [status, setStatus] = useState('idle') // idle | loading | success | error
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [model, setModel] = useState('')
  const ctrlRef = useRef(null)
  const resultRef = useRef(null)

  useEffect(() => {
    if (status === 'success') {
      resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [status])

  const handleSubmit = async (url, models) => {
    const ctrl = new AbortController()
    ctrlRef.current = ctrl
    setStatus('loading')
    setError(null)
    try {
      const result = await analyze(url, models, ctrl.signal)
      if (!result?.results || result.total_comments === 0) {
        setError({ code: 'NO_COMMENTS', message: ERROR_MESSAGES.NO_COMMENTS })
        setStatus('error')
        return
      }
      setData(result)
      setModel(Object.keys(result.results)[0])
      setStatus('success')
    } catch (err) {
      if (err.code === 'CANCELLED') {
        setStatus('idle')
        return
      }
      setError(err)
      setStatus('error')
    }
  }

  const handleCancel = () => ctrlRef.current?.abort()

  const reset = () => {
    setStatus('idle')
    setData(null)
    setError(null)
  }

  return (
    <div className="container">
      <header className="hero">
        <div className="logo" />
        <h1>Phân tích cảm xúc bình luận YouTube</h1>
        <p className="hint">Dán link video, chọn mô hình và xem khoảng 500 bình luận mới nhất đang nói gì.</p>
      </header>

      <div className="narrow">
        {USE_MOCK && (
          <p className="hint" style={{ textAlign: 'center', marginBottom: 8 }}>
            Chế độ demo: đang hiển thị dữ liệu mẫu, chưa kết nối backend.
          </p>
        )}
        <UrlForm onSubmit={handleSubmit} disabled={status === 'loading'} />
        {status === 'loading' && <LoadingView onCancel={handleCancel} />}
        {status === 'error' && (
          <div className="card error-card" role="alert">
            <strong>Không thể phân tích</strong>
            <p style={{ margin: '8px 0 0' }}>{error?.message}</p>
            <button className="btn" onClick={reset}>Thử lại</button>
          </div>
        )}
      </div>

      {status === 'success' && data && (
        <div ref={resultRef}>
          <div className="grid3">
            <div className="col">
              <Summary data={data} model={model} />
              <PieCard results={data.results} model={model} onModel={setModel} />
            </div>
            <div className="col">
              <ResultChart results={data.results} />
            </div>
            <div className="col">
              <CommentList key={model} samples={data.results[model].samples} model={model} />
            </div>
          </div>
          <ModelTables results={data.results} />
        </div>
      )}
    </div>
  )
}