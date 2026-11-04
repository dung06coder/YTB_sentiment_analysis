import { useState } from 'react'
import './App.css'
import UrlForm from './components/UrlForm'
import LoadingView from './components/LoadingView'
import Summary from './components/Summary'
import PieCard from './components/PieCard'
import ResultChart from './components/ResultChart'
import CommentList from './components/CommentList'
import ModelTables from './components/ModelTables'
import { analyze } from './api'

export default function App() {
  const [status, setStatus] = useState('idle') // idle | loading | success | error
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [model, setModel] = useState('')

  const handleSubmit = async (url, models) => {
    setStatus('loading')
    setError(null)
    try {
      const result = await analyze(url, models)
      setData(result)
      setModel(Object.keys(result.results)[0])
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

  return (
    <div className="container">
      <header className="hero">
        <div className="logo" />
        <h1>Phân tích cảm xúc bình luận YouTube</h1>
        <p className="hint">Dán link video, chọn mô hình và xem khoảng 500 bình luận mới nhất đang nói gì.</p>
      </header>

      <div className="narrow">
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
      </div>

      {status === 'success' && data && (
        <>
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
        </>
      )}
    </div>
  )
}