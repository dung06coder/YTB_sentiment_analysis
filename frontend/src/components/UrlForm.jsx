import { useState } from 'react'

const MODELS = [
  { id: 'svm', label: 'TF-IDF + SVM', desc: 'Nhanh, nhẹ' },
  { id: 'phobert', label: 'PhoBERT', desc: 'Chính xác hơn, chậm hơn' },
]
const SAMPLE = 'https://www.youtube.com/watch?v=abc12345'

export default function UrlForm({ onSubmit, disabled }) {
  const [url, setUrl] = useState('')
  const [selected, setSelected] = useState(['svm', 'phobert'])
  const [formError, setFormError] = useState('')

  const toggle = (id) =>
    setSelected((prev) => (prev.includes(id) ? prev.filter((m) => m !== id) : [...prev, id]))

  const handleSubmit = () => {
    if (!url.trim()) return setFormError('Bạn chưa nhập link video.')
    if (selected.length === 0) return setFormError('Hãy chọn ít nhất 1 mô hình.')
    setFormError('')
    onSubmit(url.trim(), selected)
  }

  return (
    <div className="card">
      <div className="label">Link video YouTube</div>
      <div className="input-row">
        <input
          className="input"
          type="text"
          placeholder="https://www.youtube.com/watch?v=..."
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && !disabled && handleSubmit()}
          disabled={disabled}
        />
        <button className="ghost" onClick={() => setUrl(SAMPLE)} disabled={disabled}>
          Dùng link mẫu
        </button>
      </div>

      <div className="label mt">Chọn mô hình để so sánh</div>
      <div className="models">
        {MODELS.map((m) => (
          <label key={m.id} className={`model-option ${selected.includes(m.id) ? 'active' : ''}`}>
            <input type="checkbox" checked={selected.includes(m.id)} onChange={() => toggle(m.id)} disabled={disabled} />
            <span className="tick">✓</span>
            <span>
              <div className="model-name">{m.label}</div>
              <div className="model-desc">{m.desc}</div>
            </span>
          </label>
        ))}
      </div>

      {formError && <p className="error-text">{formError}</p>}

      <button className="btn" onClick={handleSubmit} disabled={disabled}>
        {disabled ? 'Đang phân tích...' : 'Phân tích ngay'}
      </button>
    </div>
  )
}