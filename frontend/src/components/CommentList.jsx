import { useState } from 'react'
import { LABELS, KEYS, modelName } from '../models'

const fmt = (iso) =>
  iso ? new Date(iso).toLocaleString('vi-VN', { hour: '2-digit', minute: '2-digit', day: '2-digit', month: '2-digit' }) : ''

export default function CommentList({ samples, model }) {
  const [filter, setFilter] = useState('all')
  const list = samples.filter((s) => filter === 'all' || s.label === filter)
  const tabs = [['all', 'Tất cả'], ...KEYS.map((k) => [k, LABELS[k].text])]

  return (
    <div className="card">
      <h3>Bình luận tiêu biểu</h3>
      <p className="hint">Dự đoán bởi {modelName(model)}</p>
      <div className="tabs" style={{ marginTop: 10 }}>
        {tabs.map(([key, text]) => (
          <button key={key} className={`tab ${filter === key ? 'on' : ''}`} onClick={() => setFilter(key)}>
            {text}
          </button>
        ))}
      </div>
      {list.length === 0 && <p className="hint">Không có bình luận nào ở nhóm này.</p>}
      <ul className="comments">
        {list.map((s, i) => (
          <li key={i} className="comment">
            <div className="c-top">
              <span className="hint">{fmt(s.published_at)} · {s.author || 'Ẩn danh'}</span>
              <span className="badge" style={{ background: LABELS[s.label]?.color }}>{LABELS[s.label]?.text}</span>
            </div>
            <div className="c-text">{s.text}</div>
            <div className="c-bot">
              <span>👍 {s.like_count ?? 0}</span>
              <span>Tin cậy {(s.confidence * 100).toFixed(0)}%</span>
            </div>
          </li>
        ))}
      </ul>
    </div>
  )
}