import { useState } from 'react'
import { LABELS, MODEL_NAMES } from './ResultChart'

export default function CommentList({ results }) {
  const [filter, setFilter] = useState('all')
  const tabs = [['all', 'Tất cả'], ...Object.keys(LABELS).map((k) => [k, `${LABELS[k].emoji} ${LABELS[k].text}`])]

  return (
    <div className="card">
      <h3>Bình luận tiêu biểu</h3>
      <div className="tabs">
        {tabs.map(([key, text]) => (
          <button key={key} className={`tab ${filter === key ? 'on' : ''}`} onClick={() => setFilter(key)}>
            {text}
          </button>
        ))}
      </div>

      {Object.keys(results).map((m) => {
        const list = results[m].samples.filter((s) => filter === 'all' || s.label === filter)
        return (
          <div key={m} className="comment-group">
            <h4>{MODEL_NAMES[m] || m}</h4>
            {list.length === 0 ? (
              <p className="empty">Không có bình luận nào ở nhóm này.</p>
            ) : (
              <ul className="comments">
                {list.map((s, i) => (
                  <li key={i} className="comment">
                    <span className="badge" style={{ background: LABELS[s.label].color }}>
                      {LABELS[s.label].emoji} {LABELS[s.label].text}
                    </span>
                    <span className="comment-text">{s.text}</span>
                    <span className="conf">
                      {(s.confidence * 100).toFixed(0)}%
                      <div><i style={{ width: `${s.confidence * 100}%` }} /></div>
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )
      })}
    </div>
  )
}