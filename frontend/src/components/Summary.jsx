import { modelName } from '../models'

export default function Summary({ data, model }) {
  const r = data.results[model]
  return (
    <div className="card">
      <h3>Tổng quan</h3>
      <div className="row"><span>Bình luận đã thu thập</span><b>{data.total_comments}</b></div>
      <div className="row">
        <span>Độ tin cậy TB ({modelName(model)})</span>
        <b>{r.avg_confidence != null ? `${r.avg_confidence}%` : '—'}</b>
      </div>
      <div className="row"><span>Thời gian xử lý</span><b>{(data.latency_ms / 1000).toFixed(1)}s</b></div>
    </div>
  )
}