import { useEffect, useState } from 'react'
import { modelName } from '../models'

function useCountUp(target, duration = 900) {
  const [v, setV] = useState(0)
  useEffect(() => {
    let raf
    const start = performance.now()
    const tick = (now) => {
      const t = Math.min((now - start) / duration, 1)
      setV(target * (1 - Math.pow(1 - t, 3)))
      if (t < 1) raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [target, duration])
  return v
}

export default function Summary({ data, model }) {
  const r = data.results[model]
  const total = Math.round(useCountUp(data.total_comments))
  const conf = useCountUp(r.avg_confidence ?? 0)

  return (
    <div className="card">
      <h3>Tổng quan</h3>
      <div className="row"><span>Bình luận đã thu thập</span><b>{total}</b></div>
      <div className="row">
        <span>Độ tin cậy TB ({modelName(model)})</span>
        <b>{r.avg_confidence != null ? `${conf.toFixed(1)}%` : '—'}</b>
      </div>
      <div className="row"><span>Thời gian xử lý</span><b>{(data.latency_ms / 1000).toFixed(1)}s</b></div>
    </div>
  )
}