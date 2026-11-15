import { LABELS, KEYS, modelName } from '../models'

const METRICS = [['accuracy', 'Accuracy'], ['precision', 'Precision'], ['recall', 'Recall'], ['f1', 'F1-score']]

export default function ModelTables({ results }) {
  const ids = Object.keys(results)
  const pct = (v) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`)

  const rows = [
    ...METRICS.map(([k, name]) => ({ name, get: (m) => results[m].metrics?.[k] })),
    ...KEYS.map((k) => ({
      name: `F1 - ${LABELS[k].text}`,
      get: (m) => results[m].metrics?.f1_per_class?.[k],
    })),
  ]
  const best = (get) => Math.max(...ids.map((m) => get(m) ?? -1))

  return (
    <div className="two-col">
      <div className="card">
        <h3>Bảng so sánh phân loại cảm xúc</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Cảm xúc</th>{ids.map((m) => <th key={m}>{modelName(m)}</th>)}</tr>
            </thead>
            <tbody>
              {KEYS.map((k) => (
                <tr key={k}>
                  <td><i className="dot" style={{ background: LABELS[k].color }} /> {LABELS[k].text}</td>
                  {ids.map((m) => <td key={m}>{results[m].counts[k]} ({results[m][k]}%)</td>)}
                </tr>
              ))}
              <tr>
                <td>Độ tin cậy TB</td>
                {ids.map((m) => <td key={m}>{results[m].avg_confidence != null ? `${results[m].avg_confidence}%` : '—'}</td>)}
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      <div className="card">
        <h3>Hiệu năng mô hình (trên tập test)</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Chỉ số (macro)</th>{ids.map((m) => <th key={m}>{modelName(m)}</th>)}</tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.name}>
                  <td>{r.name}</td>
                  {ids.map((m) => {
                    const v = r.get(m)
                    return <td key={m} className={v != null && v === best(r.get) ? 'best' : ''}>{pct(v)}</td>
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="hint" style={{ marginTop: 10 }}>Số liệu do nhóm ML đánh giá trên tập test, chữ xanh là mô hình tốt nhất.</p>
      </div>
    </div>
  )
}