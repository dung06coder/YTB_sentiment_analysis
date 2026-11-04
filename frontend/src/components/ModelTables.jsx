import { LABELS, KEYS, modelName } from '../models'

const METRICS = [['accuracy', 'Accuracy'], ['precision', 'Precision'], ['recall', 'Recall'], ['f1', 'F1-score']]

export default function ModelTables({ results }) {
  const ids = Object.keys(results)
  const best = (k) => Math.max(...ids.map((m) => results[m].metrics?.[k] ?? -1))
  const pct = (v) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`)

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
        <h3>Hiệu năng mô hình (tập test 20%)</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr><th>Chỉ số (macro)</th>{ids.map((m) => <th key={m}>{modelName(m)}</th>)}</tr>
            </thead>
            <tbody>
              {METRICS.map(([k, name]) => (
                <tr key={k}>
                  <td>{name}</td>
                  {ids.map((m) => {
                    const v = results[m].metrics?.[k]
                    return <td key={m} className={v != null && v === best(k) ? 'best' : ''}>{pct(v)}</td>
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