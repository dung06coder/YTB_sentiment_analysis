import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from 'recharts'
import { LABELS, KEYS, modelName, tip } from '../models'

export default function PieCard({ results, model, onModel }) {
  const r = results[model]
  const data = KEYS.map((k) => ({ key: k, name: LABELS[k].text, value: r.counts[k] }))
  return (
    <div className="card">
      <h3>Biểu đồ tròn</h3>
      <div className="tabs">
        {Object.keys(results).map((m) => (
          <button key={m} className={`tab ${m === model ? 'on' : ''}`} onClick={() => onModel(m)}>
            {modelName(m)}
          </button>
        ))}
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <PieChart>
          <Pie data={data} dataKey="value" nameKey="name" outerRadius={80} stroke="none">
            {data.map((d) => <Cell key={d.key} fill={LABELS[d.key].color} />)}
          </Pie>
          <Tooltip {...tip} formatter={(v) => `${v} bình luận`} />
        </PieChart>
      </ResponsiveContainer>
      <ul className="legend">
        {KEYS.map((k) => (
          <li key={k}>
            <i className="dot" style={{ background: LABELS[k].color }} /> {LABELS[k].text}
            <span>{r[k]}%</span>
          </li>
        ))}
      </ul>
    </div>
  )
}