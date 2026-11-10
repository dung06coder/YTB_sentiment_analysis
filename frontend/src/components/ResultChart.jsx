import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { LABELS, KEYS, modelName, modelColor, tip } from '../models'

export default function ResultChart({ results }) {
  const ids = Object.keys(results)
  const data = KEYS.map((k) => {
    const row = { name: LABELS[k].text }
    ids.forEach((m) => (row[m] = results[m][k]))
    return row
  })
  return (
    <div className="card">
      <h3>So sánh mô hình ({ids.map(modelName).join(' vs ')})</h3>
      <ResponsiveContainer width="100%" height={360}>
        <BarChart data={data}>
          <XAxis dataKey="name" tick={{ fill: '#94a3b8' }} axisLine={false} tickLine={false} />
          <YAxis unit="%" domain={[0, 100]} tick={{ fill: '#94a3b8' }} axisLine={false} tickLine={false} />
          <Tooltip {...tip} cursor={{ fill: 'rgba(255,255,255,.05)' }} formatter={(v) => `${v}%`} />
          <Legend />
          {ids.map((m) => (
            <Bar key={m} dataKey={m} name={modelName(m)} fill={modelColor(m)} radius={[6, 6, 0, 0]} />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}