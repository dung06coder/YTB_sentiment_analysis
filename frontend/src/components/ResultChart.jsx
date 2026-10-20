import { PieChart, Pie, Cell, Tooltip, BarChart, Bar, XAxis, YAxis, Legend, ResponsiveContainer } from 'recharts'

export const LABELS = {
  positive: { text: 'Tích cực', color: '#34d399', emoji: '😄' },
  negative: { text: 'Tiêu cực', color: '#fb7185', emoji: '😡' },
  neutral: { text: 'Trung lập', color: '#94a3b8', emoji: '😐' },
}
export const MODEL_NAMES = { svm: 'TF-IDF + SVM', phobert: 'PhoBERT' }
const MODEL_COLORS = { svm: '#818cf8', phobert: '#f472b6' }
const tip = {
  contentStyle: { background: '#111827', border: '1px solid #334155', borderRadius: 10 },
  itemStyle: { color: '#e5e7eb' },
  labelStyle: { color: '#e5e7eb' },
}

export default function ResultChart({ results }) {
  const modelIds = Object.keys(results)
  const keys = Object.keys(LABELS)

  const barData = keys.map((k) => {
    const row = { name: LABELS[k].text }
    modelIds.forEach((m) => (row[m] = results[m][k]))
    return row
  })

  let note = null
  if (modelIds.length > 1) {
    const [a, b] = modelIds
    const diff = Math.abs(results[a].positive - results[b].positive)
    note = diff < 5
      ? `🤝 Hai mô hình khá đồng thuận (chênh ${diff.toFixed(1)}% ở nhóm tích cực).`
      : `⚖️ Hai mô hình lệch nhau ${diff.toFixed(1)}% ở nhóm tích cực, đáng để so sánh kỹ.`
  }

  return (
    <div>
      <div className="pie-grid">
        {modelIds.map((m) => {
          const r = results[m]
          const top = keys.reduce((a, b) => (r[a] >= r[b] ? a : b))
          const pieData = keys.map((k) => ({ name: LABELS[k].text, value: r.counts[k], key: k }))
          return (
            <div key={m} className="card">
              <h3>{MODEL_NAMES[m] || m}</h3>
              <div className="donut">
                <ResponsiveContainer width="100%" height={220}>
                  <PieChart>
                    <Pie data={pieData} dataKey="value" innerRadius={62} outerRadius={88} paddingAngle={3} stroke="none">
                      {pieData.map((d) => <Cell key={d.key} fill={LABELS[d.key].color} />)}
                    </Pie>
                    <Tooltip {...tip} formatter={(v) => `${v} bình luận`} />
                  </PieChart>
                </ResponsiveContainer>
                <div className="donut-center">
                  <span className="emoji">{LABELS[top].emoji}</span>
                  <b>{r[top].toFixed(1)}%</b>
                  <small>{LABELS[top].text}</small>
                </div>
              </div>
              <ul className="legend">
                {keys.map((k) => (
                  <li key={k}>
                    <i style={{ background: LABELS[k].color }} /> {LABELS[k].text}
                    <span>{r.counts[k]} bình luận · {r[k]}%</span>
                  </li>
                ))}
              </ul>
            </div>
          )
        })}
      </div>

      {modelIds.length > 1 && (
        <div className="card">
          <h3>So sánh giữa các mô hình (%)</h3>
          <p className="note">{note}</p>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={barData}>
              <XAxis dataKey="name" tick={{ fill: '#94a3b8' }} axisLine={false} tickLine={false} />
              <YAxis unit="%" tick={{ fill: '#94a3b8' }} axisLine={false} tickLine={false} />
              <Tooltip {...tip} cursor={{ fill: 'rgba(255,255,255,.05)' }} formatter={(v) => `${v}%`} />
              <Legend />
              {modelIds.map((m) => (
                <Bar key={m} dataKey={m} name={MODEL_NAMES[m] || m} fill={MODEL_COLORS[m] || '#64748b'} radius={[6, 6, 0, 0]} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}