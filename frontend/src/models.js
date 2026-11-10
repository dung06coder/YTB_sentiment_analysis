export const MODELS = [
  { id: 'svm', label: 'TF-IDF + SVM', desc: 'Nhanh, nhẹ', color: '#818cf8' },
  { id: 'phobert', label: 'PhoBERT', desc: 'Chính xác hơn, chậm hơn', color: '#f472b6' },
]
export const modelName = (id) => MODELS.find((m) => m.id === id)?.label || id
export const modelColor = (id) => MODELS.find((m) => m.id === id)?.color || '#64748b'

export const KEYS = ['positive', 'neutral', 'negative']
export const LABELS = {
  positive: { text: 'Tích cực', color: '#22c55e' },
  neutral: { text: 'Trung lập', color: '#3b82f6' },
  negative: { text: 'Tiêu cực', color: '#ef4444' },
}
export const tip = {
  contentStyle: { background: '#111827', border: '1px solid #334155', borderRadius: 10 },
  itemStyle: { color: '#e5e7eb' },
  labelStyle: { color: '#e5e7eb' },
}