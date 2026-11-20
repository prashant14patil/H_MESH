/**
 * SizeRecommendation.jsx
 * ----------------------
 * Garment size card driven by the height-scaled girths (cm). Shows the top and
 * trouser size, a plain fit note, and a tape-style bar per driving measurement.
 * The size chart is chosen from built-in examples, src/sizeCharts/*.json, or a
 * JSON chart the operator uploads (kept in this browser).
 */
import { useEffect, useRef, useState } from 'react'
import { recommendSizes, sizeChange, validateChart } from './sizeRecommendation.js'
import {
  BUILT_IN_CHARTS, customId, loadCustomCharts, loadSelectedChartId, saveCustomCharts, saveSelectedChartId,
} from './sizeCharts.js'

const FIT_TEXT = {
  regular: 'regular fit',
  snug: 'snug fit',
  relaxed: 'relaxed fit',
  'above chart': 'larger than the chart',
  'below chart': 'smaller than the chart',
}

const LABELS = { chest: 'Chest', waist: 'Waist', hip: 'Hip' }
const MAX_UPLOAD_BYTES = 64 * 1024

const TEMPLATE = `{
  "name": "Brand men's tops",
  "sizes": ["S", "M", "L"],
  "bounds": {
    "chest": [88, 96, 104, 112],
    "waist": [76, 84, 92, 100],
    "hip":   [92, 100, 108, 116]
  }
}`

const CHANGE_EVENT = 'sizechart-change'
const notify = () => window.dispatchEvent(new Event(CHANGE_EVENT))

/** Selected chart shared by every size view on the page, remembered in this browser. */
export function useSizeChart() {
  const [custom, setCustom] = useState(loadCustomCharts)
  const [selectedId, setSelectedId] = useState(() => loadSelectedChartId() || BUILT_IN_CHARTS[0].id)
  const all = [...BUILT_IN_CHARTS, ...custom]
  const chart = all.find((c) => c.id === selectedId) || BUILT_IN_CHARTS[0]

  // Keep every size view on the page on the same chart.
  useEffect(() => {
    const sync = () => {
      setCustom(loadCustomCharts())
      setSelectedId(loadSelectedChartId() || BUILT_IN_CHARTS[0].id)
    }
    window.addEventListener(CHANGE_EVENT, sync)
    return () => window.removeEventListener(CHANGE_EVENT, sync)
  }, [])

  const select = (id) => { setSelectedId(id); saveSelectedChartId(id); notify() }
  const add = (c) => {
    const withId = { ...c, id: customId(c), example: false }
    const next = [...custom.filter((x) => x.id !== withId.id), withId]
    setCustom(next); saveCustomCharts(next); select(withId.id)
  }
  const remove = (id) => {
    const next = custom.filter((x) => x.id !== id)
    setCustom(next); saveCustomCharts(next)
    if (selectedId === id) select(BUILT_IN_CHARTS[0].id)
    else notify()
  }
  return { chart, builtIn: BUILT_IN_CHARTS, custom, select, add, remove }
}

export default function SizeRecommendation({ measurementsCm }) {
  const charts = useSizeChart()
  const { chart } = charts
  const cm = {
    chest: measurementsCm?.chest,
    waist: measurementsCm?.waist,
    hip: measurementsCm?.hip,
  }
  const rec = recommendSizes(cm, chart)

  return (
    <section className="rounded-lg border border-amber-300 bg-amber-50 p-3 space-y-3" aria-label="Size recommendation">
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <p className="font-medium text-gray-900 text-sm">Size recommendation</p>
        <ChartPicker {...charts} />
      </div>

      {!rec.top && !rec.bottom ? (
        <p className="text-gray-600">This chart has no range for the measurements available on this run.</p>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-2">
            <SizeTile title="Top" rec={rec.top} missing={chart.bounds.chest ? 'No valid chest measurement' : 'Chart has no chest range'} />
            <SizeTile title="Trousers" rec={rec.bottom} missing={chart.bounds.waist || chart.bounds.hip ? 'No valid waist or hip' : 'Chart has no waist or hip range'} />
          </div>
          <div className="space-y-2">
            {['chest', 'waist', 'hip'].map((key) => chart.bounds[key] && (
              <TapeBar key={key} label={LABELS[key]} value={cm[key]} bounds={chart.bounds[key]}
                sizes={chart.sizes} active={rec.placements[key]?.index} />
            ))}
          </div>
        </>
      )}

      <p className="text-gray-500">
        {chart.example ? 'Example chart, not a real brand’s. ' : `Chart: ${chart.name}. `}
        Loose clothing in the video can push waist and hip up by a size.
      </p>
    </section>
  )
}

function ChartPicker({ chart, builtIn, custom, select, add, remove }) {
  const fileRef = useRef(null)
  const [error, setError] = useState(null)
  const [added, setAdded] = useState(null)

  const onFile = async (e) => {
    const file = e.target.files?.[0]
    e.target.value = ''
    setError(null); setAdded(null)
    if (!file) return
    if (file.size > MAX_UPLOAD_BYTES) { setError('That file is too large for a size chart (64 KB max).'); return }
    let raw
    try { raw = JSON.parse(await file.text()) } catch { setError('That file is not valid JSON.'); return }
    const { chart: c, error: err } = validateChart(raw)
    if (err) { setError(err); return }
    add(c); setAdded(c.name)
  }

  const isCustom = custom.some((c) => c.id === chart.id)
  return (
    <div className="flex flex-col items-end gap-1">
      <div className="flex items-center gap-1.5 flex-wrap justify-end">
        <label htmlFor="size-chart-select" className="text-gray-600">Chart</label>
        <select id="size-chart-select" value={chart.id} onChange={(e) => { setError(null); setAdded(null); select(e.target.value) }}
          className="rounded border border-amber-300 bg-white px-1.5 py-1 text-xs">
          <optgroup label="Examples">
            {builtIn.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
          </optgroup>
          {custom.length > 0 && (
            <optgroup label="Uploaded">
              {custom.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
            </optgroup>
          )}
        </select>
        <button type="button" onClick={() => fileRef.current?.click()}
          className="rounded border border-amber-400 bg-white px-2 py-1 hover:bg-amber-100">Add chart…</button>
        {isCustom && (
          <button type="button" onClick={() => remove(chart.id)}
            className="rounded border border-gray-300 bg-white px-2 py-1 text-gray-600 hover:bg-gray-100">Remove</button>
        )}
        <input ref={fileRef} type="file" accept="application/json,.json" className="hidden" onChange={onFile}
          aria-label="Upload a size chart JSON file" />
      </div>
      {error && <p className="text-red-700 max-w-xs text-right" role="alert">{error}</p>}
      {added && <p className="text-green-800" role="status">Added “{added}”. Kept in this browser only.</p>}
      <details className="text-gray-600 max-w-xs">
        <summary className="cursor-pointer text-right">Chart format</summary>
        <p className="mt-1">
          Centimetres. Each list gives the lower bound of every size, then the upper bound of the last.
          Chest sets the top size; the larger of waist and hip sets trousers.
        </p>
        <pre className="mt-1 rounded bg-white border border-amber-200 p-2 text-[10px] overflow-x-auto">{TEMPLATE}</pre>
      </details>
    </div>
  )
}

function SizeTile({ title, rec, missing }) {
  if (!rec) {
    return (
      <div className="rounded border border-amber-200 bg-white p-2">
        <p className="uppercase tracking-wide text-[10px] text-gray-500">{title}</p>
        <p className="text-gray-500">{missing}</p>
      </div>
    )
  }
  return (
    <div className="rounded border border-amber-200 bg-white p-2">
      <p className="uppercase tracking-wide text-[10px] text-gray-500">{title}</p>
      <p className="text-3xl font-bold text-gray-900 leading-tight">{rec.size}</p>
      <p className="text-gray-600">
        {FIT_TEXT[rec.fit] || rec.fit} · {LABELS[rec.by].toLowerCase()} {rec.cm.toFixed(0)} cm
        {rec.alternative && <> · or {rec.alternative}</>}
      </p>
    </div>
  )
}

function TapeBar({ label, value, bounds, sizes, active }) {
  if (value == null || !Number.isFinite(value)) return null
  const lo = bounds[0]
  const hi = bounds[bounds.length - 1]
  const pct = (v) => Math.max(0, Math.min(100, ((v - lo) / (hi - lo)) * 100))
  return (
    <div>
      <div className="flex justify-between text-gray-600">
        <span>{label} · {lo}–{hi} cm</span>
        <span className="font-mono text-gray-900">{value.toFixed(1)} cm</span>
      </div>
      <div className="relative h-6 rounded bg-amber-300 overflow-hidden"
        role="img" aria-label={`${label} ${value.toFixed(1)} cm, size ${sizes[active] ?? 'outside chart'}`}>
        {sizes.map((s, i) => (
          <div key={s}
            className={`absolute inset-y-0 border-l border-amber-900/60 pl-0.5 pt-1.5 text-[10px] font-semibold text-amber-950 overflow-hidden ${i === active ? 'bg-amber-900/20' : ''}`}
            style={{ left: `${pct(bounds[i])}%`, width: `${pct(bounds[i + 1]) - pct(bounds[i])}%` }}>
            {s}
          </div>
        ))}
        <div className="absolute -inset-y-0.5 w-[3px] bg-gray-900 rounded" style={{ left: `calc(${pct(value)}% - 1.5px)` }} />
      </div>
    </div>
  )
}

/** One-line size change for a before/after comparison, on the currently selected chart. */
export function SizeChangeLine({ beforeCm, afterCm }) {
  const { chart } = useSizeChart()
  const c = sizeChange(beforeCm, afterCm, chart)
  if (!c.top && !c.bottom) return null
  const part = (title, x) => x && (
    <span>
      {title} <b className="text-slate-900">{x.before}</b> → <b className="text-slate-900">{x.after}</b>
      {x.steps === 0
        ? <span className="text-slate-500"> (same)</span>
        : <span className={x.steps > 0 ? 'text-amber-800' : 'text-teal-700'}> ({x.steps > 0 ? '+' : '−'}{Math.abs(x.steps)} size{Math.abs(x.steps) > 1 ? 's' : ''})</span>}
    </span>
  )
  return (
    <p className="mt-2 flex flex-wrap gap-x-4 gap-y-1 rounded-lg bg-amber-50 px-3 py-2 text-sm text-slate-700" aria-label="Size change">
      <span className="text-xs font-semibold uppercase tracking-wide text-amber-800">Size</span>
      {part('Top', c.top)}
      {part('Trousers', c.bottom)}
      <span className="text-xs text-slate-500">{chart.name}</span>
    </p>
  )
}
