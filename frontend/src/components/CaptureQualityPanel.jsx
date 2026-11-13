import { useEffect, useState } from 'react'
import { DetectionSummary, QualityChecks, QualityReasons } from './FrameQualityDetails.jsx'

const base = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

async function json(path, options) {
  const response = await fetch(`${base}${path}`, options)
  if (!response.ok) {
    let detail = `HTTP ${response.status}`
    try { detail = (await response.json()).detail || detail } catch { /* HTTP status remains */ }
    throw new Error(detail)
  }
  return response.json()
}

export default function CaptureQualityPanel({ videoId, onReport, restoredReport: report, reviewSession, reviewDecisions = {} }) {
  const [job, setJob] = useState(null)
  const [error, setError] = useState('')
  const [frame, setFrame] = useState('')
  const [starting, setStarting] = useState(false)

  useEffect(() => {
    if (!job?.run_id || job.status !== 'processing') return undefined
    let cancelled = false
    let timer
    async function poll() {
      try {
        const next = await json(`/api/capture-quality/runs/${encodeURIComponent(job.run_id)}`)
        if (cancelled) return
        if (next.status === 'completed') {
          const result = await json(`/api/capture-quality/runs/${encodeURIComponent(job.run_id)}/report`)
          if (!cancelled) { onReport(result); setJob(next) }
        } else if (next.status === 'failed') {
          setError(next.detail || 'Quality analysis failed')
          setJob(next)
        } else {
          timer = setTimeout(poll, 1500)
        }
      } catch (err) {
        if (!cancelled) { setError(err.message); timer = setTimeout(poll, 4000) }
      }
    }
    timer = setTimeout(poll, 1000)
    return () => { cancelled = true; clearTimeout(timer) }
  }, [job?.run_id, job?.status])

  async function start() {
    setStarting(true); setError(''); setFrame(''); onReport(null)
    try {
      const next = await json('/api/capture-quality/runs', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_id: videoId }),
      })
      setJob(next)
    } catch (err) { setError(err.message) } finally { setStarting(false) }
  }

  const detail = report?.frames?.find(item => item.filename === frame)
  return (
    <section className="rounded-xl border border-gray-200 bg-white p-4 space-y-3">
      <h3 className="font-semibold text-gray-800">Capture quality diagnostics</h3>
      <p className="text-xs text-gray-500">Screening only. Person identity, bystanders and 360° coverage cannot be confirmed by these heuristics.</p>
      <button type="button" onClick={start} disabled={starting || job?.status === 'processing'}
        className="rounded-lg bg-teal-600 px-3 py-2 text-white text-sm disabled:bg-gray-300">
        {starting || job?.status === 'processing' ? 'Analyzing…' : 'Analyze capture quality'}
      </button>
      {error && <p role="alert" className="text-red-700 text-xs">{error}</p>}
      {report && <>
        <p className="text-xs text-gray-600">
          {report.summary.extracted_frames} frames · {report.summary.extractor_non_blurry} not flagged blurry · {report.summary.extractor_blurry} blurry · {report.summary.rejected_duplicates} duplicate candidates rejected · {report.summary.full_body_visible} full-body hints · {report.summary.possible_multiple_people} multi-person flags (HOG can miss people)
        </p>
        <p className="text-xs text-amber-700">Viewpoint coverage: {report.viewpoint_coverage.status}. {report.viewpoint_coverage.note}</p>
        <p className="text-xs text-gray-600">Unvalidated hints: {Object.entries(report.viewpoint_coverage.counts).map(([label, count]) => `${label}: ${count}`).join(' · ') || 'none'}</p>
        <p className="text-xs text-amber-700">Recommended frames: {report.summary.recommended_frames.length ? report.summary.recommended_frames.join(', ') : 'none — subject identity is unverified'}</p>
        {report.rejected_candidates.length > 0 && <details className="text-xs text-gray-600">
          <summary>Rejected extraction candidates ({report.rejected_candidates.length})</summary>
          <ul>{report.rejected_candidates.map((item, index) => <li key={index}>{item.timestamp_sec ?? 'unknown'}s: {item.reason || 'unknown'}</li>)}</ul>
        </details>}
        <div className="flex flex-wrap gap-2">
          {report.frames.map(item => (
            <button key={item.filename} type="button" onClick={() => setFrame(item.filename)}
              className={`rounded px-2 py-1 text-xs ${item.status === 'acceptable' ? 'bg-green-100 text-green-800' : item.status === 'rejected' ? 'bg-red-100 text-red-800' : 'bg-amber-100 text-amber-800'}`}>
              {item.filename}: {item.status}
            </button>
          ))}
        </div>
        {detail && <div className="min-w-0 rounded bg-gray-50 p-3 text-xs space-y-2 [overflow-wrap:anywhere] [word-break:break-word]">
          <p className="font-semibold">{detail.filename} · {detail.timestamp_sec ?? 'unknown'}s</p>
          <DetectionSummary frame={detail} />
          <QualityChecks frame={detail} session={reviewSession} decision={reviewDecisions[detail.filename]} />
          <QualityReasons frame={detail} />
          <p>A screening pass is not verified identity or reconstruction accuracy. Review the subject manually; failed or unknown mandatory checks remain blocked.</p>
          <p>Extractor blur: {String(detail.extractor_quality?.is_blurry)}; ROI Laplacian: {detail.image_metrics?.roi_laplacian_variance?.toFixed(1) ?? 'unknown'}; lighting mean: {detail.image_metrics?.roi_mean_brightness?.toFixed(1) ?? 'unknown'}</p>
        </div>}
      </>}
    </section>
  )
}
