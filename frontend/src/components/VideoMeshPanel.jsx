import { useEffect, useMemo, useRef, useState } from 'react'
import MeshViewer from './MeshViewer.jsx'
import { getFrameImageUrl } from '../services/captureApi.js'
import { checkMeshReadiness, getMeshBatch, meshAssetUrl, startMeshBatch } from '../services/videoMeshApi.js'
import { DetectionBoxes, DetectionSummary, QualityChecks, QualityReasons } from './FrameQualityDetails.jsx'
import { qualityGroup, reasonText, validDecision } from './frameQuality.js'

const stem = (name) => name.replace(/\.[^.]+$/, '')

export default function VideoMeshPanel({ videoId, metadata, viewpointData, qualityReport, reviewSession, reviewDecisions = {}, onReviewFrame }) {
  const frames = metadata?.frames || []
  const viewpoints = useMemo(() => Object.fromEntries((viewpointData?.frames || []).map(p => [p.frame, p])), [viewpointData])
  const [selected, setSelected] = useState([])
  const [reviewed, setReviewed] = useState([])
  const [run, setRun] = useState(null)
  const [active, setActive] = useState('')
  const [error, setError] = useState('')
  const [viewerError, setViewerError] = useState('')
  const [starting, setStarting] = useState(false)
  const [authorization, setAuthorization] = useState('unknown')
  const [preflight, setPreflight] = useState(null)
  const [confirmation, setConfirmation] = useState(false)
  const latestFingerprint = useRef('')
  const readinessVersion = useRef(0)

  useEffect(() => {
    setSelected([])
    setReviewed([])
    setRun(null)
    setActive('')
    setError('')
  }, [videoId]) // A new capture starts a separate selection; pose results are already loaded.

  useEffect(() => {
    if (!run?.run_id || run.status !== 'processing') return undefined
    let cancelled = false
    let timer
    const poll = async () => {
      try {
        const next = await getMeshBatch(run.run_id)
        if (!cancelled) {
          setRun(next)
          setError('')
          if (next.status === 'processing') timer = setTimeout(poll, 2000)
        }
      } catch (err) {
        if (!cancelled) {
          setError(`Status unavailable; retrying: ${err.message}`)
          timer = setTimeout(poll, 4000)
        }
      }
    }
    timer = setTimeout(poll, 1000)
    return () => { cancelled = true; clearTimeout(timer) }
  }, [run?.run_id, run?.status])

  useEffect(() => {
    let cancelled = false
    setAuthorization('unknown')
    if (!videoId) return undefined
    const base = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')
    fetch(`${base}/api/capture/${encodeURIComponent(videoId)}/authorization`)
      .then(response => { if (!cancelled) setAuthorization(response.ok ? 'confirmed' : 'unavailable') })
      .catch(() => { if (!cancelled) setAuthorization('unavailable') })
    return () => { cancelled = true }
  }, [videoId, qualityReport?.run_id])

  useEffect(() => { setSelected([]); setReviewed([]) }, [qualityReport?.run_id, reviewSession?.session_id])
  const decisionVersion = JSON.stringify(reviewDecisions)
  useEffect(() => { setReviewed([]) }, [decisionVersion])

  const qualityByFrame = Object.fromEntries((qualityReport?.frames || []).map(f => [f.filename, f]))
  function eligible(f) {
    const decision = reviewDecisions[f?.filename]
    return !!reviewSession && qualityGroup(f) === 'passed' &&
      ['confirm', 'reassign'].includes(decision?.reviewer_decision) &&
      decision?.identity_status === 'human_confirmed_not_biometric' && validDecision(f, reviewSession, decision)
  }
  const busy = starting || run?.status === 'processing'
  const selectedFrames = frames.filter(f => selected.includes(f.filename))
  const blurry = selectedFrames.filter(f => f.is_blurry !== false).length
  const completed = run?.frames?.filter(f => f.status === 'completed') || []
  const current = completed.find(f => f.filename === active) || completed[0]
  const currentView = current ? viewpoints[stem(current.filename)] : null
  const fingerprint = JSON.stringify([videoId, qualityReport?.run_id, reviewSession?.session_id, selected, reviewed, decisionVersion])
  if (latestFingerprint.current !== fingerprint) {
    latestFingerprint.current = fingerprint
    readinessVersion.current += 1
  }
  useEffect(() => { setPreflight(null); setConfirmation(false) }, [fingerprint])
  const localBlockers = [
    ...(!qualityReport ? ['A completed quality report is required.'] : []),
    ...(!reviewSession ? ['Select a reference subject and create or resume a review session.'] : []),
    ...(authorization !== 'confirmed' ? ['Operator-confirmed authorization and consent are unavailable or still being checked.'] : []),
    ...(selected.length === 0 ? ['Select at least one individually reviewed frame.'] : []),
    ...selected.filter(name => !eligible(qualityByFrame[name])).map(name => `${name}: mandatory checks or subject review are incomplete.`),
    ...selected.filter(name => !reviewed.includes(name)).map(name => `${name}: acknowledge the frame and its warnings.`),
  ]
  const preflightCurrent = preflight?.fingerprint === fingerprint
  const locallyReady = !busy && localBlockers.length === 0

  async function prepareConfirmation() {
    const checkedFingerprint = fingerprint
    const checkedVersion = readinessVersion.current
    setStarting(true); setError(''); setPreflight(null); setConfirmation(false)
    try {
      const result = await checkMeshReadiness(videoId, selected, qualityReport?.run_id, reviewed, reviewSession?.session_id)
      if (latestFingerprint.current === checkedFingerprint && readinessVersion.current === checkedVersion) setPreflight({ ...result, fingerprint: checkedFingerprint })
    } catch (err) {
      if (latestFingerprint.current === checkedFingerprint && readinessVersion.current === checkedVersion) setError(`Server readiness check failed: ${err.message}`)
    } finally { setStarting(false) }
  }

  async function generate() {
    if (!locallyReady || !preflightCurrent || !preflight.ready || !confirmation) return
    setStarting(true)
    setError('')
    setViewerError('')
    try {
      const next = await startMeshBatch(videoId, selected, qualityReport?.run_id, reviewed, reviewSession?.session_id, true)
      setPreflight(null); setConfirmation(false)
      setActive('')
      setRun(next)
    } catch (err) {
      setPreflight(null); setConfirmation(false)
      setError(`Could not start reconstruction: ${err.message}`)
    } finally {
      setStarting(false)
    }
  }

  function toggle(name) {
    setReviewed(previous => previous.filter(x => x !== name))
    setSelected(previous => previous.includes(name)
      ? previous.filter(x => x !== name)
      : previous.length < 12 ? [...previous, name] : previous)
  }

  return (
    <section className="bg-white rounded-2xl border border-gray-200 shadow-sm p-5 space-y-4">
      <div>
        <h2 className="text-base font-semibold text-gray-800">Step 3 — 3D Mesh Reconstruction</h2>
        <p className="text-xs text-gray-500 mt-1">Individual 3D Meshes — Multi-view reconstruction pending.</p>
        <p className="text-xs text-gray-500 mt-1">Each mesh is an independent prediction, not a unified scan or validated body measurement.</p>
      </div>
      <p className="text-sm text-gray-700">{selected.length} selected (maximum 12). Analyze quality, inspect each frame and explicitly acknowledge its warnings. Identity is not automatically verified.</p>
      {!qualityReport && <p role="alert" className="text-xs text-amber-700">Run capture quality diagnostics before selecting frames.</p>}
      {qualityReport && <p className="text-xs text-amber-700">Review and confirm each subject box above. Failed or unknown mandatory quality checks remain blocked. Human confirmation is not biometric identity verification.</p>}
      {selected.length > 0 && blurry * 2 >= selected.length && (
        <div role="alert" className="rounded-lg bg-amber-50 border border-amber-200 p-3 text-xs text-amber-800">
          {blurry === selected.length ? 'All selected frames are blurry or have unknown blur status.' : 'Most selected frames are blurry or have unknown blur status.'} Results may be unreliable; consider recording again.
        </div>
      )}
      {[
        ['passed', 'Mandatory checks passed — subject review required'],
        ['blocked', 'Failed, rejected or warning on mandatory checks — blocked'],
        ['unknown', 'Mandatory checks unknown — blocked'],
      ].map(([group, title]) => {
        const groupFrames = frames.filter(frame => qualityGroup(qualityByFrame[frame.filename]) === group)
        return <div key={group} className="space-y-2">
          <h3 className="text-sm font-semibold text-gray-800">{title} ({groupFrames.length})</h3>
          {group === 'passed' && <p className="text-xs text-amber-800">A quality pass is not subject approval or identity verification.</p>}
          <div className="grid grid-cols-[repeat(auto-fit,minmax(min(100%,14rem),1fr))] gap-3 items-stretch">
        {groupFrames.map(frame => {
          const checked = selected.includes(frame.filename)
          const quality = qualityByFrame[frame.filename]
          const decision = reviewDecisions[frame.filename]
          const allowed = eligible(quality)
          return (
            <article key={frame.filename} className="min-w-0 rounded-lg border border-gray-200 p-3 text-xs text-gray-700 space-y-2 [overflow-wrap:anywhere] [word-break:break-word]">
              <div className="flex justify-center bg-gray-100 rounded">
                <div className="relative inline-block max-w-full">
                  <img src={getFrameImageUrl(videoId, frame.filename)} alt={frame.filename} className="block max-w-full max-h-36 w-auto" loading="lazy" />
                  <DetectionBoxes frame={quality} />
                </div>
              </div>
              <label className="flex items-start gap-2 font-semibold">
                <input type="checkbox" className="mt-0.5 shrink-0" checked={checked} disabled={busy || (!checked && (!allowed || selected.length >= 12))} onChange={() => toggle(frame.filename)} />
                <span>{frame.filename}</span>
              </label>
              <p className="text-gray-600">{frame.timestamp_sec?.toFixed(2) ?? '—'}s</p>
              <DetectionSummary frame={quality} />
              <p className={allowed ? 'font-semibold text-green-800' : 'font-semibold text-amber-800'}>
                {allowed ? 'Subject reviewed; eligible for selection' : quality ? 'Blocked: quality checks or subject review required' : 'Blocked: quality report required'}
              </p>
              <QualityChecks frame={quality} session={reviewSession} decision={decision} />
              {quality && <details>
                <summary className="cursor-pointer font-semibold">Warnings and reasons</summary>
                <QualityReasons frame={quality} />
              </details>}
              {quality && <button type="button" onClick={() => onReviewFrame?.(frame.filename)} className="text-teal-700 underline font-semibold">
                Review this frame and detection boxes
              </button>}
              {checked && <label className="flex items-start gap-2">
                <input aria-label={`Review ${frame.filename}`} type="checkbox" className="mt-0.5 shrink-0" disabled={busy} checked={reviewed.includes(frame.filename)} onChange={e => setReviewed(old => e.target.checked ? [...old, frame.filename] : old.filter(x => x !== frame.filename))} />
                <span>I reviewed the frame and its warnings; subject identity remains unverified.</span>
              </label>}
            </article>
          )
        })}
          </div>
        </div>
      })}
      <div className="rounded-lg border bg-gray-50 p-3 text-xs space-y-2" aria-label="Submission readiness">
        <h3 className="font-semibold text-sm">Submission readiness</h3>
        <p>Authorization and consent: {authorization === 'confirmed' ? 'operator-confirmed (not independently verified)' : 'not confirmed or unavailable'}.</p>
        <p>Reference session: {reviewSession ? 'selected' : 'missing'} · Selected: {selected.length} · Warning acknowledgements: {reviewed.filter(name => selected.includes(name)).length}/{selected.length}</p>
        {localBlockers.length ? <ul className="list-disc pl-4">{localBlockers.map(item => <li key={item}>{item}</li>)}</ul>
          : <p>Local prerequisites appear complete. The server must verify current files and decisions.</p>}
        {preflightCurrent && !preflight.ready && <div role="alert"><p>Server check blocked submission:</p><ul className="list-disc pl-4">{preflight.blockers.map(item => <li key={item}>{item}</li>)}</ul>
          {(preflight.unevaluated || []).length > 0 && <><p>Not yet evaluated:</p><ul className="list-disc pl-4">{preflight.unevaluated.map(item => <li key={item}>{item}</li>)}</ul></>}
        </div>}
        <button type="button" onClick={prepareConfirmation} disabled={busy} className="px-3 py-2 border rounded disabled:text-gray-400">
          {starting ? 'Checking…' : 'Check server readiness'}
        </button>
      </div>
      {locallyReady && preflightCurrent && preflight.ready && <section aria-label="Final inference confirmation" className="rounded-lg border border-amber-300 bg-amber-50 p-3 text-xs space-y-2">
        <h3 className="font-semibold text-sm">Final confirmation before HMR2 inference</h3>
        <p>Recording: {videoId} · Quality report: {qualityReport.run_id} · Review session: {reviewSession.session_id}</p>
        <p>Authorization and consent are operator-confirmed, not independently verified. Subject confirmation is not biometric identification.</p>
        <ul className="list-disc pl-4">{selected.map(name => <li key={name}>{name} — human-reviewed; warnings: {(qualityByFrame[name]?.warnings || []).map(reasonText).join(' ') || 'none'}</li>)}</ul>
        <p>Each frame produces an independent prediction, not a fused scan or validated measurement. The server will recheck all prerequisites when submitted.</p>
        <label className="flex gap-2"><input type="checkbox" checked={confirmation} onChange={e => setConfirmation(e.target.checked)} disabled={busy} /> I confirm these selected frames and understand that real HMR2 inference will start.</label>
        <div className="flex gap-2">
          <button type="button" onClick={() => { setPreflight(null); setConfirmation(false) }} disabled={busy} className="px-3 py-2 border rounded">Cancel</button>
          <button type="button" onClick={generate} disabled={!locallyReady || !confirmation}
            className="px-4 py-2 rounded-xl bg-teal-600 text-white font-semibold disabled:bg-gray-300">
            {busy ? 'Generating meshes…' : 'Confirm and generate 3D meshes'}
          </button>
        </div>
      </section>}
      {error && <p role="alert" className="text-xs text-red-700">{error}</p>}
      {run && (
        <div className="space-y-3">
          <p className="text-sm text-gray-700">Run {run.run_id.slice(0, 8)}: {run.status.replaceAll('_', ' ')} · {run.frames.filter(f => ['completed', 'failed'].includes(f.status)).length}/{run.frames.length} processed</p>
          <ul className="text-xs space-y-1">
            {run.frames.map(f => (
              <li key={f.filename} className="flex flex-wrap gap-2">
                <span className="font-medium">{f.filename}</span>
                <span className={f.status === 'failed' ? 'text-red-700' : 'text-gray-600'}>{f.status}</span>
                {f.detail && <span className="text-red-700">{f.detail}</span>}
              </li>
            ))}
          </ul>
          {completed.length > 0 && (
            <div className="space-y-3">
              <label className="text-sm text-gray-700">Mesh
                <select className="ml-2 border rounded p-1" value={current?.filename || ''} onChange={e => { setActive(e.target.value); setViewerError('') }}>
                  {completed.map(f => <option key={f.filename} value={f.filename}>{f.filename}</option>)}
                </select>
              </label>
              <p className="text-xs text-gray-500">
                {current.filename} · viewpoint hint: {currentView?.viewpoint || 'unknown'} · heuristic score: {currentView?.confidence ?? 'unavailable'} · {current.metadata?.vertices ?? '—'} vertices · {current.metadata?.faces ?? '—'} faces
              </p>
              {currentView?.reason && <p className="text-xs text-amber-700">Unvalidated viewpoint hint: {currentView.reason.replaceAll('_', ' ')}</p>}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                <img src={meshAssetUrl(run.run_id, 'frame', current.filename)} alt={`Source ${current.filename}`} className="w-full rounded-lg border object-contain bg-gray-100" />
                <MeshViewer key={`${run.run_id}-${current.filename}`} meshUrl={meshAssetUrl(run.run_id, 'mesh', current.filename)} onError={setViewerError} />
              </div>
              {viewerError && <p role="alert" className="text-xs text-red-700">Mesh viewer: {viewerError}</p>}
            </div>
          )}
        </div>
      )}
    </section>
  )
}
