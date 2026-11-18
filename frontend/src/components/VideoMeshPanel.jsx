import { useEffect, useMemo, useRef, useState } from 'react'
import MeshViewer from './MeshViewer.jsx'
import { getFrameImageUrl } from '../services/captureApi.js'
import { checkMeshReadiness, getMeshBatch, meshAssetUrl, startMeshBatch, startExperimentalMeshBatch } from '../services/videoMeshApi.js'
import { saveSelection } from '../services/subjectReviewApi.js'
import { DetectionBoxes, DetectionSummary, QualityChecks, QualityReasons } from './FrameQualityDetails.jsx'
import { frameStatus, reasonText, selectionBlockReason, statusCounts } from './frameQuality.js'
import { recommendFrames } from './frameRecommendations.js'

const stem = (name) => name.replace(/\.[^.]+$/, '')
const STATUS_GROUPS = [
  ['confirmed', 'Subject confirmed — eligible for selection', 'text-green-800'],
  ['review_pending', 'Quality passed — subject review pending', 'text-amber-800'],
  ['rejected', 'Rejected by reviewer', 'text-red-700'],
  ['quality_failed', 'Quality failed — blocked', 'text-red-700'],
  ['quality_unknown', 'Quality unknown — blocked', 'text-gray-600'],
]

export default function VideoMeshPanel({ videoId, metadata, viewpointData, qualityReport, reviewSession, reviewDecisions = {}, restoredSelection = null, meshRuns = [], onReviewFrame }) {
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
  const [experimentalConfirm, setExperimentalConfirm] = useState(false)
  const [unknownInspections, setUnknownInspections] = useState({})
  const [restoreRunId, setRestoreRunId] = useState('')
  const latestFingerprint = useRef('')
  const readinessVersion = useRef(0)

  useEffect(() => {
    setSelected([])
    setReviewed([])
    setRun(null)
    setActive('')
    setError('')
    setRestoreRunId('')
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

  const hydratedFor = useRef(null)
  const lastSaved = useRef('')
  const previousDecisions = useRef(reviewDecisions)
  const [selectionSave, setSelectionSave] = useState('')
  const decisionVersion = JSON.stringify(reviewDecisions)

  // Restore the server-persisted selection for this session (or start empty for a new one).
  useEffect(() => {
    const sid = reviewSession?.session_id || null
    const restored = sid && restoredSelection?.session_id === sid ? restoredSelection : null
    const frames = restored?.frames || []
    const acknowledged = (restored?.acknowledged || []).filter(name => frames.includes(name))
    setSelected(frames); setReviewed(acknowledged)
    previousDecisions.current = reviewDecisions
    lastSaved.current = JSON.stringify([sid, frames, acknowledged])
    hydratedFor.current = sid
    setSelectionSave(restored && frames.length ? 'restored' : '')
  }, [qualityReport?.run_id, reviewSession?.session_id, restoredSelection])

  // A changed decision invalidates only that frame's acknowledgement; a rejection also deselects it.
  useEffect(() => {
    const before = previousDecisions.current || {}
    previousDecisions.current = reviewDecisions
    const changed = Object.keys(reviewDecisions).filter(name => before[name]?.decision_id !== reviewDecisions[name]?.decision_id)
    if (!changed.length) return
    setReviewed(old => old.filter(name => !changed.includes(name)))
    setSelected(old => old.filter(name => !changed.includes(name) ||
      ['confirm', 'reassign'].includes(reviewDecisions[name]?.reviewer_decision)))
  }, [decisionVersion])

  // Persist selection and acknowledgements so they survive reloads; the server re-validates eligibility.
  useEffect(() => {
    const sid = reviewSession?.session_id
    if (!sid || hydratedFor.current !== sid) return undefined
    const acknowledged = reviewed.filter(name => selected.includes(name))
    const key = JSON.stringify([sid, selected, acknowledged])
    if (lastSaved.current === key) return undefined
    setSelectionSave('saving')
    const timer = setTimeout(() => {
      saveSelection(sid, selected, acknowledged)
        .then(() => { lastSaved.current = key; setSelectionSave('saved') })
        .catch(err => setSelectionSave(`not saved: ${err.message}`))
    }, 300)
    return () => clearTimeout(timer)
  }, [reviewSession?.session_id, selected, reviewed])

  // Show the latest run from this review session after a reload.
  useEffect(() => {
    if (run || !meshRuns?.length) return undefined
    let cancelled = false
    getMeshBatch(meshRuns[0].run_id).then(next => { if (!cancelled && next.video_id === videoId) setRun(next) }).catch(() => {})
    return () => { cancelled = true }
  }, [meshRuns])
  useEffect(() => { setUnknownInspections({}); setExperimentalConfirm(false) }, [videoId, qualityReport?.run_id])

  const qualityByFrame = Object.fromEntries((qualityReport?.frames || []).map(f => [f.filename, f]))
  const mandatory = ['readability', 'person_presence', 'sharpness', 'lighting', 'full_body']
  const experimentalClass = f => {
    if (!f || !['acceptable', 'review'].includes(f.status) || f.rejections?.length) return 'blocked'
    const values = mandatory.map(key => f.checks?.[key])
    if (values.every(value => value === 'pass')) return 'passed'
    if (values.every(value => ['pass', 'unknown'].includes(value)) && values.includes('unknown')) return 'unknown'
    return 'blocked'
  }
  const experimentalPassed = frames.filter(f => experimentalClass(qualityByFrame[f.filename]) === 'passed')
  const experimentalUnknown = frames.filter(f => experimentalClass(qualityByFrame[f.filename]) === 'unknown')
  const advisory = useMemo(() => recommendFrames(frames, qualityReport, viewpointData), [frames, qualityReport, viewpointData])
  const advisoryByFrame = Object.fromEntries(advisory.recommended.map(item => [item.filename, item]))
  function eligible(f) {
    const decision = reviewDecisions[f?.filename]
    return selectionBlockReason(f, reviewSession, decision) === null
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
  const scrollTo = id => () => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  const statusByFrame = Object.fromEntries(frames.map(f => [f.filename,
    frameStatus(qualityByFrame[f.filename], reviewSession, reviewDecisions[f.filename])]))
  const status = statusCounts(frames.map(f => qualityByFrame[f.filename] || { filename: f.filename }), reviewSession, reviewDecisions)
  const counts = {
    passed: frames.length - status.quality_failed - status.quality_unknown,
    eligible: frames.filter(f => eligible(qualityByFrame[f.filename])).length,
    ...status,
  }
  const prerequisites = [
    { label: 'Recording authorization and consent', done: authorization === 'confirmed',
      detail: authorization === 'unknown' ? 'checking…' : 'not found for this recording; it is recorded when the video is uploaded.' },
    { label: 'Capture quality analysis', done: !!qualityReport,
      detail: 'not run yet — every check shows NOT RUN until it is.',
      action: { label: 'Go to “Analyze capture quality”', run: scrollTo('capture-quality') } },
    { label: 'Reference subject and review session', done: !!reviewSession,
      detail: qualityReport ? 'next: in subject review, draw a box around the person on a clear frame and click “Select reference subject”. The session is created automatically.' : 'available after quality analysis.',
      action: qualityReport ? { label: 'Go to subject review', run: scrollTo('subject-review') } : null },
    { label: 'Confirm the subject on each frame you want to use', done: counts.eligible > 0,
      detail: reviewSession ? `next: confirm frames in subject review (${counts.review_pending} pending).` : 'available after the reference subject is selected.',
      action: reviewSession ? { label: 'Go to subject review', run: scrollTo('subject-review') } : null },
    { label: 'Select 1–12 eligible frames and acknowledge their warnings', done: selected.length > 0 && selected.every(n => reviewed.includes(n)),
      detail: !counts.eligible ? 'no eligible frames yet.' : !selected.length ? `next: tick up to 12 of the ${counts.eligible} confirmed frames below.`
        : 'next: tick “I reviewed the frame and its warnings” on each selected frame.' },
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

  async function generateExperimental() {
    if (!experimentalConfirm || busy || authorization !== 'confirmed' || !qualityReport) return
    const inspections = experimentalUnknown.filter(f => unknownInspections[f.filename]?.inspected && unknownInspections[f.filename]?.acknowledged)
      .map(f => ({ filename: f.filename, source_sha256: qualityByFrame[f.filename].source_sha256, inspected: true, warning_acknowledged: true }))
    setStarting(true); setError(''); setViewerError('')
    try {
      const next = await startExperimentalMeshBatch(videoId, qualityReport.run_id, inspections)
      setRun(next); setActive(''); setExperimentalConfirm(false)
    } catch (err) {
      setError(`Could not start experimental inference: ${err.message}`)
    } finally { setStarting(false) }
  }

  async function restoreRun() {
    setStarting(true); setError(''); setViewerError('')
    try {
      const restored = await getMeshBatch(restoreRunId.trim())
      if (restored.video_id !== videoId) throw new Error('Run belongs to a different recording')
      setRun(restored); setActive('')
    } catch (err) {
      setError(`Could not open run: ${err.message}`)
    } finally { setStarting(false) }
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
        <h2 className="text-base font-semibold text-gray-800">Experimental — individual-frame meshes (diagnostic)</h2>
        <p className="text-xs text-gray-500 mt-1">Each selected frame produces its own independent HMR2 mesh. This is NOT the unified 360° body mesh — use “Step 3 — Reconstruct one 3D body mesh” above for that.</p>
      </div>
      <section aria-label="Prerequisites" className="rounded-lg border border-gray-200 p-3 text-xs space-y-2">
        <h3 className="font-semibold text-sm">Prerequisites</h3>
        <ol className="space-y-1">
          {prerequisites.map((p, i) => (
            <li key={p.label} className="flex flex-wrap items-center gap-2">
              <span className={`font-bold ${p.done ? 'text-green-700' : 'text-amber-700'}`}>{p.done ? '✓' : `${i + 1}.`}</span>
              <span className={p.done ? 'text-gray-600' : 'font-semibold text-gray-800'}>{p.label}</span>
              {!p.done && p.detail && <span className="text-gray-600">— {p.detail}</span>}
              {!p.done && p.action && <button type="button" onClick={p.action.run} className="border border-teal-600 text-teal-700 rounded px-2 py-0.5">{p.action.label}</button>}
            </li>
          ))}
        </ol>
        {qualityReport && <p className="text-gray-700" aria-label="Frame counts">
          Frames: {frames.length} · mandatory checks passed: {counts.passed} ·{' '}
          <span className="text-green-800 font-semibold">subject confirmed — eligible: {counts.eligible}</span> ·{' '}
          subject review pending: {counts.review_pending} · rejected by reviewer: {counts.rejected} ·{' '}
          quality failed: {counts.quality_failed} · quality unknown: {counts.quality_unknown} ·{' '}
          <span className="font-semibold">selected: {selected.length}/12</span>
          {reviewSession && <> · review session <span className="font-mono">{reviewSession.session_id.slice(0, 8)}</span></>}
          {selectionSave && <> · selection {selectionSave}</>}
        </p>}
      </section>
      <section aria-label="Experimental video-to-mesh" className="border border-amber-400 bg-amber-50 rounded-lg p-4 space-y-3">
        <h3 className="font-semibold">Experimental Video-to-Mesh — whole-frame inference</h3>
        <p className="text-xs">No subject-review session or subject identity verification. HMR2 may choose a different person or use the whole image. Everyone depicted must have consented to processing; the existing recording authorization and consent are checked by the server. Operator assertions are not independently verified. Results are independent predictions, not a unified scan or measurements.</p>
        <p className="text-xs">The server automatically selects every frame passing all five mandatory checks: {experimentalPassed.length}. Unknown frames are excluded unless individually inspected and acknowledged below. Failed frames remain excluded.</p>
        <details><summary className="cursor-pointer text-xs font-semibold">Default frames and quality warnings</summary>
          <ul className="text-xs list-disc pl-4">{experimentalPassed.map(f => <li key={f.filename}>{f.filename}: {(qualityByFrame[f.filename]?.warnings || []).map(reasonText).join(' ') || 'No warnings recorded'}</li>)}</ul>
        </details>
        <details><summary className="cursor-pointer text-xs font-semibold">Optional unknown-quality frames ({experimentalUnknown.length})</summary>
          <div className="grid grid-cols-[repeat(auto-fit,minmax(min(100%,14rem),1fr))] gap-2">
            {experimentalUnknown.map(f => {
              const quality = qualityByFrame[f.filename]
              const inspection = unknownInspections[f.filename] || {}
              const update = (key, value) => setUnknownInspections(old => ({ ...old, [f.filename]: { ...old[f.filename], [key]: value } }))
              return <article key={f.filename} className="bg-white p-2 border rounded text-xs space-y-2">
                <img src={getFrameImageUrl(videoId, f.filename)} alt={`Inspect ${f.filename}`} className="max-h-48 w-auto" loading="lazy" />
                <p className="font-semibold">{f.filename}</p>
                <QualityChecks frame={quality} /><QualityReasons frame={quality} />
                <label className="flex gap-2"><input type="checkbox" disabled={busy} checked={!!inspection.inspected} onChange={e => update('inspected', e.target.checked)} /> I inspected this image.</label>
                <label className="flex gap-2"><input type="checkbox" disabled={busy} checked={!!inspection.acknowledged} onChange={e => update('acknowledged', e.target.checked)} /> I acknowledge the unknown checks and warnings and choose to include this frame.</label>
              </article>
            })}
          </div>
        </details>
        <label className="flex gap-2 text-xs"><input type="checkbox" checked={experimentalConfirm} disabled={busy} onChange={e => setExperimentalConfirm(e.target.checked)} /> I authorize this experimental whole-frame inference and acknowledge the quality, subject-selection and identity risks above.</label>
        <button type="button" onClick={generateExperimental} disabled={busy || !qualityReport || authorization !== 'confirmed' || !experimentalConfirm || (!experimentalPassed.length && !experimentalUnknown.some(f => unknownInspections[f.filename]?.inspected && unknownInspections[f.filename]?.acknowledged))} className="px-4 py-2 rounded bg-amber-700 text-white font-semibold disabled:bg-gray-300">
          Generate experimental meshes
        </button>
        <p className="text-xs">Authorization and consent: {authorization === 'confirmed' ? 'operator-confirmed' : 'unavailable or not confirmed'}. Server rechecks the attestation and frame hashes on submission.</p>
      </section>
      <p className="text-sm text-gray-700">{selected.length} selected (maximum 12). Analyze quality, inspect each frame and explicitly acknowledge its warnings. Identity is not automatically verified.</p>
      {!qualityReport && <p role="alert" className="text-xs text-amber-700">Run capture quality diagnostics before selecting frames.</p>}
      {qualityReport && <p className="text-xs text-amber-700">Only frames you confirmed in subject review above can be selected. Failed or unknown mandatory quality checks remain blocked. Human confirmation is not biometric identity verification.</p>}
      {qualityReport && <div className="rounded-lg border bg-gray-50 p-3 text-xs space-y-1">
        <p className="font-semibold">Advisory review candidates ({advisory.recommended.length}) — suggestions for which frames to review first</p>
        <p>These are <strong>not</strong> confirmed, eligible or selected. A candidate becomes eligible only after you confirm its subject in subject review. Prioritized by mandatory quality, extractor blur score, unvalidated viewpoint hints and time spacing; identity is not verified.</p>
        <div className="flex flex-wrap gap-1">{advisory.recommended.length ? advisory.recommended.map(item => {
          const s = statusByFrame[item.filename]
          return <button key={item.filename} type="button" onClick={() => onReviewFrame?.(item.filename)}
            className={`border rounded px-1.5 py-0.5 ${s?.key === 'confirmed' ? 'border-green-600 text-green-800' : s?.key === 'rejected' ? 'border-red-300 text-red-700' : 'border-gray-300'}`}
            title={s?.label}>
            {item.filename} ({item.bucket.replaceAll('_', ' ')} hint){s?.key === 'confirmed' ? ' ✓' : s?.key === 'rejected' ? ' ✗' : ''}
          </button>
        }) : 'No candidates meet the screening criteria.'}</div>
        <p>Click a candidate to open it in subject review. Viewpoint coverage remains unconfirmed; unknown views are not proof of missing coverage.</p>
      </div>}
      {selected.length > 0 && blurry * 2 >= selected.length && (
        <div role="alert" className="rounded-lg bg-amber-50 border border-amber-200 p-3 text-xs text-amber-800">
          {blurry === selected.length ? 'All selected frames are blurry or have unknown blur status.' : 'Most selected frames are blurry or have unknown blur status.'} Results may be unreliable; consider recording again.
        </div>
      )}
      {STATUS_GROUPS.map(([group, title, color]) => {
        const groupFrames = frames.filter(frame => statusByFrame[frame.filename]?.key === group)
        if (!groupFrames.length) return null
        const body = <>
          {group === 'review_pending' && <p className="text-xs text-amber-800">Quality screening passed, but these frames are blocked until you confirm their subject against the reference subject. A quality pass is not subject approval or identity verification.</p>}
          <div className="grid grid-cols-[repeat(auto-fit,minmax(min(100%,14rem),1fr))] gap-3 items-stretch">
        {groupFrames.map(frame => {
          const checked = selected.includes(frame.filename)
          const quality = qualityByFrame[frame.filename]
          const decision = reviewDecisions[frame.filename]
          const reason = selectionBlockReason(quality, reviewSession, decision, {
            busy, limitReached: !checked && selected.length >= 12,
          })
          const allowed = reason === null
          return (
            <article key={frame.filename} className="min-w-0 rounded-lg border border-gray-200 p-3 text-xs text-gray-700 space-y-2 [overflow-wrap:anywhere] [word-break:break-word]">
              <div className="flex justify-center bg-gray-100 rounded">
                <div className="relative inline-block max-w-full">
                  <img src={getFrameImageUrl(videoId, frame.filename)} alt={frame.filename} className="block max-w-full max-h-36 w-auto" loading="lazy" />
                  <DetectionBoxes frame={quality} />
                </div>
              </div>
              <label className="flex items-start gap-2 font-semibold">
                <input type="checkbox" className="mt-0.5 shrink-0" checked={checked} disabled={busy || (!checked && !allowed)} onChange={() => toggle(frame.filename)} aria-describedby={`selection-reason-${frame.filename}`} />
                <span>{frame.filename}</span>
              </label>
              <p className="text-gray-600">{frame.timestamp_sec?.toFixed(2) ?? '—'}s</p>
              {qualityReport && <p className="text-gray-600">{advisoryByFrame[frame.filename]
                ? `Advisory candidate: ${advisoryByFrame[frame.filename].bucket.replaceAll('_', ' ')} hint; quality screening passed. Review required.`
                : `Not prioritized: ${advisory.excluded[frame.filename] || 'No advisory evidence available.'}`}</p>}
              <DetectionSummary frame={quality} />
              <p id={`selection-reason-${frame.filename}`} className={allowed || checked ? 'font-semibold text-green-800' : 'font-semibold text-amber-800'}>
                {statusByFrame[frame.filename]?.label}
                {!allowed && !checked && <span className="block font-normal">Cannot select: {reason}</span>}
                {statusByFrame[frame.filename]?.action && <span className="block font-normal">Next: {statusByFrame[frame.filename].action}</span>}
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
        </>
        const heading = <span className={`text-sm font-semibold ${color}`}>{title} ({groupFrames.length})</span>
        // Actionable groups are expanded; blocked and rejected frames are collapsed but still listed.
        return ['confirmed', 'review_pending'].includes(group)
          ? <div key={group} className="space-y-2" aria-label={title}><h3>{heading}</h3>{body}</div>
          : <details key={group} className="space-y-2" aria-label={title}><summary className="cursor-pointer">{heading}</summary>{body}</details>
      })}
      <div className="rounded-lg border bg-gray-50 p-3 text-xs space-y-2" aria-label="Submission readiness">
        <h3 className="font-semibold text-sm">Submission readiness</h3>
        <p>Authorization and consent: {authorization === 'confirmed' ? 'operator-confirmed (not independently verified)' : 'not confirmed or unavailable'}.</p>
        <p>Review session: {reviewSession ? <span className="font-mono">{reviewSession.session_id}</span> : 'missing — select a reference subject in subject review'} · Selected: {selected.length} · Warning acknowledgements: {reviewed.filter(name => selected.includes(name)).length}/{selected.length}</p>
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
      {meshRuns.length > 0 && <div className="flex flex-wrap gap-2 text-xs items-center" aria-label="Previous runs">
        <span>Saved runs for this review session:</span>
        {meshRuns.map(item => <button key={item.run_id} type="button" disabled={busy}
          onClick={() => getMeshBatch(item.run_id).then(next => { setRun(next); setActive('') }).catch(err => setError(`Could not open run: ${err.message}`))}
          className={`border rounded px-2 py-1 ${run?.run_id === item.run_id ? 'border-teal-600 text-teal-700' : ''}`}>
          {item.run_id.slice(0, 8)} · {String(item.status).replaceAll('_', ' ')} · {item.completed}/{item.frames.length} meshes
        </button>)}
      </div>}
      <div className="flex flex-wrap gap-2 text-xs items-center">
        <label htmlFor="restore-mesh-run">Open an existing mesh run by ID</label>
        <input id="restore-mesh-run" className="border rounded p-2" value={restoreRunId} onChange={e => setRestoreRunId(e.target.value)} disabled={busy} />
        <button type="button" className="border rounded px-3 py-2 disabled:text-gray-400" onClick={restoreRun} disabled={busy || !restoreRunId.trim()}>Open run</button>
      </div>
      {error && <p role="alert" className="text-xs text-red-700">{error}</p>}
      {run && (
        <div className="space-y-3">
          <p className="text-sm text-gray-700">{run.mode === 'experimental_whole_frame' ? 'Experimental whole-frame ' : ''}Run {run.run_id.slice(0, 8)}: {run.status.replaceAll('_', ' ')} · {run.frames.filter(f => f.status === 'completed').length} successful · {run.frames.filter(f => f.status === 'failed').length} failed · {run.frames.length} total</p>
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
