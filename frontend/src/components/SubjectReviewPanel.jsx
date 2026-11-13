import { useEffect, useRef, useState } from 'react'
import { getFrameImageUrl } from '../services/captureApi.js'
import { boxStyle, DetectionBoxes, DetectionSummary, QualityChecks, QualityReasons } from './FrameQualityDetails.jsx'
import { association, validDecision } from './frameQuality.js'

const base = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

async function api(path, options) {
  const response = await fetch(`${base}/api/subject-review${path}`, options)
  const data = await response.json()
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `HTTP ${response.status}`)
  return data
}

export default function SubjectReviewPanel({ videoId, report, onSession, onDecision, requestedFrame, decisions = {} }) {
  const [session, setSession] = useState(null)
  const [filename, setFilename] = useState('')
  const [box, setBox] = useState(null)
  const [suggestion, setSuggestion] = useState(null)
  const [error, setError] = useState('')
  const [saving, setSaving] = useState(false)
  const [resumeId, setResumeId] = useState('')
  const [resumeNotice, setResumeNotice] = useState('')
  const start = useRef(null)
  const image = useRef(null)

  useEffect(() => {
    setSession(null); setFilename(report?.frames?.[0]?.filename || '')
    setBox(null); setSuggestion(null); setError(''); onSession(null)
  }, [report?.run_id])

  useEffect(() => {
    if (requestedFrame?.name && report?.frames?.some(f => f.filename === requestedFrame.name)) {
      setFilename(requestedFrame.name); setBox(null)
    }
  }, [requestedFrame, report])

  useEffect(() => {
    if (!session || !filename) return undefined
    let cancelled = false
    setBox(null); setSuggestion(null); setError('')
    api(`/sessions/${session.session_id}/suggestions/${encodeURIComponent(filename)}`)
      .then(data => { if (!cancelled) setSuggestion(data) })
      .catch(err => { if (!cancelled) setError(err.message) })
    return () => { cancelled = true }
  }, [session?.session_id, filename])

  function point(event) {
    const rect = image.current.getBoundingClientRect()
    return [Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
      Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height))]
  }

  async function create() {
    setSaving(true)
    try {
      const result = await api('/sessions', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_id: videoId, quality_run_id: report.run_id, reference_frame: filename, bbox: box }) })
      setSession(result); onSession(result); setError('')
    } catch (err) { setError(err.message) } finally { setSaving(false) }
  }

  async function resume(event) {
    event.preventDefault()
    setSaving(true); setError(''); setResumeNotice('')
    try {
      const params = new URLSearchParams({ video_id: videoId, quality_run_id: report.run_id })
      const result = await api(`/sessions/${encodeURIComponent(resumeId.trim())}/resume?${params}`)
      setSession(result.session)
      onSession(result.session, result.decisions)
      setFilename(result.session.reference_frame)
      setBox(null)
      setResumeNotice(result.unavailable_frames.length
        ? `Unavailable or stale frames: ${result.unavailable_frames.join(', ')}. Re-analyze or review again.`
        : 'Session restored. Warning acknowledgements must be made again.')
    } catch (err) { setError(err.message) } finally { setSaving(false) }
  }

  async function decide(decision) {
    setSaving(true)
    try {
      const result = await api(`/sessions/${session.session_id}/decisions`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename, decision, bbox: decision === 'reject' ? null : box }),
      })
      setSuggestion(old => ({ ...old, decision: result, identity_status: result.identity_status }))
      onDecision(result); setBox(null); setError('')
    } catch (err) { setError(err.message) } finally { setSaving(false) }
  }

  if (!report) return null
  const frame = report.frames.find(f => f.filename === filename)
  const saved = validDecision(frame, session, decisions[filename]) ? decisions[filename] : null
  const state = association(frame, session, saved)
  const savedBox = saved?.selected_subject_box
  const unresolved = report.frames.filter(f => {
    const state = association(f, session, decisions[f.filename])
    return state.status !== 'PASS' && state.status !== 'FAIL'
  })
  const ambiguous = unresolved.filter(f => f.warnings?.includes('subject_association_ambiguous') || f.warnings?.includes('subject_continuity_uncertain'))
  const missing = unresolved.filter(f => f.person_detected === false || f.warnings?.includes('subject_association_unknown'))
  return (
    <section id="subject-review" className="min-w-0 rounded-xl border border-gray-200 bg-white p-4 space-y-3 [overflow-wrap:anywhere] [word-break:break-word]">
      <h3 className="font-semibold">Subject-targeted manual review</h3>
      <ol className="text-xs text-gray-700 list-decimal pl-4 space-y-1">
        <li>Choose a clear reference frame, drag a box around the intended person, and select the reference subject.</li>
        <li>Choose each frame, inspect the detector boxes, and draw or choose a subject box. Confirm, reassign, or reject each frame explicitly.</li>
        <li>Return to Step 3 to select eligible frames and acknowledge their warnings.</li>
      </ol>
      <p className="text-xs text-amber-800">Candidates are unvalidated, not biometric identification. The reviewed crop must contain the pose landmark extent and cannot contain multiple candidate centers. Failed or unknown mandatory checks remain blocked; reject or recapture if uncertain.</p>
      {!session && <form onSubmit={resume} className="flex flex-wrap gap-2 items-end">
        <label className="text-xs">Existing review session ID
          <input value={resumeId} onChange={e => setResumeId(e.target.value)} className="block border rounded p-1" required />
        </label>
        <button type="submit" disabled={saving} className="border rounded px-2 py-1 text-xs">Resume session</button>
      </form>}
      {session && <p className="text-xs font-semibold text-teal-800">Reference subject selected in {session.reference_frame}. Session ID: {session.session_id}. Selecting the reference does not confirm any frame.</p>}
      {resumeNotice && <p className="text-xs text-amber-800" role="status">{resumeNotice}</p>}
      <div className="text-xs space-y-1">
        <p>Ambiguous association ({ambiguous.length}): {ambiguous.map(f => f.filename).join(', ') || 'none'}</p>
        <p>Missing or unknown association evidence ({missing.length}): {missing.map(f => f.filename).join(', ') || 'none'}</p>
        <p>These categories can overlap. All unconfirmed frames require individual review.</p>
      </div>
      <label className="text-sm">Frame <select value={filename} disabled={saving} onChange={e => { setFilename(e.target.value); setBox(null) }} className="border rounded p-1 max-w-full">
        {report.frames.map(f => {
          const status = association(f, session, decisions[f.filename])
          return <option key={f.filename} value={f.filename}>{f.filename} — {status.status === 'PASS' ? 'confirmed' : status.status === 'FAIL' ? 'rejected' : 'review required'}</option>
        })}
      </select></label>
      {filename && <div className="relative inline-block max-w-full touch-none" style={{ maxWidth: 640 }}
        onPointerDown={e => { if (saving) return; start.current = point(e); e.currentTarget.setPointerCapture(e.pointerId) }}
        onPointerUp={e => { if (!start.current) return; const end = point(e); const a = start.current; start.current = null
          setBox([Math.min(a[0], end[0]), Math.min(a[1], end[1]), Math.max(a[0], end[0]), Math.max(a[1], end[1])]) }}
        onPointerCancel={() => { start.current = null }}>
        <img ref={image} src={`${getFrameImageUrl(videoId, filename)}?sha256=${encodeURIComponent(frame?.source_sha256 || '')}`} alt={`Review ${filename}`} draggable="false" className="block max-w-full" />
        <DetectionBoxes frame={frame} />
        {session?.reference_frame === filename && <div className="absolute border-2 border-blue-600 pointer-events-none" style={boxStyle(session.reference_box)} />}
        {savedBox && <div className="absolute border-2 border-green-600 pointer-events-none" style={boxStyle(savedBox)} />}
        {box && <div className="absolute border-2 border-teal-500 border-dashed pointer-events-none" style={boxStyle(box)} />}
      </div>}
      <p className="text-xs">Amber solid: retained or unfiltered candidate · Magenta dashed: NMS-suppressed candidate · Blue: reference subject · Green: saved reviewed subject · Dashed teal: unsaved selection.</p>
      <div className="text-xs"><DetectionSummary frame={frame} /></div>
      <p className="text-xs">HOG scores are uncalibrated and are not probabilities; a box may be a person, duplicate, reflection, or background object. Suggested boxes below are sorted separately.</p>
      {(frame?.detector_evidence || []).length > 0 && <ul className="text-xs list-disc pl-4">
        {frame.detector_evidence.map((d, index) => <li key={index}>Candidate {index + 1}: score {typeof d.score === 'number' ? d.score.toFixed(2) : 'unavailable'} (uncalibrated)</li>)}
      </ul>}
      <p className="text-xs">Drag to draw a subject box. {box ? 'Unsaved box selected.' : 'No unsaved box selected.'}</p>
      <p className="text-xs font-semibold">Subject association: {state.status} — {state.text}</p>
      <div className="text-xs space-y-2"><QualityChecks frame={frame} session={session} decision={saved} /><QualityReasons frame={frame} /></div>
      {!session ? <button type="button" onClick={create} disabled={!box || saving} className="bg-teal-600 text-white rounded px-3 py-2 disabled:bg-gray-300">Select reference subject</button> : <>
        <p className="text-xs">Suggestion: {suggestion ? suggestion.suggestion?.status === 'ambiguous' ? 'ambiguous — review required' : 'unconfirmed candidate — review required' : 'loading or unavailable'} · comparison frame: {suggestion?.suggestion?.anchor_frame || 'unknown'}</p>
        <div className="flex flex-wrap gap-2">
          {(suggestion?.suggestion?.candidates || []).map((candidate, index) => (
            <button key={index} type="button" disabled={saving} onClick={() => setBox(candidate.box)} className="border border-amber-400 rounded px-2 py-1 text-xs">
              Use suggested box {index + 1} · overlap {candidate.overlap.toFixed(2)} · distance {candidate.center_distance.toFixed(2)}
            </button>
          ))}
        </div>
        <div className="flex gap-2 flex-wrap">
          <button type="button" disabled={!box || saving} onClick={() => decide('confirm')} className="bg-teal-600 text-white rounded px-3 py-2 disabled:bg-gray-300">Confirm subject</button>
          <button type="button" disabled={!box || saving} onClick={() => decide('reassign')} className="border rounded px-3 py-2 disabled:text-gray-300">Reassign subject</button>
          <button type="button" disabled={saving} onClick={() => decide('reject')} className="border border-red-300 rounded px-3 py-2 text-red-700">Reject frame</button>
        </div>
      </>}
      {error && <p role="alert" className="text-red-700 text-xs">{error}</p>}
    </section>
  )
}
