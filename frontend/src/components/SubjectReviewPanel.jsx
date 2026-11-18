import { useEffect, useRef, useState } from 'react'
import { getFrameImageUrl } from '../services/captureApi.js'
import { createSession, getSuggestion, listSessions, resumeSession, saveDecision } from '../services/subjectReviewApi.js'
import { boxStyle, DetectionBoxes, DetectionSummary, QualityChecks, QualityReasons } from './FrameQualityDetails.jsx'
import { association, frameStatus, statusCounts, validDecision } from './frameQuality.js'

const statusColor = {
  confirmed: 'text-green-800', review_pending: 'text-amber-800', rejected: 'text-red-700',
  quality_failed: 'text-red-700', quality_unknown: 'text-gray-600',
}

function dragBox(a, b) {
  return [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[0], b[0]), Math.max(a[1], b[1])]
}

function clampBox(box, pad) {
  return [Math.max(0, box[0] - pad), Math.max(0, box[1] - pad), Math.min(1, box[2] + pad), Math.min(1, box[3] + pad)]
}

export default function SubjectReviewPanel({ videoId, report, onSession, onDecision, requestedFrame, decisions = {} }) {
  const [session, setSession] = useState(null)
  const [lookup, setLookup] = useState({ state: 'loading', sessions: [] })
  const [filename, setFilename] = useState('')
  const [box, setBox] = useState(null)
  const [draft, setDraft] = useState(null)
  const [suggestion, setSuggestion] = useState(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [conflict, setConflict] = useState(null)
  const [replacing, setReplacing] = useState(false)
  const [saving, setSaving] = useState(false)
  const [advance, setAdvance] = useState(true)
  const [manualId, setManualId] = useState('')
  const start = useRef(null)
  const image = useRef(null)

  async function activate(sessionId, message) {
    const result = await resumeSession(sessionId, videoId, report.run_id)
    setSession(result.session)
    onSession(result.session, result.decisions, { ...result.selection, session_id: result.session.session_id }, result.mesh_runs || [])
    setReplacing(false); setConflict(null); setBox(null); setError('')
    const restored = Object.keys(result.decisions).length
    const parts = [message || `Resumed review session ${result.session.session_id}.`,
      `${restored} saved frame decision${restored === 1 ? '' : 's'} restored.`]
    if (result.selection?.frames?.length) parts.push(`${result.selection.frames.length} selected frame(s) restored.`)
    if (result.unavailable_frames?.length) parts.push(`Stale or unavailable decisions (review again): ${result.unavailable_frames.join(', ')}.`)
    if (result.dropped_selection?.length) parts.push(`Removed from selection because they are no longer confirmed: ${result.dropped_selection.join(', ')}.`)
    setNotice(parts.join(' '))
    return result
  }

  async function refreshLookup() {
    const listing = await listSessions(videoId, report.run_id)
    setLookup({ state: listing.active_session_id ? 'ready' : 'none', sessions: listing.sessions })
    return listing
  }

  // Sessions are bound to this recording + quality report; find and resume the active one automatically.
  useEffect(() => {
    let cancelled = false
    setSession(null); setFilename(report?.frames?.[0]?.filename || ''); setBox(null); setSuggestion(null)
    setError(''); setNotice(''); setConflict(null); setReplacing(false); setLookup({ state: 'loading', sessions: [] })
    onSession(null)
    if (!report?.run_id || !videoId) return undefined
    listSessions(videoId, report.run_id)
      .then(async listing => {
        if (cancelled) return
        setLookup({ state: listing.active_session_id ? 'ready' : 'none', sessions: listing.sessions })
        if (listing.active_session_id) {
          setSaving(true)
          try { await activate(listing.active_session_id, `Automatically resumed review session ${listing.active_session_id}.`) } finally { setSaving(false) }
        }
      })
      .catch(err => { if (!cancelled) { setLookup({ state: 'error', sessions: [] }); setError(`Could not look up review sessions: ${err.message}`) } })
    return () => { cancelled = true }
  }, [report?.run_id, videoId])

  useEffect(() => {
    if (requestedFrame?.name && report?.frames?.some(f => f.filename === requestedFrame.name)) {
      setFilename(requestedFrame.name); setBox(null)
    }
  }, [requestedFrame, report])

  useEffect(() => {
    if (!session || !filename || replacing) return undefined
    let cancelled = false
    setBox(null); setSuggestion(null); setError('')
    getSuggestion(session.session_id, filename)
      .then(data => { if (!cancelled) setSuggestion(data) })
      .catch(err => { if (!cancelled) setError(err.message) })
    return () => { cancelled = true }
  }, [session?.session_id, filename, replacing])

  function point(event) {
    const rect = image.current.getBoundingClientRect()
    return [Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width)),
      Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height))]
  }

  function goTo(name) { setFilename(name); setBox(null); setDraft(null); start.current = null }

  async function selectReference() {
    setSaving(true); setError(''); setConflict(null)
    try {
      const created = await createSession(videoId, report.run_id, filename, box, replacing)
      await activate(created.session_id, created.resumed_existing
        ? `This reference subject is already saved; resumed session ${created.session_id} (no duplicate created).`
        : `Reference subject saved. Review session ${created.session_id} created for this recording and quality report.`)
      await refreshLookup()
    } catch (err) {
      if (err.status === 409) setConflict({ message: err.message, sessionId: err.detail?.active_session_id })
      else setError(err.message)
    } finally { setSaving(false) }
  }

  async function resumeById(sessionId) {
    setSaving(true); setError('')
    try { await activate(sessionId) } catch (err) { setError(err.message) } finally { setSaving(false) }
  }

  async function decide(decision) {
    setSaving(true)
    try {
      const result = await saveDecision(session.session_id, filename, decision, box)
      setSuggestion(old => ({ ...old, decision: result, identity_status: result.identity_status }))
      onDecision(result); setBox(null); setError('')
      const updated = { ...decisions, [filename]: result }
      if (advance) {
        const index = report.frames.findIndex(f => f.filename === filename)
        const ordered = [...report.frames.slice(index + 1), ...report.frames.slice(0, index)]
        const next = ordered.find(f => frameStatus(f, session, updated[f.filename]).key === 'review_pending')
        if (next) goTo(next.filename)
      }
    } catch (err) { setError(err.message) } finally { setSaving(false) }
  }

  if (!report) return null
  const frame = report.frames.find(f => f.filename === filename)
  const frameIndex = report.frames.findIndex(f => f.filename === filename)
  const validBox = box && box.every(Number.isFinite) && box[0] >= 0 && box[1] >= 0 &&
    box[2] <= 1 && box[3] <= 1 && box[0] < box[2] && box[1] < box[3] &&
    (box[2] - box[0]) * (box[3] - box[1]) >= 0.005
  const landmark = frame?.landmark_bbox_normalized
  const containsLandmarks = validBox && Array.isArray(landmark) && landmark.length === 4 &&
    box[0] <= landmark[0] && box[1] <= landmark[1] && box[2] >= landmark[2] && box[3] >= landmark[3]
  const centers = (frame?.detector_evidence || []).filter(d => Array.isArray(d?.box) && d.box.length === 4).map(d => [(d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2])
  const multipleCenters = validBox && centers.filter(([x, y]) => box[0] <= x && x <= box[2] && box[1] <= y && y <= box[3]).length > 1
  const boxReady = containsLandmarks && !multipleCenters
  const saved = validDecision(frame, session, decisions[filename]) ? decisions[filename] : null
  const state = association(frame, session, saved)
  const status = frameStatus(frame, session, decisions[filename])
  const qualityBlocked = status.key === 'quality_failed' || status.key === 'quality_unknown'
  const savedBox = saved?.selected_subject_box
  const counts = statusCounts(report.frames, session, decisions)
  const choosingReference = !session || replacing
  const detectorName = report.detector_method === 'torchvision_keypoint_rcnn' ? 'Keypoint R-CNN' : 'detector'
  const otherSessions = lookup.sessions.filter(s => s.session_id !== session?.session_id)

  return (
    <section id="subject-review" className="min-w-0 rounded-xl border border-gray-200 bg-white p-4 space-y-3 [overflow-wrap:anywhere] [word-break:break-word]">
      <h3 className="font-semibold">Subject-targeted manual review</h3>
      <ol className="text-xs text-gray-700 list-decimal pl-4 space-y-1">
        <li className={session ? 'text-green-800' : 'font-semibold'}>{session ? '✓ ' : ''}Choose a clear reference frame, draw (or start from a {detectorName} box and adjust) a box around the intended person, then click “Select reference subject”. A review session is created and saved automatically.</li>
        <li className={session && counts.confirmed ? 'text-green-800' : session ? 'font-semibold' : ''}>For each frame you want to use, inspect the boxes, choose or draw the subject box, then confirm, reassign or reject it.</li>
        <li>In “Experimental — individual-frame meshes” below, select 1–12 confirmed frames and acknowledge their warnings.</li>
      </ol>
      <p className="text-xs text-amber-800">Human-confirmed subject association only — not biometric identity verification. The reviewed crop must contain the pose landmark extent and cannot contain multiple detected person centers. Failed or unknown mandatory checks stay blocked.</p>

      <div className="rounded border p-2 text-xs space-y-1" aria-label="Review session">
        {lookup.state === 'loading' && <p>Looking up saved review sessions for this recording…</p>}
        {session ? <p className="font-semibold text-teal-800">Active review session: <span className="font-mono">{session.session_id}</span> · reference subject in {session.reference_frame} · recording {session.video_id.slice(0, 8)} · quality report {session.quality_run_id.slice(0, 8)}</p>
          : lookup.state !== 'loading' && <p className="font-semibold text-amber-800">No reference subject yet. Next action: draw a box around the intended person below and click “Select reference subject”.</p>}
        {session && !replacing && <button type="button" disabled={saving} onClick={() => { setReplacing(true); setBox(null); setNotice('') }} className="border rounded px-2 py-0.5">Change reference subject…</button>}
        {replacing && <p className="text-amber-800">Replacing the reference subject starts a new session; decisions from session {session.session_id.slice(0, 8)} are not carried over. <button type="button" onClick={() => setReplacing(false)} className="underline">Cancel</button></p>}
        {notice && <p role="status" className="text-gray-700">{notice}</p>}
        {(otherSessions.length > 0 || !session) && <details>
          <summary className="cursor-pointer">Other saved sessions ({otherSessions.length}) and manual resume</summary>
          <ul className="pl-4 list-disc">
            {otherSessions.map(s => <li key={s.session_id}>
              <span className="font-mono">{s.session_id}</span> · ref {s.reference_frame} · {s.decision_count} decision(s) · {s.valid ? 'resumable' : s.problem}
              {s.valid && <button type="button" disabled={saving} onClick={() => resumeById(s.session_id)} className="ml-2 border rounded px-1">Resume</button>}
            </li>)}
          </ul>
          <form onSubmit={e => { e.preventDefault(); resumeById(manualId.trim()) }} className="flex flex-wrap gap-2 items-end mt-1">
            <label>Session ID <input value={manualId} onChange={e => setManualId(e.target.value)} className="block border rounded p-1" required /></label>
            <button type="submit" disabled={saving} className="border rounded px-2 py-1">Resume session</button>
          </form>
        </details>}
      </div>

      <p className="text-xs" aria-label="Review progress">
        Progress: <span className="text-green-800 font-semibold">{counts.confirmed} subject confirmed — eligible</span> · <span className="text-amber-800">{counts.review_pending} quality passed — subject review pending</span> · <span className="text-red-700">{counts.rejected} rejected by reviewer</span> · {counts.quality_failed} quality failed · {counts.quality_unknown} quality unknown
      </p>

      <label className="text-sm">Frame <select value={filename} disabled={saving} onChange={e => goTo(e.target.value)} className="border rounded p-1 max-w-full">
        {report.frames.map(f => <option key={f.filename} value={f.filename}>{f.filename} — {frameStatus(f, session, decisions[f.filename]).label}</option>)}
      </select></label>
      <div className="flex flex-wrap gap-2 text-xs items-center">
        <button type="button" className="border rounded px-2 py-1 disabled:text-gray-400" disabled={saving || frameIndex <= 0} onClick={() => goTo(report.frames[frameIndex - 1].filename)}>Previous frame</button>
        <span>Frame {frameIndex + 1} of {report.frames.length}</span>
        <button type="button" className="border rounded px-2 py-1 disabled:text-gray-400" disabled={saving || frameIndex < 0 || frameIndex >= report.frames.length - 1} onClick={() => goTo(report.frames[frameIndex + 1].filename)}>Next frame</button>
        {session && <label className="flex gap-1 items-center"><input type="checkbox" checked={advance} onChange={e => setAdvance(e.target.checked)} /> After saving, go to the next pending frame</label>}
      </div>
      <p className={`text-xs font-semibold ${statusColor[status.key]}`}>Status: {status.label}{status.action && !choosingReference ? ` · Next: ${status.action}` : ''}</p>

      {filename && <div className="relative inline-block max-w-full touch-none select-none cursor-crosshair" style={{ maxWidth: 640 }}
        onPointerDown={e => { if (saving) return; start.current = point(e); setDraft(null); e.currentTarget.setPointerCapture(e.pointerId) }}
        onPointerMove={e => { if (start.current) setDraft(dragBox(start.current, point(e))) }}
        onPointerUp={e => { if (!start.current) return; const next = dragBox(start.current, point(e)); start.current = null; setDraft(null); if ((next[2] - next[0]) * (next[3] - next[1]) > 0.0005) setBox(next) }}
        onPointerCancel={() => { start.current = null; setDraft(null) }}>
        <img ref={image} src={`${getFrameImageUrl(videoId, filename)}?sha256=${encodeURIComponent(frame?.source_sha256 || '')}`} alt={`Review ${filename}`} draggable="false" className="block max-w-full max-h-[65vh] w-auto" />
        <DetectionBoxes frame={frame} />
        {session?.reference_frame === filename && <div className="absolute border-2 border-blue-600 pointer-events-none" style={boxStyle(session.reference_box)} />}
        {savedBox && <div className="absolute border-2 border-green-600 pointer-events-none" style={boxStyle(savedBox)} />}
        {(draft || validBox) && <div className="absolute border-2 border-teal-500 border-dashed pointer-events-none" style={boxStyle(draft || box)} />}
      </div>}
      <p className="text-xs">Amber solid: {detectorName} candidate · Magenta dashed: NMS-suppressed candidate · Blue: reference subject · Green: saved reviewed subject · Dashed teal: your unsaved box. Drag on the image to draw a box.</p>
      <div className="text-xs"><DetectionSummary frame={frame} /></div>

      <div className="flex flex-wrap gap-2 text-xs">
        {(frame?.detector_evidence || []).map((d, index) => Array.isArray(d?.box) && d.box.length === 4 &&
          <button key={`det-${index}`} type="button" disabled={saving} onClick={() => setBox(d.box.map(Number))} className="border border-amber-500 rounded px-2 py-1">
            Start from {detectorName} box {index + 1} (score {typeof d.score === 'number' ? d.score.toFixed(2) : 'n/a'}, uncalibrated)
          </button>)}
        {!choosingReference && (suggestion?.suggestion?.candidates || []).map((candidate, index) => (
          <button key={`sug-${index}`} type="button" disabled={saving} onClick={() => setBox(candidate.box)} className="border border-teal-500 rounded px-2 py-1">
            Use suggested box {index + 1} · overlap with {suggestion.suggestion.anchor_frame === session?.reference_frame ? 'reference' : suggestion.suggestion.anchor_frame} {candidate.overlap.toFixed(2)}
          </button>
        ))}
      </div>
      {!choosingReference && <p className="text-xs">Suggestion: {suggestion ? suggestion.suggestion?.status === 'ambiguous' ? 'ambiguous — inspect carefully' : 'unconfirmed candidate — your confirmation is still required' : 'loading or unavailable'} (box overlap, not identity).</p>}

      {box && <div className="flex flex-wrap gap-2 text-xs items-end">
        {['Left', 'Top', 'Right', 'Bottom'].map((label, index) => <label key={label}>{label}
          <input type="number" min="0" max="1" step="0.005" value={Number.isFinite(box[index]) ? box[index] : ''}
            onChange={e => setBox(old => old.map((value, i) => i === index ? (e.target.value === '' ? NaN : Number(e.target.value)) : value))}
            className="block w-20 border rounded p-1" />
        </label>)}
        <button type="button" disabled={!validBox} onClick={() => setBox(old => clampBox(old, 0.02))} className="border rounded px-2 py-1">Expand 2%</button>
        <button type="button" disabled={!validBox} onClick={() => setBox(old => clampBox(old, -0.01))} className="border rounded px-2 py-1">Shrink 1%</button>
        <button type="button" onClick={() => setBox(null)} className="border rounded px-2 py-1">Clear box</button>
      </div>}
      <p className="text-xs text-amber-800">{!box ? 'No box yet: draw one on the image or start from a detector box.'
        : !validBox ? 'Enter a valid box inside the image with sufficient area.'
          : !containsLandmarks ? 'The box must contain all pose landmarks used for quality checks; expand or redraw it.'
            : multipleCenters ? 'The box contains more than one detected person center; tighten it around one person or reject the frame.'
              : 'Box contains the pose landmarks and one person center. The server re-validates the crop and frame hash.'}</p>

      <p className="text-xs font-semibold">Subject association: {state.status} — {state.text}</p>
      <div className="text-xs space-y-2"><QualityChecks frame={frame} session={session} decision={saved} /><QualityReasons frame={frame} /></div>

      {choosingReference ? <>
        {qualityBlocked && <p className="text-xs text-red-700">This frame failed or has unknown mandatory checks; choose a frame whose quality passed as the reference.</p>}
        <button type="button" onClick={selectReference} disabled={!boxReady || saving || qualityBlocked} className="bg-teal-600 text-white rounded px-3 py-2 disabled:bg-gray-300">
          {saving ? 'Saving…' : replacing ? 'Replace reference subject' : 'Select reference subject'}
        </button>
        {conflict && <div role="alert" className="text-xs text-amber-800 space-y-1">
          <p>{conflict.message}</p>
          {conflict.sessionId && <button type="button" onClick={() => resumeById(conflict.sessionId)} className="border rounded px-2 py-1 mr-2">Resume session {conflict.sessionId.slice(0, 8)}</button>}
          {conflict.sessionId && !replacing && <button type="button" onClick={() => { setReplacing(true); setConflict(null) }} className="border border-red-300 rounded px-2 py-1">Replace it with this reference…</button>}
        </div>}
      </> : <div className="flex gap-2 flex-wrap">
        <button type="button" disabled={!boxReady || saving || qualityBlocked} onClick={() => decide('confirm')} className="bg-teal-600 text-white rounded px-3 py-2 disabled:bg-gray-300">Confirm subject</button>
        <button type="button" disabled={!boxReady || saving || qualityBlocked} onClick={() => decide('reassign')} className="border rounded px-3 py-2 disabled:text-gray-300">Reassign subject</button>
        <button type="button" disabled={saving} onClick={() => decide('reject')} className="border border-red-300 rounded px-3 py-2 text-red-700">Reject frame</button>
        {qualityBlocked && <p className="text-xs text-red-700 w-full">Mandatory quality checks did not pass, so this frame can only be rejected.</p>}
      </div>}
      {error && <p role="alert" className="text-red-700 text-xs">{error}</p>}
    </section>
  )
}
