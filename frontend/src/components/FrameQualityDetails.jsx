import { qualityChecks, reasonText } from './frameQuality.js'

const colors = {
  PASS: 'text-green-800',
  WARNING: 'text-amber-800',
  FAIL: 'text-red-700',
  UNKNOWN: 'text-gray-600',
}

export function QualityChecks({ frame, session, decision }) {
  return <ul className="space-y-1">
    {qualityChecks(frame, session, decision).map(item => (
      <li key={item.label} className="min-w-0">
        <span className="font-semibold">{item.label}: </span>
        <span className={`font-semibold ${colors[item.status]}`}>{item.status}</span>
        <span> — {item.text}</span>
      </li>
    ))}
  </ul>
}

export function QualityReasons({ frame }) {
  const reasons = [...(frame?.rejections || []), ...(frame?.warnings || [])]
  if (!reasons.length) return null
  return <ul className="list-disc pl-4 space-y-1">
    {reasons.map((reason, index) => <li key={`${reason}-${index}`}>{reasonText(reason)}</li>)}
  </ul>
}

export function boxStyle(box) {
  return { left: `${box[0] * 100}%`, top: `${box[1] * 100}%`,
    width: `${(box[2] - box[0]) * 100}%`, height: `${(box[3] - box[1]) * 100}%` }
}

export function DetectionSummary({ frame }) {
  const nms = frame?.detector_nms
  const raw = nms?.raw_count ?? frame?.person_count_observed
  const filtered = nms?.invalid_indices?.length ? 'incomplete' : (nms?.filtered_count ?? 'unavailable')
  return <div className="space-y-1">
    <p>HOG-returned candidates: {raw ?? 'unknown'} · After NMS: {filtered} (unverified estimates)</p>
    {nms?.invalid_indices?.length > 0 && <p>Some candidates could not be filtered; inspect raw evidence.</p>}
    <p>NMS is diagnostic only; suppressed boxes still count for warnings and crop validation. People may overlap or be missed.</p>
  </div>
}

export function DetectionBoxes({ frame }) {
  const suppressed = new Set((frame?.detector_nms?.suppressed || []).map(item => item.index))
  return (frame?.detector_evidence || []).map((detection, index) => {
    const box = detection?.box
    if (!Array.isArray(box) || box.length !== 4 || !box.every(Number.isFinite)) return null
    const isSuppressed = suppressed.has(index)
    return <div key={index} data-detection-index={index} data-suppressed={isSuppressed}
      className={`absolute border-2 pointer-events-none ${isSuppressed ? 'border-fuchsia-600 border-dashed' : 'border-amber-500'}`} style={boxStyle(box)}>
      <span className={`${isSuppressed ? 'bg-fuchsia-600 text-white' : 'bg-amber-500 text-black'} text-xs font-bold leading-none`}>{index + 1}</span>
    </div>
  })
}
