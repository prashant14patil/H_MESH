const labels = {
  frame_unreadable: 'The frame could not be read.',
  person_not_detected: 'No person was detected by the pose model.',
  person_detection_unknown: 'Person detection is unavailable.',
  full_body_visibility_unconfirmed: 'Full-body visibility could not be confirmed.',
  full_body_self_occlusion_one_side: 'Side view: limbs on one side are hidden behind the body; the whole body is inside the frame.',
  extractor_blur_warning: 'The frame extractor flagged possible blur.',
  frame_sharpness_review: 'Overall image sharpness needs review.',
  roi_sharpness_review: 'Subject-region sharpness needs review.',
  lighting_review: 'Lighting or clipping needs review.',
  possible_multiple_people: 'Multiple person candidates were detected; inspect the boxes.',
  person_count_unknown: 'Person detection evidence is unavailable.',
  subject_association_ambiguous: 'The detected subject association is ambiguous.',
  subject_association_unknown: 'The subject association could not be determined.',
  subject_continuity_uncertain: 'Subject position changed substantially between frames.',
  person_count_and_identity_unverified: 'Person count and identity are not automatically verified.',
}

export function reasonText(code) {
  return labels[code] || String(code || 'Unknown reason').replaceAll('_', ' ')
}

export const mandatoryChecks = ['readability', 'person_presence', 'sharpness', 'lighting', 'full_body']

export function qualityGroup(frame) {
  if (!frame) return 'unknown'
  const checks = mandatoryChecks.map(key => frame.checks?.[key])
  if (frame.status === 'rejected' || frame.rejections?.length || checks.some(state => state === 'fail' || state === 'warning')) return 'blocked'
  if (!['acceptable', 'review'].includes(frame.status) || checks.some(state => state !== 'pass')) return 'unknown'
  return 'passed'
}

export function validDecision(frame, session, decision) {
  return !!frame && !!session && !!decision && decision.session_id === session.session_id &&
    decision.quality_run_id === session.quality_run_id && decision.frame_id === frame.filename &&
    !!frame.source_sha256 && decision.frame_sha256 === frame.source_sha256
}

export function selectionBlockReason(frame, session, decision, { busy = false, limitReached = false } = {}) {
  if (busy) return 'A reconstruction is in progress.'
  if (!frame) return 'A quality report is required.'
  const group = qualityGroup(frame)
  if (group === 'blocked') return 'A mandatory quality check failed, has a warning, or the frame was rejected.'
  if (group !== 'passed') return 'Mandatory quality checks are unknown or unavailable.'
  if (!session || !session.session_id || !session.quality_run_id) return 'Select a reference subject and save a review session first.'
  if (!decision) return 'Review this frame and explicitly confirm or reassign its subject.'
  if (!validDecision(frame, session, decision)) return 'The review is stale or does not match this frame and session; review it again.'
  if (decision.reviewer_decision === 'reject') return 'This frame was rejected by the reviewer.'
  if (!['confirm', 'reassign'].includes(decision.reviewer_decision) ||
      decision.identity_status !== 'human_confirmed_not_biometric') {
    return 'This frame needs an explicit subject confirmation.'
  }
  if (session.video_id && decision.recording_id !== session.video_id) return 'The review does not match this recording; review it again.'
  if (limitReached) return 'The 12-frame selection limit has been reached.'
  return null
}

export const FRAME_STATUS = {
  confirmed: 'Subject confirmed — eligible for selection',
  review_pending: 'Quality passed — subject review pending',
  rejected: 'Rejected by reviewer',
  quality_failed: 'Quality failed — blocked',
  quality_unknown: 'Quality unknown — blocked',
}

// One status and next action per frame, shared by every panel so messages never disagree.
export function frameStatus(frame, session, decision) {
  const group = qualityGroup(frame)
  if (!frame || group === 'unknown') {
    return { key: 'quality_unknown', label: FRAME_STATUS.quality_unknown,
      action: frame ? 'Mandatory checks are unknown; re-run quality analysis or choose another frame.' : 'Run capture quality analysis.' }
  }
  if (group === 'blocked') {
    return { key: 'quality_failed', label: FRAME_STATUS.quality_failed,
      action: 'A mandatory check failed or warned; choose another frame or recapture.' }
  }
  const valid = validDecision(frame, session, decision)
  if (valid && decision.reviewer_decision === 'reject') {
    return { key: 'rejected', label: FRAME_STATUS.rejected,
      action: 'Open it in subject review and confirm it if the rejection was a mistake.' }
  }
  if (valid && ['confirm', 'reassign'].includes(decision.reviewer_decision) &&
      decision.identity_status === 'human_confirmed_not_biometric') {
    return { key: 'confirmed', label: FRAME_STATUS.confirmed, action: null }
  }
  return { key: 'review_pending', label: FRAME_STATUS.review_pending,
    action: !session ? 'Select a reference subject first, then confirm the subject on this frame.'
      : decision ? 'The saved review is stale for this frame; review it again.'
        : 'Open it in subject review and confirm, reassign or reject the subject.' }
}

export function statusCounts(frames, session, decisions = {}) {
  const counts = { confirmed: 0, review_pending: 0, rejected: 0, quality_failed: 0, quality_unknown: 0 }
  for (const frame of frames || []) counts[frameStatus(frame, session, decisions[frame?.filename]).key] += 1
  return counts
}

export function association(frame, session, decision) {
  if (validDecision(frame, session, decision)) {
    if (decision.reviewer_decision === 'reject') return { status: 'FAIL', text: 'Rejected by reviewer.' }
    if (['confirm', 'reassign'].includes(decision.reviewer_decision) &&
        decision.identity_status === 'human_confirmed_not_biometric') {
      return { status: 'PASS', text: 'Confirmed by reviewer; not biometric identity verification.' }
    }
  }
  if (frame?.person_detected === false) return { status: 'UNKNOWN', text: 'Subject missing from pose detection; reviewer action required.' }
  if (frame?.warnings?.includes('subject_association_ambiguous') ||
      frame?.warnings?.includes('subject_continuity_uncertain')) {
    return { status: 'WARNING', text: 'Association is ambiguous; reviewer action required.' }
  }
  return { status: 'UNKNOWN', text: session
    ? 'Subject review pending: confirm, reassign or reject this frame against the reference subject.'
    : 'Subject review pending: select a reference subject first, then confirm this frame.' }
}

function check(frame, key, descriptions) {
  const raw = frame?.checks?.[key]
  const status = ['pass', 'warning', 'fail', 'unknown'].includes(raw) ? raw.toUpperCase() : 'UNKNOWN'
  return { status, text: descriptions[status] }
}

const NOT_RUN_TEXT = 'Quality analysis has not been run for this recording yet — click "Analyze capture quality".'

export function qualityChecks(frame, session, decision) {
  if (!frame) {
    return ['Blur', 'Lighting', 'Full-body visibility', 'Person count', 'Subject association', 'Viewpoint evidence']
      .map(label => ({ label, status: 'NOT RUN', text: NOT_RUN_TEXT }))
  }
  const count = frame?.person_count_observed
  const score = typeof frame?.person_detection_confidence === 'number' ? ` (score ${frame.person_detection_confidence.toFixed(2)})` : ''
  const people = frame?.checks?.multiple_people
  const countCheck = people === 'pass'
    ? { status: 'PASS', text: `One person detected by Keypoint R-CNN${score}; screening evidence, not identity verification.` }
    : typeof count === 'number' && Number.isFinite(count)
      ? count > 1
        ? { status: 'WARNING', text: `${count} person detections; inspect for bystanders, duplicates or reflections.` }
        : { status: 'UNKNOWN', text: `${count} detector ${count === 1 ? 'candidate' : 'candidates'} from a legacy detector; person count is not verified.` }
      : { status: 'UNKNOWN', text: 'Detector evidence unavailable; person count is not verified.' }
  const view = frame?.viewpoint
  const fullBodyPass = frame?.full_body_basis === 'in_frame_one_sided_self_occlusion_detector_confirmed'
    ? 'Whole body is inside the frame; some limbs are hidden behind the body (side view), confirmed by an independent detector.'
    : 'Required body landmarks are visible.'
  return [
    { label: 'Blur', ...check(frame, 'sharpness', {
      PASS: 'Sharpness screening passed.', WARNING: 'Possible blur or low sharpness needs review.',
      FAIL: 'Sharpness screening failed.', UNKNOWN: 'Sharpness is unavailable.',
    }) },
    { label: 'Lighting', ...check(frame, 'lighting', {
      PASS: 'Lighting screening passed.', WARNING: 'Lighting or clipping needs review.',
      FAIL: 'Lighting screening failed.', UNKNOWN: 'Lighting is unavailable.',
    }) },
    { label: 'Full-body visibility', ...check(frame, 'full_body', {
      PASS: fullBodyPass, WARNING: 'Full-body visibility needs review.',
      FAIL: 'Full-body visibility failed.', UNKNOWN: 'Full-body visibility is unconfirmed.',
    }) },
    { label: 'Person count', ...countCheck },
    { label: 'Subject association', ...association(frame, session, decision) },
    view?.label && view.label !== 'unknown'
      ? { label: 'Viewpoint evidence', status: 'HINT', text: `Unvalidated ${String(view.label).replaceAll('_', ' ')} viewpoint hint; not ground truth and coverage is not confirmed.` }
      : { label: 'Viewpoint evidence', status: 'UNKNOWN', text: 'No usable viewpoint hint; coverage is unconfirmed.' },
  ]
}
