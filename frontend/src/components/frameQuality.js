const labels = {
  frame_unreadable: 'The frame could not be read.',
  person_not_detected: 'No person was detected by the pose model.',
  person_detection_unknown: 'Person detection is unavailable.',
  full_body_visibility_unconfirmed: 'Full-body visibility could not be confirmed.',
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
  return { status: 'UNKNOWN', text: 'Not confirmed; reviewer action required.' }
}

function check(frame, key, descriptions) {
  const raw = frame?.checks?.[key]
  const status = ['pass', 'warning', 'fail', 'unknown'].includes(raw) ? raw.toUpperCase() : 'UNKNOWN'
  return { status, text: descriptions[status] }
}

export function qualityChecks(frame, session, decision) {
  const count = frame?.person_count_observed
  const countCheck = typeof count === 'number' && Number.isFinite(count)
    ? count > 1
      ? { status: 'WARNING', text: `${count} detector candidates; inspect for people, duplicates, reflections or background objects.` }
      : { status: 'UNKNOWN', text: `${count} detector ${count === 1 ? 'candidate' : 'candidates'}; person count is not verified.` }
    : { status: 'UNKNOWN', text: 'Detector evidence unavailable; person count is not verified.' }
  const view = frame?.viewpoint
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
      PASS: 'Required body landmarks are visible.', WARNING: 'Full-body visibility needs review.',
      FAIL: 'Full-body visibility failed.', UNKNOWN: 'Full-body visibility is unconfirmed.',
    }) },
    { label: 'Person count', ...countCheck },
    { label: 'Subject association', ...association(frame, session, decision) },
    { label: 'Viewpoint evidence', status: 'UNKNOWN', text: view?.label && view.label !== 'unknown'
      ? `Unvalidated ${String(view.label).replaceAll('_', ' ')} viewpoint hint; coverage is not confirmed.`
      : 'Viewpoint and coverage are unconfirmed.' },
  ]
}
