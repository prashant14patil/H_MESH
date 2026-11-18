import test from 'node:test'
import assert from 'node:assert/strict'
import { association, qualityChecks, qualityGroup, reasonText, selectionBlockReason, validDecision } from './frameQuality.js'

const frame = {
  filename: 'frame.jpg', source_sha256: 'hash', person_detected: true,
  checks: { sharpness: 'pass', lighting: 'warning', full_body: 'unknown' },
  warnings: ['possible_multiple_people', 'subject_association_ambiguous'],
  person_count_observed: 2, viewpoint: { label: 'side' },
}
const session = { session_id: 'session', quality_run_id: 'quality' }
const decision = { session_id: 'session', quality_run_id: 'quality', frame_id: 'frame.jpg',
  frame_sha256: 'hash', reviewer_decision: 'confirm', identity_status: 'human_confirmed_not_biometric' }

test('separate checks retain uncertain and warning statuses', () => {
  const checks = Object.fromEntries(qualityChecks(frame).map(item => [item.label, item]))
  assert.equal(checks.Blur.status, 'PASS')
  assert.equal(checks.Lighting.status, 'WARNING')
  assert.equal(checks['Full-body visibility'].status, 'UNKNOWN')
  assert.equal(checks['Person count'].status, 'WARNING')
  assert.equal(checks['Subject association'].status, 'WARNING')
  assert.equal(checks['Viewpoint evidence'].status, 'HINT')
  assert.match(checks['Person count'].text, /2 person detections/)
  assert.match(checks['Viewpoint evidence'].text, /Unvalidated.*not ground truth/)
})

test('only the keypoint detector pass reports a single person; unknown viewpoint stays unknown', () => {
  const single = { ...frame, person_count_observed: 1, person_detection_confidence: 0.998,
    checks: { ...frame.checks, multiple_people: 'pass' }, viewpoint: { label: 'unknown' } }
  const checks = Object.fromEntries(qualityChecks(single).map(item => [item.label, item]))
  assert.equal(checks['Person count'].status, 'PASS')
  assert.match(checks['Person count'].text, /not identity verification/)
  assert.equal(checks['Viewpoint evidence'].status, 'UNKNOWN')
})

test('self-occlusion full-body pass is explained', () => {
  const side = { ...frame, checks: { ...frame.checks, full_body: 'pass' },
    full_body_basis: 'in_frame_one_sided_self_occlusion_detector_confirmed' }
  assert.match(qualityChecks(side).find(c => c.label === 'Full-body visibility').text, /hidden behind the body/)
  assert.match(reasonText('full_body_self_occlusion_one_side'), /Side view/)
})

test('zero, one or unavailable detections never prove a single person', () => {
  for (const count of [0, 1, null, undefined]) {
    const checks = qualityChecks({ ...frame, person_count_observed: count })
    assert.equal(checks.find(c => c.label === 'Person count').status, 'UNKNOWN')
  }
})

test('only current explicit human decisions are confirmed', () => {
  assert.equal(validDecision(frame, session, decision), true)
  assert.equal(association(frame, session, decision).status, 'PASS')
  assert.match(association(frame, session, decision).text, /not biometric/)
  for (const stale of [
    { ...decision, frame_sha256: 'old' }, { ...decision, session_id: 'old' },
    { ...decision, quality_run_id: 'old' }, { ...decision, frame_id: 'other.jpg' },
  ]) assert.notEqual(association(frame, session, stale).status, 'PASS')
  assert.equal(association(frame, session, { ...decision, reviewer_decision: 'reject' }).status, 'FAIL')
  assert.equal(association({ ...frame, person_detected: false }, session, null).status, 'UNKNOWN')
})

test('mandatory quality grouping never approves subjects', () => {
  const passing = { ...frame, status: 'review', checks: {
    readability: 'pass', person_presence: 'pass', sharpness: 'pass', lighting: 'pass', full_body: 'pass',
  } }
  assert.equal(qualityGroup(passing), 'passed')
  assert.equal(association(passing, null, null).status, 'WARNING')
  assert.equal(qualityGroup({ ...passing, checks: { ...passing.checks, lighting: 'warning' } }), 'blocked')
  assert.equal(qualityGroup({ ...passing, checks: { ...passing.checks, full_body: 'unknown' } }), 'unknown')
  assert.equal(qualityGroup({ ...passing, rejections: ['frame_unreadable'] }), 'blocked')
  assert.equal(qualityGroup(null), 'unknown')
})

test('selection requires current per-frame review and explains every block', () => {
  const passing = { ...frame, status: 'review', checks: {
    readability: 'pass', person_presence: 'pass', sharpness: 'pass', lighting: 'pass', full_body: 'pass',
  } }
  assert.match(selectionBlockReason(passing, null, null), /reference subject/)
  assert.match(selectionBlockReason(passing, session, null), /Review this frame/)
  assert.equal(selectionBlockReason(passing, session, decision), null)
  assert.match(selectionBlockReason(passing, session, { ...decision, reviewer_decision: 'reject' }), /rejected/)
  assert.match(selectionBlockReason(passing, session, { ...decision, frame_sha256: 'old' }), /stale/)
  assert.match(selectionBlockReason({ ...passing, checks: { ...passing.checks, full_body: 'unknown' } }, session, decision), /unknown/)
  assert.match(selectionBlockReason({ ...passing, checks: { ...passing.checks, sharpness: 'warning' } }, session, decision), /quality/)
  assert.match(selectionBlockReason(passing, session, decision, { limitReached: true }), /12-frame/)
  assert.match(selectionBlockReason(passing, session, decision, { busy: true }), /progress/)
  assert.match(selectionBlockReason(passing, { ...session, video_id: 'recording' }, decision), /recording/)
})

test('unknown and missing data stay unknown; reason text is readable', () => {
  // No report yet is "not run", never a pass; it is distinguishable from an analysed unknown.
  assert.ok(qualityChecks(null).every(c => c.status === 'NOT RUN' && /Analyze capture quality/.test(c.text)))
  assert.match(reasonText('possible_multiple_people'), /Multiple person candidates/)
  assert.equal(reasonText('unexpected_long_code'), 'unexpected long code')
})

test('frame status distinguishes pending review, confirmation, rejection and quality blocks', async () => {
  const { frameStatus, statusCounts, FRAME_STATUS } = await import('./frameQuality.js')
  const passed = { filename: 'a.jpg', source_sha256: 'h', status: 'review',
    checks: { readability: 'pass', person_presence: 'pass', sharpness: 'pass', lighting: 'pass', full_body: 'pass' } }
  const s = { session_id: 's', quality_run_id: 'q' }
  const confirm = { session_id: 's', quality_run_id: 'q', frame_id: 'a.jpg', frame_sha256: 'h',
    reviewer_decision: 'confirm', identity_status: 'human_confirmed_not_biometric' }
  assert.equal(frameStatus(passed, null, null).label, FRAME_STATUS.review_pending)
  assert.match(frameStatus(passed, null, null).action, /reference subject/)
  assert.equal(frameStatus(passed, s, null).key, 'review_pending')
  assert.equal(frameStatus(passed, s, confirm).label, FRAME_STATUS.confirmed)
  assert.equal(frameStatus(passed, s, { ...confirm, reviewer_decision: 'reject', identity_status: 'rejected' }).key, 'rejected')
  assert.match(frameStatus(passed, s, { ...confirm, frame_sha256: 'stale' }).action, /stale/)
  assert.equal(frameStatus({ ...passed, checks: { ...passed.checks, lighting: 'fail' } }, s, confirm).key, 'quality_failed')
  assert.equal(frameStatus({ ...passed, checks: { ...passed.checks, lighting: 'unknown' } }, s, confirm).key, 'quality_unknown')
  // A confirmation never upgrades a quality-blocked frame.
  const counts = statusCounts([passed, { ...passed, filename: 'b.jpg' }], s, { 'a.jpg': confirm })
  assert.deepEqual(counts, { confirmed: 1, review_pending: 1, rejected: 0, quality_failed: 0, quality_unknown: 0 })
})
