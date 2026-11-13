import test from 'node:test'
import assert from 'node:assert/strict'
import { association, qualityChecks, qualityGroup, reasonText, validDecision } from './frameQuality.js'

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
  assert.equal(checks['Viewpoint evidence'].status, 'UNKNOWN')
  assert.match(checks['Person count'].text, /2 detector candidates/)
  assert.match(checks['Viewpoint evidence'].text, /Unvalidated/)
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

test('unknown and missing data stay unknown; reason text is readable', () => {
  assert.ok(qualityChecks(null).every(c => c.status === 'UNKNOWN'))
  assert.match(reasonText('possible_multiple_people'), /Multiple person candidates/)
  assert.equal(reasonText('unexpected_long_code'), 'unexpected long code')
})
