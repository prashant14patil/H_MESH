import test from 'node:test'
import assert from 'node:assert/strict'
import { recommendFrames } from './frameRecommendations.js'

const checks = { readability: 'pass', person_presence: 'pass', sharpness: 'pass', lighting: 'pass', full_body: 'pass' }
const frames = [
  { filename: 'frame_001.jpg', is_blurry: false, blur_score: 100, timestamp_sec: 0 },
  { filename: 'frame_002.jpg', is_blurry: false, blur_score: 200, timestamp_sec: 3 },
  { filename: 'frame_003.jpg', is_blurry: false, blur_score: 300, timestamp_sec: 6 },
  { filename: 'frame_004.jpg', is_blurry: true, blur_score: 20, timestamp_sec: 9 },
]
const report = { frames: frames.map(f => ({ filename: f.filename, status: 'review', checks, warnings: ['subject_association_unknown'] })) }

test('advisory selection includes ambiguous profile without claiming direction or approval', () => {
  const result = recommendFrames(frames, report, { frames: [
    { frame: 'frame_001', viewpoint: 'front' },
    { frame: 'frame_002', viewpoint: 'side_ambiguous' },
    { frame: 'frame_003', viewpoint: 'back' },
  ] })
  assert.deepEqual(result.recommended.map(x => x.bucket), ['front', 'side_ambiguous', 'back'])
  assert.match(result.excluded['frame_004.jpg'], /blur/)
  assert.equal(result.recommended.some(x => x.approved || x.selected), false)
})

test('unknown mandatory checks and nearby frames are excluded with reasons', () => {
  const changed = { frames: report.frames.map((f, index) => index === 2 ? { ...f, checks: { ...checks, full_body: 'unknown' } } : f) }
  const close = frames.map((f, index) => index === 1 ? { ...f, timestamp_sec: .5 } : f)
  const result = recommendFrames(close, changed, null)
  assert.equal(result.recommended.length, 1)
  assert.match(result.excluded['frame_001.jpg'], /Close in time/)
  assert.match(result.excluded['frame_003.jpg'], /quality/)
})
