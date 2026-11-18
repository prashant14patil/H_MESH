import test from 'node:test'
import assert from 'node:assert/strict'
import {
  parseObj, parseRings, centroid, scanValues, comparableValues, interpolateValues, uncertaintyCm,
  siteStatus, changeStatus, segmentStatus, asymmetricSites, formatDelta, CALLOUTS, repeatabilityCoefficient,
} from './bodyAvatarModel.js'

test('parseObj keeps vertex order and triangulates faces', () => {
  const m = parseObj('# x\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nf 1 2 3 4\n')
  assert.equal(m.vertexCount, 4)
  assert.deepEqual([...m.indices], [0, 1, 2, 0, 2, 3])
})

test('parseRings groups polylines by object name', () => {
  const r = parseRings('o chest\nv 0 1 0\nv 2 1 0\nl 1 2 1\no waist\nv 0 0.5 0\n')
  assert.deepEqual(Object.keys(r), ['chest', 'waist'])
  assert.deepEqual(centroid(r.chest), [1, 1, 0])
})

test('scanValues prefers cm when the run was scaled, else ratios', () => {
  const rel = scanValues({ measurements: { waist: 0.5 } })
  assert.equal(rel.unit, 'ratio')
  assert.equal(rel.values.waist, 0.5)
  const cm = scanValues({ measurements: { waist: 0.5, hip: null },
    metric_scale: { height_cm: 170, relative_scale_uncertainty: 1 / 170, measurements_cm: { waist: 85 } } })
  assert.equal(cm.unit, 'cm')
  assert.equal(cm.values.waist, 85)
  assert.equal(cm.values.hip, null)
})

test('mixed scaled/unscaled scans fall back to ratios', () => {
  const a = { measurements: { waist: 0.5 }, metric_scale: { height_cm: 170, measurements_cm: { waist: 85 } } }
  const b = { measurements: { waist: 0.52 } }
  const c = comparableValues(a, b)
  assert.equal(c.unit, 'ratio')
  assert.equal(c.now.waist, 0.5)
  assert.equal(c.before.waist, 0.52)
})

test('interpolation runs from the earlier to the current scan', () => {
  assert.deepEqual(interpolateValues({ w: 80 }, { w: 90 }, 0.5), { w: 85 })
  assert.deepEqual(interpolateValues({ w: 80 }, { w: null }, 1), { w: null })
})

test('uncertainty combines height scale and repeatability', () => {
  // sqrt((100 x 1%)^2 + (100 x repeatability%)^2): larger than either part, smaller than their sum
  const u = uncertaintyCm('waist', 100, 0.01)
  const rep = uncertaintyCm('waist', 100, 0)
  assert.ok(u > rep && u > 1 && u < rep + 1, `got ${u}`)
  assert.equal(uncertaintyCm('waist', null, 0.01), null)
})

test('asymmetry, invalid and tape statuses', () => {
  const values = { thigh_left: 60, thigh_right: 50, calf_left: 38, calf_right: 38, waist: null, hip: 100 }
  assert.ok(asymmetricSites(values).has('thigh_left'))
  assert.equal(siteStatus('thigh_right', values), 'asymmetric')
  assert.equal(siteStatus('waist', values), 'invalid')
  assert.equal(siteStatus('hip', values, { hip: { error_cm: 1.2 } }), 'tape_ok')
  assert.equal(siteStatus('hip', values, { hip: { error_cm: 4 } }), 'measured')
})

test('changes within repeatability are steady', () => {
  const rc = repeatabilityCoefficient('waist', 90)
  assert.equal(changeStatus('waist', 90, 90 + rc * 0.5), 'steady')
  assert.equal(changeStatus('waist', 90, 90 - rc * 2), 'smaller')
  assert.equal(changeStatus('waist', 90, 90 + rc * 2), 'larger')
  assert.equal(changeStatus('waist', null, 90), 'not_comparable')
})

test('a segment shows its most notable site status', () => {
  const trunk = CALLOUTS.find((c) => c.id === 'trunk')
  const s = segmentStatus(trunk, (site) => (site === 'hip' ? 'invalid' : 'measured'))
  assert.equal(s, 'invalid')
})

test('deltas carry an explicit sign', () => {
  assert.equal(formatDelta(-1.23, 'cm'), '−1.2')
  assert.equal(formatDelta(0.4, 'cm'), '+0.4')
})
