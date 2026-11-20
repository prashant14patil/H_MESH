import test from 'node:test'
import assert from 'node:assert/strict'
import {
  parseObj, parseRings, centroid, scanValues, comparableValues, interpolateValues, uncertaintyCm,
  siteStatus, changeStatus, segmentStatus, asymmetricSites, formatDelta, CALLOUTS, repeatabilityCoefficient,
  transferLabels, tapeErrors, meanAbsError,
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

const meshOf = (pts) => ({ positions: new Float32Array(pts.flat()), vertexCount: pts.length })

test('transferLabels copies the nearest labelled vertex, independent of overall size', () => {
  // Source: a "head" vertex on top (label 0) and two "arm" vertices left/right (labels 2, 3), plus feet at y=0.
  const source = meshOf([[0, 1.7, 0], [-0.6, 1.3, 0], [0.6, 1.3, 0], [0, 0, 0]])
  const labels = [0, 2, 3, 1]
  // Target: the same layout 10% taller, with extra vertices near each source vertex and a different count.
  const target = meshOf([[0.02, 1.86, 0], [-0.64, 1.42, 0.01], [0.66, 1.44, 0], [0.01, 0.02, 0], [0, 1.8, 0.03]])
  assert.deepEqual([...transferLabels(target, source, labels)], [0, 2, 3, 1, 0])
})

test('transferLabels is exact nearest neighbour across grid cells', () => {
  const source = meshOf([[0, 0, 0], [0, 1, 0], [0.1, 0.5, 0]])
  const target = meshOf([[0.05, 0.5, 0], [0, 0.97, 0], [0, 0, 0], [0, 1, 0]]) // same 0..1 height as the source
  // A tiny cell forces the search across many shells; the answer must not change.
  assert.deepEqual([...transferLabels(target, source, [7, 8, 9], 0.01)], [9, 8, 7, 8])
  assert.deepEqual([...transferLabels(target, source, [7, 8, 9])], [9, 8, 7, 8])
})

test('tapeErrors scores any model against the run tape and meanAbsError averages it', () => {
  const tape = { waist: { tape_cm: 89, model_cm: 100.3, error_cm: 11.3 }, hip: { tape_cm: 100.5 }, chest: { tape_cm: null } }
  const shapy = tapeErrors({ waist: 88.5, hip: 105.0, chest: 106.1 }, tape)
  assert.deepEqual(Object.keys(shapy).sort(), ['hip', 'waist'])
  assert.ok(Math.abs(shapy.waist.error_cm + 0.5) < 1e-9)
  assert.ok(Math.abs(meanAbsError(shapy).cm - 2.5) < 1e-9)
  assert.equal(meanAbsError(shapy).n, 2)
  assert.equal(meanAbsError({}), null)
  assert.deepEqual(tapeErrors({ waist: 90 }, null), {})
})
