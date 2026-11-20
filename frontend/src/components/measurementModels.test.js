import test from 'node:test'
import assert from 'node:assert/strict'
import { DEFAULT_VIEW, VIEWS, compareRows, loadView } from './measurementModels.js'

test('uses cm when both models have cm and flags >3 cm disagreements', () => {
  const pipeline = { measurements: { waist: 0.49, hip: 0.57 }, measurements_cm: { waist: 86.4, hip: 101.0 } }
  const shapy = { measurements: { waist: 0.49, hip: 0.6 }, measurements_cm: { waist: 86.3, hip: 105.2 } }
  const { unit, rows } = compareRows(pipeline, shapy, ['waist', 'hip'])
  assert.equal(unit, 'cm')
  assert.equal(rows[0].disagree, false)
  assert.ok(Math.abs(rows[0].diff + 0.1) < 1e-9)
  assert.equal(rows[1].disagree, true)
})

test('falls back to relative units when no height is set', () => {
  const { unit, rows } = compareRows({ measurements: { waist: 0.49 } }, { measurements: { waist: 0.52 } }, ['waist'])
  assert.equal(unit, '× height')
  assert.equal(rows[0].disagree, true)
})

test('missing or invalid sites give no diff and no flag', () => {
  const { rows } = compareRows({ measurements: { calf_left: null } }, { measurements: { calf_left: 0.2 } }, ['calf_left'])
  assert.deepEqual(rows[0], { site: 'calf_left', pipeline: null, shapy: 0.2, diff: null, disagree: false })
})

test('view preference defaults to comparing both models without browser storage', () => {
  assert.equal(loadView(), 'compare')
  assert.equal(DEFAULT_VIEW, 'compare')
})

test('Compare both is the first tab, SHAPY before Pipeline, and every view id is unique', () => {
  assert.deepEqual(VIEWS.map(([id]) => id), ['compare', 'shapy', 'pipeline'])
})
