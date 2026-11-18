import test from 'node:test'
import assert from 'node:assert/strict'
import { parseTapeInputs, hasTapeValues } from './tapeValidation.js'

test('blank fields are skipped and numbers parsed', () => {
  const r = parseTapeInputs({ waist: ' 81.5 ', hip: '', chest: undefined })
  assert.deepEqual(r, { values: { waist: 81.5 }, errors: {} })
  assert.equal(hasTapeValues(r), true)
})

test('nothing entered cannot be submitted', () => {
  assert.equal(hasTapeValues(parseTapeInputs({})), false)
})

test('implausible or non-numeric values are flagged per site and block submission', () => {
  const r = parseTapeInputs({ waist: '80', hip: 'abc', calf_left: '5', thigh_right: '300' })
  assert.deepEqual(r.values, { waist: 80 })
  assert.deepEqual(Object.keys(r.errors).sort(), ['calf_left', 'hip', 'thigh_right'])
  assert.equal(hasTapeValues(r), false)
})

test('unknown keys are ignored', () => {
  assert.deepEqual(parseTapeInputs({ neck: '38' }).values, {})
})
