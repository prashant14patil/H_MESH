import test from 'node:test'
import assert from 'node:assert/strict'
import { parseHeightCm } from './heightScale.js'

test('empty height means no metric scale and no error', () => {
  assert.deepEqual(parseHeightCm(''), { value: null, error: null })
  assert.deepEqual(parseHeightCm('   '), { value: null, error: null })
  assert.deepEqual(parseHeightCm(undefined), { value: null, error: null })
})

test('plausible heights parse to numbers', () => {
  assert.deepEqual(parseHeightCm('172'), { value: 172, error: null })
  assert.deepEqual(parseHeightCm(' 165.5 '), { value: 165.5, error: null })
  assert.equal(parseHeightCm('100').value, 100)
  assert.equal(parseHeightCm('250').value, 250)
})

test('non-numeric or implausible heights are rejected', () => {
  for (const bad of ['abc', '1.72m', 'NaN', 'Infinity', '99.9', '251', '-170']) {
    const r = parseHeightCm(bad)
    assert.equal(r.value, null, bad)
    assert.ok(r.error, bad)
  }
})
