import test from 'node:test'
import assert from 'node:assert/strict'
import {
  profileParameters, groupParameters, formFromRecord, payloadFromForm, countEntered, segmentRows, hasSegmental,
  sourceBadge, originalReading, formatChange, fieldErrors, formatBcmValue,
} from './bcmModel.js'

const SCHEMA = {
  default_profile: 'generic_whole_body',
  parameters: [
    { key: 'weight', unit: 'kg', units: ['kg', 'lb'], group: 'whole_body', decimals: 1 },
    { key: 'body_fat_pct', unit: '%', units: ['%'], group: 'whole_body', decimals: 1 },
    { key: 'bmr', unit: 'kcal', units: ['kcal', 'kJ'], group: 'whole_body', decimals: 0 },
    { key: 'segment_fat_pct_trunk', unit: '%', units: ['%'], group: 'segmental_fat', segment: 'trunk', decimals: 1 },
  ],
  profiles: [
    { key: 'generic_whole_body', parameters: ['weight', 'body_fat_pct', 'bmr'] },
    { key: 'generic_segmental', parameters: ['weight', 'body_fat_pct', 'bmr', 'segment_fat_pct_trunk'] },
    { key: 'weight_only', parameters: ['weight'] },
  ],
}

test('profile limits the fields offered', () => {
  assert.deepEqual(profileParameters(SCHEMA, 'weight_only').map((p) => p.key), ['weight'])
  assert.deepEqual(groupParameters(profileParameters(SCHEMA, 'generic_segmental')).map((g) => g.key), ['whole_body', 'segmental_fat'])
  assert.deepEqual(profileParameters(SCHEMA, 'missing'), [])
})

test('empty form for a new record has blank values in canonical units', () => {
  const f = formFromRecord(null, SCHEMA, new Date(2026, 10, 19, 9, 5))
  assert.equal(f.profile, 'generic_whole_body')
  assert.equal(f.measured_at, '2026-11-19T09:05')
  assert.deepEqual(f.values.weight, { value: '', unit: 'kg' })
})

test('editing shows the original reading and unit, so lb stays lb', () => {
  const rec = { profile: { key: 'generic_whole_body' }, measured_at: '2026-11-18T08:00', source: { type: 'manual' }, note: 'n',
    values: { weight: { value: 80.01, unit: 'kg', original: { value: 176.4, unit: 'lb' } } } }
  const f = formFromRecord(rec, SCHEMA)
  assert.deepEqual(f.values.weight, { value: '176.4', unit: 'lb' })
  assert.equal(f.measured_at, '2026-11-18T08:00')
  assert.equal(f.note, 'n')
})

test('payload sends only the profile parameters and blanks as null', () => {
  const f = formFromRecord(null, SCHEMA)
  f.profile = 'weight_only'
  f.values.weight = { value: ' 72,5 ', unit: 'kg' }
  f.values.body_fat_pct = { value: '20', unit: '%' }   // not in weight_only -> dropped
  const p = payloadFromForm(f, SCHEMA)
  assert.deepEqual(p.values, { weight: { value: '72.5', unit: 'kg' } })
  assert.equal(countEntered(f, SCHEMA), 1)
  f.values.weight.value = ''
  assert.deepEqual(payloadFromForm(f, SCHEMA).values, { weight: { value: null, unit: 'kg' } })
})

test('segment rows pair camera girths with BCM composition and never invent BCM values', () => {
  const camera = { measurements_cm: { chest: 100, waist: 85, abdomen: 90, hip: 102, upper_arm_right: 31 }, site_valid: { hip: false } }
  const rows = segmentRows(camera, null)
  const trunk = rows.find((r) => r.key === 'trunk')
  assert.equal(trunk.unit, 'cm')
  assert.deepEqual(trunk.bcm, { fat_pct: null, fat_mass: null, muscle_mass: null })
  assert.equal(trunk.camera.find((c) => c.site === 'hip').valid, false)
  const rec = { values: { segment_fat_pct_trunk: { value: 24 } } }
  assert.equal(segmentRows(camera, rec).find((r) => r.key === 'trunk').bcm.fat_pct, 24)
  assert.equal(hasSegmental(rec), true)
  assert.equal(hasSegmental({ values: { weight: { value: 70 } } }), false)
  assert.equal(segmentRows({ measurements_ratio: { chest: 0.57 } }, null)[0].unit, 'ratio')
})

test('demo records are always badged DEMO DATA', () => {
  assert.deepEqual(sourceBadge({ source: { type: 'demo', is_demo: true } }), { text: 'DEMO DATA', tone: 'demo' })
  assert.equal(sourceBadge({ source: { type: 'manual', is_demo: false } }).text, 'Manual entry')
  assert.equal(sourceBadge(null), null)
})

test('formatting helpers', () => {
  assert.equal(originalReading({ value: 80, unit: 'kg', original: { value: 176.4, unit: 'lb' } }), '176.4 lb')
  assert.equal(originalReading({ value: 80, unit: 'kg', original: { value: 80, unit: 'kg' } }), null)
  assert.equal(formatChange(-2.04), '−2.0')
  assert.equal(formatChange(0.01), '±0')
  assert.equal(formatChange(null), null)
  assert.equal(formatBcmValue({ value: 1700.4 }, { decimals: 0 }), '1700')
  assert.equal(formatBcmValue(null, {}), '—')
  assert.deepEqual(fieldErrors({ message: 'x', errors: { weight: 'bad' } }), { weight: 'bad' })
  assert.deepEqual(fieldErrors('plain'), {})
})
