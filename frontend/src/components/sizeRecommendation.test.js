import test from 'node:test'
import assert from 'node:assert/strict'
import { EXAMPLE_CHART, describeFit, placeOnChart, recommendSizes, sizeChange, validateChart } from './sizeRecommendation.js'

const chest = EXAMPLE_CHART.bounds.chest

test('places a girth inside the right band', () => {
  const p = placeOnChart(100, chest)
  assert.equal(p.size, 'M')
  assert.equal(p.frac, 0.5)
  assert.equal(describeFit(p).fit, 'regular')
})

test('band edges: lower bound belongs to the band, upper bound to the next', () => {
  assert.equal(placeOnChart(96, chest).size, 'M')
  assert.equal(placeOnChart(104, chest).size, 'L')
})

test('near the top of a band is snug and suggests sizing up', () => {
  const d = describeFit(placeOnChart(111.7, chest))
  assert.deepEqual(d, { fit: 'snug', alternative: 'XL' })
})

test('near the bottom of a band is relaxed and suggests sizing down', () => {
  assert.deepEqual(describeFit(placeOnChart(96.5, chest)), { fit: 'relaxed', alternative: 'S' })
})

test('outside the chart is flagged, not silently clamped', () => {
  assert.equal(placeOnChart(70, chest).outside, 'below')
  assert.equal(placeOnChart(150, chest).outside, 'above')
  assert.equal(describeFit(placeOnChart(150, chest)).fit, 'above chart')
})

test('missing values give no placement', () => {
  assert.equal(placeOnChart(null, chest), null)
  assert.equal(placeOnChart(NaN, chest), null)
  const r = recommendSizes({ chest: 100 })
  assert.equal(r.top.size, 'M')
  assert.equal(r.bottom, null)
})

test('same trouser size from waist and hip: the tighter one decides', () => {
  // run 268f55fd at 175 cm: chest 111.7, waist 98.1 (XL, 26% in), hip 114.5 (XL, 6% in)
  const r = recommendSizes({ chest: 111.7, waist: 98.1, hip: 114.5 })
  assert.equal(r.top.size, 'L')
  assert.equal(r.top.fit, 'snug')
  assert.equal(r.bottom.size, 'XL')
  assert.equal(r.bottom.by, 'waist')
})

test('hip drives trousers when it needs the bigger size', () => {
  const r = recommendSizes({ waist: 90, hip: 115 })
  assert.equal(r.bottom.size, 'XL')
  assert.equal(r.bottom.by, 'hip')
})

test('waist drives trousers when it needs the bigger size', () => {
  const r = recommendSizes({ waist: 105, hip: 100 })
  assert.equal(r.bottom.size, 'XXL')
  assert.equal(r.bottom.by, 'waist')
})


const good = { name: 'Brand men', sizes: ['S', 'M', 'L'], bounds: { chest: [88, 96, 104, 112] } }

test('a well-formed chart validates and gets an id', () => {
  const { chart, error } = validateChart(good)
  assert.equal(error, null)
  assert.equal(chart.id, 'brand-men')
  assert.deepEqual(chart.bounds, { chest: [88, 96, 104, 112] })
})

test('the shipped example chart is valid', () => {
  assert.equal(validateChart(EXAMPLE_CHART).error, null)
})

test('chart errors say what to fix', () => {
  assert.match(validateChart(null).error, /JSON object/)
  assert.match(validateChart({ ...good, name: '' }).error, /name/)
  assert.match(validateChart({ ...good, bounds: { chest: [88, 96, 104] } }).error, /needs 4 numbers/)
  assert.match(validateChart({ ...good, bounds: { chest: [88, 96, 96, 112] } }).error, /go up/)
  assert.match(validateChart({ ...good, bounds: { inseam: [1, 2, 3, 4] } }).error, /Unknown measurement "inseam"/)
  assert.match(validateChart({ ...good, bounds: {} }).error, /at least one/)
  assert.match(validateChart({ ...good, sizes: ['S', 'S', 'L'] }).error, /unique/)
  assert.match(validateChart({ ...good, bounds: { chest: [0.5, 0.6, 0.7, 0.8] } }).error, /centimetres/)
})

test('a chart without hip still recommends trousers from waist', () => {
  const { chart } = validateChart({ name: 'Waist only', sizes: ['M', 'L'], bounds: { waist: [80, 90, 100] } })
  const r = recommendSizes({ chest: 100, waist: 95, hip: 110 }, chart)
  assert.equal(r.top, null)
  assert.equal(r.bottom.size, 'L')
  assert.equal(r.bottom.by, 'waist')
})

test('size change between two assessments', () => {
  const c = sizeChange({ chest: 100, waist: 85, hip: 100 }, { chest: 106, waist: 85, hip: 100 })
  assert.deepEqual(c.top, { before: 'M', after: 'L', steps: 1 })
  assert.deepEqual(c.bottom, { before: 'M', after: 'M', steps: 0 })
  assert.equal(sizeChange({ chest: 100 }, { waist: 80 }).top, null)
})
