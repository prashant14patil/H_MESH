/**
 * sizeRecommendation.js
 * ---------------------
 * Maps predicted girths (cm) to garment sizes using a size chart.
 *
 * The default chart is a generic EXAMPLE (unisex XS–3XL), not any brand's
 * chart. A brand chart has the same shape: for each driving measurement, the
 * lower bound of each size followed by the upper bound of the last size.
 */
export const SIZES = ['XS', 'S', 'M', 'L', 'XL', 'XXL', '3XL']

export const EXAMPLE_CHART = {
  id: 'example-unisex',
  name: 'Generic unisex (example)',
  example: true,
  sizes: SIZES,
  bounds: {
    chest: [82, 88, 96, 104, 112, 120, 128, 136],
    waist: [66, 72, 80, 88, 96, 104, 112, 120],
    hip: [84, 90, 98, 106, 114, 122, 130, 138],
  },
}

/** Fraction of the band treated as "at the edge" (snug / relaxed, or borderline). */
const EDGE = 0.2

/**
 * Place one girth on one measurement's size scale.
 * Returns null for a missing or non-finite value.
 */
export function placeOnChart(cm, bounds, sizes = SIZES) {
  if (cm == null || !Number.isFinite(cm)) return null
  if (cm < bounds[0]) return { index: 0, size: sizes[0], frac: 0, outside: 'below' }
  for (let i = 0; i < sizes.length; i++) {
    if (cm < bounds[i + 1]) {
      return { index: i, size: sizes[i], frac: (cm - bounds[i]) / (bounds[i + 1] - bounds[i]), outside: null }
    }
  }
  const last = sizes.length - 1
  return { index: last, size: sizes[last], frac: 1, outside: 'above' }
}

/** Plain-language fit for a placement, plus the neighbouring size when it sits on an edge. */
export function describeFit(p, sizes = SIZES) {
  if (!p) return { fit: 'unknown', alternative: null }
  if (p.outside === 'above') return { fit: 'above chart', alternative: null }
  if (p.outside === 'below') return { fit: 'below chart', alternative: null }
  if (p.frac >= 1 - EDGE) return { fit: 'snug', alternative: sizes[p.index + 1] ?? null }
  if (p.frac <= EDGE) return { fit: 'relaxed', alternative: sizes[p.index - 1] ?? null }
  return { fit: 'regular', alternative: null }
}

/**
 * Recommend a top size (chest) and a trouser size (larger of waist and hip).
 * `cm` is a map like { chest, waist, hip } in centimetres; missing values are skipped.
 */
export function recommendSizes(cm, chart = EXAMPLE_CHART) {
  const { sizes, bounds } = chart
  const place = (key) => (bounds[key] ? placeOnChart(cm?.[key], bounds[key], sizes) : null)
  const chest = place('chest')
  const waist = place('waist')
  const hip = place('hip')

  const top = chest ? { ...chest, by: 'chest', cm: cm.chest, ...describeFit(chest, sizes) } : null

  let bottom = null
  const candidates = [['waist', waist], ['hip', hip]].filter(([, p]) => p)
  if (candidates.length) {
    // The garment must close at both: the larger size wins; on a tie the tighter fit decides.
    candidates.sort((a, b) => b[1].index - a[1].index || b[1].frac - a[1].frac)
    const [by, p] = candidates[0]
    bottom = { ...p, by, cm: cm[by], ...describeFit(p, sizes) }
  }

  return { chart: chart.name, top, bottom, placements: { chest, waist, hip } }
}

export const CHART_KEYS = ['chest', 'waist', 'hip']
const MAX_SIZES = 20

/**
 * Validate a size chart loaded from JSON (shipped or uploaded by the operator).
 * Returns { chart, error }: exactly one of them is null. Error messages say what to fix.
 *
 * Format:
 *   { "id": "brand-mens", "name": "Brand X men's", "sizes": ["S","M","L"],
 *     "bounds": { "chest": [88,96,104,112], "waist": [...], "hip": [...] } }
 * Each bounds list has one more number than `sizes` (lower bound of every size,
 * then the upper bound of the last), strictly increasing, in centimetres.
 * At least one of chest / waist / hip is required; others are optional.
 */
export function validateChart(raw) {
  const fail = (error) => ({ chart: null, error })
  if (!raw || typeof raw !== 'object' || Array.isArray(raw)) return fail('The chart must be a JSON object.')
  const name = typeof raw.name === 'string' ? raw.name.trim() : ''
  if (!name) return fail('Add a "name" for the chart.')
  if (name.length > 80) return fail('The chart name must be 80 characters or fewer.')
  const sizes = raw.sizes
  if (!Array.isArray(sizes) || sizes.length < 1 || sizes.length > MAX_SIZES) {
    return fail(`"sizes" must be a list of 1 to ${MAX_SIZES} size labels.`)
  }
  if (!sizes.every((s) => typeof s === 'string' && s.trim() && s.length <= 12)) {
    return fail('Every size label must be text of 12 characters or fewer.')
  }
  if (new Set(sizes).size !== sizes.length) return fail('Size labels must be unique.')
  if (!raw.bounds || typeof raw.bounds !== 'object') return fail('Add "bounds" with chest, waist or hip ranges.')
  const unknown = Object.keys(raw.bounds).filter((k) => !CHART_KEYS.includes(k))
  if (unknown.length) return fail(`Unknown measurement "${unknown[0]}". Use chest, waist or hip.`)
  const bounds = {}
  for (const key of CHART_KEYS) {
    const b = raw.bounds[key]
    if (b == null) continue
    if (!Array.isArray(b) || b.length !== sizes.length + 1) {
      return fail(`"${key}" needs ${sizes.length + 1} numbers: the lower bound of each size, then the upper bound of the last.`)
    }
    if (!b.every((x) => typeof x === 'number' && Number.isFinite(x) && x > 20 && x < 300)) {
      return fail(`"${key}" values must be girths in centimetres (between 20 and 300).`)
    }
    for (let i = 1; i < b.length; i++) {
      if (!(b[i] > b[i - 1])) return fail(`"${key}" values must go up from one size to the next.`)
    }
    bounds[key] = b.slice()
  }
  if (!Object.keys(bounds).length) return fail('Give at least one of chest, waist or hip.')
  const id = typeof raw.id === 'string' && /^[a-z0-9-]{1,40}$/i.test(raw.id) ? raw.id : slug(name)
  return { chart: { id, name, example: raw.example === true, sizes: sizes.slice(), bounds }, error: null }
}

function slug(s) {
  return s.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '').slice(0, 40) || 'chart'
}

/**
 * Size change between two assessments (earlier -> later) on one chart.
 * Each of top / bottom is null when either side has no size; `steps` is the
 * number of sizes moved (positive = larger).
 */
export function sizeChange(beforeCm, afterCm, chart = EXAMPLE_CHART) {
  const a = recommendSizes(beforeCm, chart)
  const b = recommendSizes(afterCm, chart)
  const one = (x, y) => (x && y ? { before: x.size, after: y.size, steps: y.index - x.index } : null)
  return { top: one(a.top, b.top), bottom: one(a.bottom, b.bottom) }
}
