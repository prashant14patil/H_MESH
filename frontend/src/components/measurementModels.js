/**
 * measurementModels.js
 * --------------------
 * Pure helpers for choosing and comparing measurement models (SHAPY / Pipeline / Compare).
 * SHAPY is the default view: in the first tape check it was within a few cm, while the pipeline was 4-14 cm too
 * large in loose clothing (docs/SHAPY_VS_PIPELINE_MEASUREMENTS.md).
 */
export const VIEWS = [
  ['shapy', 'SHAPY'],
  ['pipeline', 'Pipeline'],
  ['compare', 'Compare side by side'],
]
export const DEFAULT_VIEW = 'shapy'
export const DISAGREE_CM = 3
export const DISAGREE_REL = 0.017 // ~3 cm on a 176 cm person, used when no height is set

// '.v2': the default changed from Pipeline to SHAPY; a new key starts every viewer on the new default once.
const STORAGE_KEY = 'measurementModelView.v2'

export function loadView() {
  try {
    const v = window.localStorage.getItem(STORAGE_KEY)
    return VIEWS.some(([id]) => id === v) ? v : DEFAULT_VIEW
  } catch {
    return DEFAULT_VIEW
  }
}

export function saveView(view) {
  try { window.localStorage.setItem(STORAGE_KEY, view) } catch { /* per-viewer convenience only */ }
}

/**
 * Rows for the comparison table. Uses cm when both sides have them, otherwise × height.
 * `pipeline` and `shapy` are { measurements: {site: rel}, measurements_cm?: {site: cm} }.
 */
export function compareRows(pipeline, shapy, sites) {
  const useCm = Boolean(pipeline?.measurements_cm && shapy?.measurements_cm)
  const pick = (src, site) => (useCm ? src?.measurements_cm?.[site] : src?.measurements?.[site])
  const limit = useCm ? DISAGREE_CM : DISAGREE_REL
  return {
    unit: useCm ? 'cm' : '× height',
    rows: sites.map((site) => {
      const a = pick(pipeline, site)
      const b = pick(shapy, site)
      const ok = typeof a === 'number' && typeof b === 'number'
      const diff = ok ? b - a : null
      return { site, pipeline: a ?? null, shapy: b ?? null, diff, disagree: ok && Math.abs(diff) > limit }
    }),
  }
}
