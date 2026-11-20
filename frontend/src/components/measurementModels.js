/**
 * measurementModels.js
 * --------------------
 * Pure helpers for choosing and comparing measurement models (Compare both / SHAPY / Pipeline).
 * The default view shows both models as Body Avatars side by side, SHAPY first. In the first tape check SHAPY was
 * about 2 cm outside the tape ranges; the pipeline 4-14 cm too large with its old prior 0 and about 3 cm with the
 * HMR2 prior 0.2 that is the default from 20 Nov (docs/SHAPY_VS_PIPELINE_MEASUREMENTS.md).
 */
export const VIEWS = [
  ['compare', 'Compare both'],
  ['shapy', 'SHAPY'],
  ['pipeline', 'Pipeline'],
]
export const DEFAULT_VIEW = 'compare'
export const DISAGREE_CM = 3
export const DISAGREE_REL = 0.017 // ~3 cm on a 176 cm person, used when no height is set

// A new key whenever the default changes, so every viewer starts on the new default once ('.v3': Compare both).
const STORAGE_KEY = 'measurementModelView.v3'

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
