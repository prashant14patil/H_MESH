/**
 * bodyAvatarModel.js
 * ------------------
 * Pure logic behind the Body Avatar view: OBJ parsing, which measurement belongs to which body
 * segment, segment status colours, change between two scans and per-site uncertainty.
 * No React / three.js here so it can be unit-tested with `node --test`.
 */

export const SEGMENTS = ['head', 'trunk', 'left_arm', 'right_arm', 'left_leg', 'right_leg']

/** Callout cards: the primary girth is shown large, the others underneath. */
export const CALLOUTS = [
  { id: 'left_arm', label: 'Left arm', segment: 'left_arm', primary: 'upper_arm_left', secondary: ['forearm_left'] },
  { id: 'right_arm', label: 'Right arm', segment: 'right_arm', primary: 'upper_arm_right', secondary: ['forearm_right'] },
  { id: 'trunk', label: 'Trunk', segment: 'trunk', primary: 'waist', secondary: ['chest', 'hip'] },
  { id: 'left_leg', label: 'Left leg', segment: 'left_leg', primary: 'thigh_left', secondary: ['calf_left'] },
  { id: 'right_leg', label: 'Right leg', segment: 'right_leg', primary: 'thigh_right', secondary: ['calf_right'] },
]

export const SITE_LABELS = {
  chest: 'Chest', waist: 'Waist', abdomen: 'Abdomen', hip: 'Hip',
  upper_arm_left: 'Upper arm', upper_arm_right: 'Upper arm', forearm_left: 'Forearm', forearm_right: 'Forearm',
  thigh_left: 'Thigh', thigh_right: 'Thigh', calf_left: 'Calf', calf_right: 'Calf',
}

export const SITE_FULL_LABELS = {
  ...SITE_LABELS,
  upper_arm_left: 'Upper arm (L)', upper_arm_right: 'Upper arm (R)', forearm_left: 'Forearm (L)',
  forearm_right: 'Forearm (R)', thigh_left: 'Thigh (L)', thigh_right: 'Thigh (R)', calf_left: 'Calf (L)', calf_right: 'Calf (R)',
}

/**
 * Recording-to-recording repeatability of each girth as a percentage (1 SD), measured with
 * scripts/measurement_accuracy_study.py on THREE independent recordings of one subject with identical fit
 * settings (shape prior 0.2, non-saturating outside term); see docs/MEASUREMENT_ACCURACY_REPEATABILITY_REPORT.md.
 * One subject: a first estimate, not an established figure. Used to judge whether a change between scans is
 * larger than noise. Replace with the validation report's values as more recordings are added.
 */
export const REPEATABILITY_SD_PCT = {
  chest: 1.0, waist: 2.0, abdomen: 2.0, hip: 1.5, upper_arm_left: 2.4, upper_arm_right: 2.3,
  forearm_left: 1.0, forearm_right: 0.9, thigh_left: 1.8, thigh_right: 1.9, calf_left: 1.1, calf_right: 1.1,
}

/** 95% repeatability coefficient (Bland-Altman): two scans of an unchanged body differ by less than this. */
export const repeatabilityCoefficient = (site, value) =>
  (Number.isFinite(value) ? 2.77 * (REPEATABILITY_SD_PCT[site] ?? 2) / 100 * value : null)

export const ASYMMETRY_LIMIT = 0.05 // left/right girths differing by more than 5% are flagged
export const TAPE_TOLERANCE_CM = 2.0

// ------------------------------------------------------------------------- OBJ
/** Indexed mesh from OBJ text: keeps SMPL's vertex order, which the segment map relies on. */
export function parseObj(text) {
  const pos = []
  const idx = []
  for (const line of String(text).split('\n')) {
    if (line.startsWith('v ')) {
      const p = line.trim().split(/\s+/)
      pos.push(+p[1], +p[2], +p[3])
    } else if (line.startsWith('f ')) {
      const p = line.trim().split(/\s+/).slice(1).map((s) => parseInt(s, 10) - 1)
      for (let k = 1; k + 1 < p.length; k++) idx.push(p[0], p[k], p[k + 1])
    }
  }
  return { positions: new Float32Array(pos), indices: new Uint32Array(idx), vertexCount: pos.length / 3 }
}

/** measurement_rings.obj: one named polyline per valid site -> { site: [[x,y,z], ...] }. */
export function parseRings(text) {
  const rings = {}
  let current = null
  for (const line of String(text).split('\n')) {
    if (line.startsWith('o ')) {
      current = line.slice(2).trim()
      rings[current] = []
    } else if (line.startsWith('v ') && current) {
      const p = line.trim().split(/\s+/)
      rings[current].push([+p[1], +p[2], +p[3]])
    }
  }
  return rings
}

export function centroid(points) {
  if (!points?.length) return null
  const c = [0, 0, 0]
  for (const p of points) { c[0] += p[0]; c[1] += p[1]; c[2] += p[2] }
  return c.map((v) => v / points.length)
}

export function decodeLabels(segmentJson) {
  const s = segmentJson.labels
  const out = new Uint8Array(s.length)
  for (let i = 0; i < s.length; i++) out[i] = s.charCodeAt(i) - 48
  return out
}

// ------------------------------------------------------------------- values
/**
 * One scan's girths in display units: cm when the run was scaled from a supplied height, otherwise
 * the scale-free fraction of body height.
 */
export function scanValues(run) {
  const ratios = run?.measurements || {}
  const cm = run?.metric_scale?.measurements_cm || run?.measurements_cm || null
  const heightCm = run?.metric_scale?.height_cm ?? run?.height_cm ?? null
  const unit = cm ? 'cm' : 'ratio'
  const values = {}
  for (const [site, r] of Object.entries(ratios)) {
    values[site] = unit === 'cm' ? (Number.isFinite(cm[site]) ? cm[site] : null) : (Number.isFinite(r) ? r : null)
  }
  return { unit, values, heightCm, relScaleUnc: run?.metric_scale?.relative_scale_uncertainty ?? (heightCm ? 1 / heightCm : null) }
}

/** Two scans are comparable in cm only if both are scaled; otherwise compare scale-free ratios. */
export function comparableValues(current, previous) {
  const a = scanValues(current)
  const b = scanValues(previous)
  if (a.unit === b.unit) return { unit: a.unit, now: a.values, before: b.values }
  const ratio = (run) => Object.fromEntries(Object.entries(run?.measurements || {})
    .map(([k, v]) => [k, Number.isFinite(v) ? v : null]))
  return { unit: 'ratio', now: ratio(current), before: ratio(previous) }
}

export const lerp = (a, b, t) => (Number.isFinite(a) && Number.isFinite(b) ? a + (b - a) * t : (t >= 1 ? b : a))

export function interpolateValues(before, now, t) {
  const out = {}
  for (const k of new Set([...Object.keys(before || {}), ...Object.keys(now || {})])) {
    out[k] = lerp(before?.[k], now?.[k], t)
  }
  return out
}

/** Combined 1-SD uncertainty (cm) of a cm girth: height scale + run-to-run repeatability. */
export function uncertaintyCm(site, valueCm, relScaleUnc) {
  if (!Number.isFinite(valueCm)) return null
  const scale = Number.isFinite(relScaleUnc) ? valueCm * relScaleUnc : 0
  const rep = (REPEATABILITY_SD_PCT[site] ?? 2) / 100 * valueCm
  return Math.sqrt(scale * scale + rep * rep)
}

export function formatValue(v, unit) {
  if (!Number.isFinite(v)) return '—'
  return unit === 'cm' ? v.toFixed(1) : v.toFixed(3)
}

export function formatDelta(d, unit) {
  if (!Number.isFinite(d)) return '—'
  const s = unit === 'cm' ? Math.abs(d).toFixed(1) : Math.abs(d).toFixed(3)
  return `${d > 0 ? '+' : d < 0 ? '−' : '±'}${s}`
}

// ------------------------------------------------------------------- status
export const STATUS = {
  measured: { label: 'Measured', color: '#5ec4a8' },
  tape_ok: { label: `Tape-checked (within ±${TAPE_TOLERANCE_CM} cm)`, color: '#6b9de8' },
  asymmetric: { label: 'Left/right differ > 5%', color: '#e2c25a' },
  invalid: { label: 'Section not measurable', color: '#e59a4c' },
  smaller: { label: 'Smaller than before', color: '#6b9de8' },
  steady: { label: 'Within repeatability', color: '#5ec4a8' },
  larger: { label: 'Larger than before', color: '#e2c25a' },
  not_comparable: { label: 'Not comparable', color: '#c4c9d2' },
}
export const HEAD_COLOR = '#e6e9ee'

const PAIRS = [['upper_arm_left', 'upper_arm_right'], ['forearm_left', 'forearm_right'],
  ['thigh_left', 'thigh_right'], ['calf_left', 'calf_right']]

export function asymmetricSites(values) {
  const flagged = new Set()
  for (const [l, r] of PAIRS) {
    const a = values?.[l]
    const b = values?.[r]
    if (Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) / ((a + b) / 2) > ASYMMETRY_LIMIT) {
      flagged.add(l); flagged.add(r)
    }
  }
  return flagged
}

/** Status of one site for a single scan. */
export function siteStatus(site, values, tapeSites) {
  if (!Number.isFinite(values?.[site])) return 'invalid'
  if (asymmetricSites(values).has(site)) return 'asymmetric'
  const tape = tapeSites?.[site]
  if (tape && Number.isFinite(tape.error_cm) && Math.abs(tape.error_cm) <= TAPE_TOLERANCE_CM) return 'tape_ok'
  return 'measured'
}

/** Change of one site between scans, judged against the repeatability coefficient. */
export function changeStatus(site, before, now) {
  if (!Number.isFinite(before) || !Number.isFinite(now)) return 'not_comparable'
  const rc = repeatabilityCoefficient(site, (before + now) / 2)
  const d = now - before
  if (Math.abs(d) <= rc) return 'steady'
  return d < 0 ? 'smaller' : 'larger'
}

const SEVERITY = ['invalid', 'asymmetric', 'not_comparable', 'larger', 'smaller', 'measured', 'tape_ok', 'steady']

/** A segment shows the most notable status of its sites. */
export function segmentStatus(callout, statusOf) {
  const sites = [callout.primary, ...callout.secondary]
  const statuses = sites.map(statusOf)
  return statuses.sort((a, b) => SEVERITY.indexOf(a) - SEVERITY.indexOf(b))[0]
}

export function legendFor(mode) {
  return mode === 'compare' ? ['smaller', 'steady', 'larger', 'not_comparable'] : ['measured', 'tape_ok', 'asymmetric', 'invalid']
}

export function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16)
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255]
}

export function formatDateTime(epochSeconds) {
  if (!Number.isFinite(epochSeconds)) return '—'
  return new Date(epochSeconds * 1000).toLocaleString('en-GB', { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })
}

export function formatDate(epochSeconds) {
  if (!Number.isFinite(epochSeconds)) return '—'
  return new Date(epochSeconds * 1000).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })
}
