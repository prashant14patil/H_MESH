/**
 * tapeValidation.js
 * -----------------
 * Parses operator-entered tape circumferences (cm) for comparison with the model.
 * Bounds and site names match backend/services/body_measurements.py.
 */
export const TAPE_SITES = ['chest', 'waist', 'abdomen', 'hip', 'upper_arm_left', 'upper_arm_right',
  'forearm_left', 'forearm_right', 'thigh_left', 'thigh_right', 'calf_left', 'calf_right']

export const CLOTHING_OPTIONS = [
  ['minimal', 'Minimal (underwear / swimwear)'],
  ['tight', 'Tight-fitting'],
  ['loose', 'Loose'],
  ['unknown', 'Not recorded'],
]

/** Blank fields are skipped. Returns { values, errors } where errors is keyed by site. */
export function parseTapeInputs(texts) {
  const values = {}
  const errors = {}
  for (const site of TAPE_SITES) {
    const raw = String(texts?.[site] ?? '').trim()
    if (raw === '') continue
    const v = Number(raw)
    if (!Number.isFinite(v)) errors[site] = 'not a number'
    else if (v < 10 || v > 250) errors[site] = '10–250 cm'
    else values[site] = v
  }
  return { values, errors }
}

export const hasTapeValues = ({ values, errors }) => Object.keys(values).length > 0 && Object.keys(errors).length === 0
