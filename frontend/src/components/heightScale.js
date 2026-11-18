/**
 * heightScale.js
 * --------------
 * Parses the operator-entered barefoot height used to give the reconstruction an
 * absolute scale. Bounds match the backend (backend/services/body_measurements.py).
 */
export const MIN_HEIGHT_CM = 100
export const MAX_HEIGHT_CM = 250

/** Empty input is allowed (no scale); anything else must be a plausible height in cm. */
export function parseHeightCm(text) {
  const raw = String(text ?? '').trim()
  if (raw === '') return { value: null, error: null }
  const value = Number(raw)
  if (!Number.isFinite(value)) return { value: null, error: 'Enter a number in centimetres' }
  if (value < MIN_HEIGHT_CM || value > MAX_HEIGHT_CM) {
    return { value: null, error: `Height must be ${MIN_HEIGHT_CM}–${MAX_HEIGHT_CM} cm` }
  }
  return { value, error: null }
}
