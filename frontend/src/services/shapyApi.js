/**
 * shapyApi.js
 * -----------
 * SHAPY measurement model for completed reconstruction runs (/api/reconstruction/...).
 * SHAPY is computed once per run on the server; centimetres follow the run's current height.
 */
const BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

async function request(path, options) {
  const res = await fetch(`${BASE_URL}${path}`, options)
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try { detail = (await res.json()).detail || detail } catch { /* keep HTTP status */ }
    throw new Error(typeof detail === 'string' ? detail : JSON.stringify(detail))
  }
  return res.json()
}

export async function listMeasurementModels() {
  return (await request('/api/reconstruction/models')).models
}

/** Returns the stored SHAPY result, computing it first if needed. */
export function computeShapy(runId) {
  return request(`/api/reconstruction/runs/${encodeURIComponent(runId)}/shapy`, { method: 'POST' })
}

export function shapyMeshUrl(runId) {
  return `${BASE_URL}/api/reconstruction/runs/${encodeURIComponent(runId)}/shapy/mesh`
}
