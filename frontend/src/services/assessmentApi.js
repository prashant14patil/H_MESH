/**
 * assessmentApi.js
 * ----------------
 * Client for /api/assessments/* (camera assessment + optional BCM data) and /api/bcm/* (schema, device status).
 * Errors carry `.status` and `.detail` so the BCM form can show per-field validation messages.
 */

const BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

async function request(path, options = {}) {
  const res = await fetch(`${BASE_URL}${path}`, {
    ...options,
    headers: options.body ? { 'Content-Type': 'application/json', ...(options.headers || {}) } : options.headers,
  })
  if (res.status === 204) return null
  let body = null
  try { body = await res.json() } catch { /* empty body */ }
  if (!res.ok) {
    const detail = body?.detail ?? `HTTP ${res.status}`
    const msg = typeof detail === 'string' ? detail : detail?.message || JSON.stringify(detail)
    // FastAPI's generic "Not Found" means the route itself is missing: the backend predates this API.
    const err = new Error(detail === 'Not Found'
      ? 'The backend does not provide the assessments API yet — restart it to load the new routes.' : msg)
    err.status = res.status
    err.detail = detail
    throw err
  }
  return body
}

const enc = encodeURIComponent

export const listAssessments = async (personId = null) =>
  (await request(`/api/assessments${personId ? `?person_id=${enc(personId)}` : ''}`)).assessments
export const listPersons = async () => (await request('/api/assessments/persons')).persons
export const getAssessment = (id) => request(`/api/assessments/${enc(id)}`)
export const setAssessmentPerson = (id, personId) =>
  request(`/api/assessments/${enc(id)}/person`, { method: 'PUT', body: JSON.stringify({ person_id: personId }) })
export const compareAssessments = (before, after) =>
  request(`/api/assessments/compare?before=${enc(before)}&after=${enc(after)}`)
export const personHistory = (personId) => request(`/api/assessments/persons/${enc(personId)}/history`)

export const saveBcm = (id, payload) =>
  request(`/api/assessments/${enc(id)}/bcm`, { method: 'PUT', body: JSON.stringify(payload) })
export const saveDemoBcm = (id, sampleSet = 1, profile = 'generic_segmental') =>
  request(`/api/assessments/${enc(id)}/bcm/demo`, { method: 'POST', body: JSON.stringify({ sample_set: sampleSet, profile }) })
export const importBcm = (id, payload, filename) =>
  request(`/api/assessments/${enc(id)}/bcm/import`, { method: 'POST', body: JSON.stringify({ format: 'bcm-json-v1', filename, payload }) })
export const removeBcm = (id) => request(`/api/assessments/${enc(id)}/bcm`, { method: 'DELETE' })
export const bcmHistory = async (id) => (await request(`/api/assessments/${enc(id)}/bcm/history`)).events

let schemaPromise = null
export function getBcmSchema() {
  schemaPromise ??= request('/api/bcm/schema').catch((e) => { schemaPromise = null; throw e })
  return schemaPromise
}
export const getDeviceStatus = () => request('/api/bcm/devices')
