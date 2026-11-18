const BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

export class ApiError extends Error {
  constructor(message, status, detail) {
    super(message)
    this.status = status
    this.detail = detail
  }
}

async function request(path, options) {
  const response = await fetch(`${BASE_URL}${path}`, options)
  let data = null
  try { data = await response.json() } catch { /* empty or non-JSON body */ }
  if (!response.ok) {
    const detail = data?.detail
    const message = typeof detail === 'string' ? detail : detail?.message || `HTTP ${response.status}`
    throw new ApiError(message, response.status, detail)
  }
  return data
}

const json = (method, body) => ({ method, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })
const review = path => `/api/subject-review${path}`

export function listSessions(videoId, qualityRunId) {
  const params = new URLSearchParams({ video_id: videoId, quality_run_id: qualityRunId })
  return request(review(`/sessions?${params}`))
}

export function createSession(videoId, qualityRunId, referenceFrame, bbox, replaceExisting = false) {
  return request(review('/sessions'), json('POST', {
    video_id: videoId, quality_run_id: qualityRunId, reference_frame: referenceFrame, bbox,
    replace_existing: replaceExisting === true,
  }))
}

export function resumeSession(sessionId, videoId, qualityRunId) {
  const params = new URLSearchParams({ video_id: videoId, quality_run_id: qualityRunId })
  return request(review(`/sessions/${encodeURIComponent(sessionId)}/resume?${params}`))
}

export function getSuggestion(sessionId, filename) {
  return request(review(`/sessions/${encodeURIComponent(sessionId)}/suggestions/${encodeURIComponent(filename)}`))
}

export function saveDecision(sessionId, filename, decision, bbox) {
  return request(review(`/sessions/${encodeURIComponent(sessionId)}/decisions`),
    json('POST', { filename, decision, bbox: decision === 'reject' ? null : bbox }))
}

export function saveSelection(sessionId, frames, acknowledged) {
  return request(review(`/sessions/${encodeURIComponent(sessionId)}/selection`), json('PUT', { frames, acknowledged }))
}

export function latestQualityReport(videoId) {
  return request(`/api/capture-quality/latest?${new URLSearchParams({ video_id: videoId })}`)
}
