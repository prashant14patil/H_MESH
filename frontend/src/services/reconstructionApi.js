/**
 * reconstructionApi.js
 * --------------------
 * Client for the unified 360° reconstruction endpoints (/api/reconstruction/*).
 * One video in, one body mesh out. Consent was attested at upload time and is
 * re-checked by the server on every run.
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

export async function startReconstruction(videoId, targetBoxNorm = null) {
  try {
    return await request('/api/reconstruction/runs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ video_id: videoId, target_box_norm: targetBoxNorm }),
    })
  } catch (e) {
    // FastAPI's generic 404 means the route itself is missing: the running backend
    // predates the reconstruction endpoints (e.g. started before an update, no --reload).
    if (e.message === 'Not Found') {
      throw new Error('The backend does not provide the reconstruction endpoint (POST /api/reconstruction/runs returned 404). '
        + 'It is probably running older code — restart it with: .venv\\Scripts\\python.exe -m uvicorn backend.main:app --port 8000')
    }
    throw e
  }
}

export function getReconstruction(runId) {
  return request(`/api/reconstruction/runs/${encodeURIComponent(runId)}`)
}

export function confirmPerson(runId, candidateId) {
  return request(`/api/reconstruction/runs/${encodeURIComponent(runId)}/person`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ candidate_id: candidateId }),
  })
}

export function reconstructionFileUrl(runId, name) {
  return `${BASE_URL}/api/reconstruction/runs/${encodeURIComponent(runId)}/files/${encodeURIComponent(name)}`
}
