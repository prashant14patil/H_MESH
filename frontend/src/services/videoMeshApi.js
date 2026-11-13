const BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

async function request(path, options) {
  const res = await fetch(`${BASE_URL}${path}`, options)
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try { detail = (await res.json()).detail || detail } catch { /* use HTTP status */ }
    throw new Error(detail)
  }
  return res.json()
}

function batchPayload(videoId, frames, qualityRunId, reviewedFrames, reviewSessionId) {
  return { video_id: videoId, frames, quality_run_id: qualityRunId, reviewed_frames: reviewedFrames, review_session_id: reviewSessionId }
}

export function checkMeshReadiness(videoId, frames, qualityRunId, reviewedFrames, reviewSessionId) {
  return request('/api/video-mesh/readiness', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(batchPayload(videoId, frames, qualityRunId, reviewedFrames, reviewSessionId)),
  })
}

export function startMeshBatch(videoId, frames, qualityRunId, reviewedFrames, reviewSessionId, finalConfirmation) {
  return request('/api/video-mesh/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...batchPayload(videoId, frames, qualityRunId, reviewedFrames, reviewSessionId), final_confirmation: finalConfirmation === true }),
  })
}

export function getMeshBatch(runId) {
  return request(`/api/video-mesh/runs/${encodeURIComponent(runId)}`, {
    signal: AbortSignal.timeout(15000),
  })
}

export function meshAssetUrl(runId, kind, filename) {
  return `${BASE_URL}/api/video-mesh/runs/${encodeURIComponent(runId)}/${kind}/${encodeURIComponent(filename)}`
}
