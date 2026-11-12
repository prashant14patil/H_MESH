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

export function startMeshBatch(videoId, frames) {
  return request('/api/video-mesh/runs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video_id: videoId, frames }),
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
