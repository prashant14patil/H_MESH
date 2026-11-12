/**
 * poseApi.js
 * ----------
 * All communication with the local FastAPI backend's Module 3 (human
 * detection, segmentation, pose estimation) endpoints.
 *
 * Mirrors the conventions established in captureApi.js (same BASE_URL
 * handling, same error-throwing pattern) but targets the /api/pose/*
 * routes. Kept in its own module so Modules 1/2 (captureApi.js) and
 * the single-image inference flow (api.js) are never touched.
 */

const BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

/**
 * Start pose/segmentation processing for a video's extracted frames.
 *
 * @param {string} videoId
 * @returns {Promise<{video_id: string, status: string}>}
 * @throws {Error}
 */
export async function processPose(videoId) {
  const res = await fetch(`${BASE_URL}/api/pose/process/${videoId}`, {
    method: 'POST',
  })

  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const err = await res.json()
      detail = err.detail || detail
    } catch { /* ignore */ }
    throw new Error(`Pose processing failed to start: ${detail}`)
  }

  return res.json()
}

/**
 * Poll the status of a pose-processing job.
 *
 * @param {string} videoId
 * @returns {Promise<object>}
 * @throws {Error}
 */
export async function getPoseStatus(videoId) {
  const res = await fetch(`${BASE_URL}/api/pose/status/${videoId}`, {
    signal: AbortSignal.timeout(15000),
  })

  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const err = await res.json()
      detail = err.detail || detail
    } catch { /* ignore */ }
    throw new Error(`Pose status check failed: ${detail}`)
  }

  return res.json()
}

/**
 * Fetch the full pose/segmentation processing summary (per-frame
 * landmarks, detection confidence, segmentation status, review flags).
 *
 * @param {string} videoId
 * @returns {Promise<object>}
 * @throws {Error}
 */
export async function getPoseResults(videoId) {
  const res = await fetch(`${BASE_URL}/api/pose/results/${videoId}`)

  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const err = await res.json()
      detail = err.detail || detail
    } catch { /* ignore */ }
    throw new Error(`Could not fetch pose results: ${detail}`)
  }

  return res.json()
}

/**
 * Build the absolute URL for a processed-frame asset.
 *
 * @param {string} videoId
 * @param {string} frameName  – e.g. "frame_001"
 * @param {string} asset      – "segmentation_mask.png" | "pose_visualization.jpg" | "metadata.json"
 * @returns {string}
 */
export function getPoseFrameAssetUrl(videoId, frameName, asset) {
  return `${BASE_URL}/api/pose/frame/${videoId}/${frameName}/${asset}`
}

/**
 * Fetch the per-frame viewpoint (orientation) classification and 360
 * coverage summary for a video. Derived from the existing pose
 * landmarks (GET /api/pose/results/{video_id}) -- requires pose
 * processing to have completed first via processPose().
 *
 * @param {string} videoId
 * @returns {Promise<object>} { video_id, frames: [...], coverage: {...} }
 * @throws {Error}
 */
export async function getViewpointCoverage(videoId) {
  const res = await fetch(`${BASE_URL}/api/pose/viewpoints/${videoId}`)

  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const err = await res.json()
      detail = err.detail || detail
    } catch { /* ignore */ }
    throw new Error(`Could not fetch viewpoint coverage: ${detail}`)
  }

  return res.json()
}
