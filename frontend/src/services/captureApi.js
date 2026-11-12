/**
 * captureApi.js
 * -------------
 * All communication with the local FastAPI backend's 360° video
 * capture + frame extraction endpoints (Module 1 & 2).
 *
 * Mirrors the conventions established in services/api.js (same
 * BASE_URL handling, same error-throwing pattern) but targets the
 * /api/capture/* routes instead of /api/inference/*. Kept in its own
 * module so the existing single-image inference flow in api.js is
 * never touched.
 */

const BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

/**
 * Upload a 360° capture video (MP4/MOV) to the backend.
 *
 * @param {File} file
 * @returns {Promise<{video_id: string, status: string, filename: string}>}
 * @throws {Error} with a human-readable message on failure.
 */
export async function uploadVideo(file) {
  const formData = new FormData()
  formData.append('video', file, file.name)

  const res = await fetch(`${BASE_URL}/api/capture/upload`, {
    method: 'POST',
    body: formData,
  })

  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const err = await res.json()
      detail = err.detail || detail
    } catch { /* ignore */ }
    throw new Error(`Upload failed: ${detail}`)
  }

  return res.json()
}

/**
 * Start frame extraction for a previously uploaded video.
 *
 * @param {string} videoId
 * @returns {Promise<{video_id: string, status: string}>}
 * @throws {Error}
 */
export async function processVideo(videoId) {
  const res = await fetch(`${BASE_URL}/api/capture/process/${videoId}`, {
    method: 'POST',
  })

  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const err = await res.json()
      detail = err.detail || detail
    } catch { /* ignore */ }
    throw new Error(`Processing failed to start: ${detail}`)
  }

  return res.json()
}

/**
 * Poll the status of an upload/extraction job.
 *
 * @param {string} videoId
 * @returns {Promise<object>}
 * @throws {Error}
 */
export async function getCaptureStatus(videoId) {
  const res = await fetch(`${BASE_URL}/api/capture/status/${videoId}`, {
    signal: AbortSignal.timeout(15000),
  })

  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const err = await res.json()
      detail = err.detail || detail
    } catch { /* ignore */ }
    throw new Error(`Status check failed: ${detail}`)
  }

  return res.json()
}

/**
 * Fetch the full extracted-frame metadata (video info, per-frame
 * timestamps + quality scores) for a completed extraction job.
 *
 * @param {string} videoId
 * @returns {Promise<object>}
 * @throws {Error}
 */
export async function getFrameMetadata(videoId) {
  const res = await fetch(`${BASE_URL}/api/capture/frames/${videoId}`)

  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const err = await res.json()
      detail = err.detail || detail
    } catch { /* ignore */ }
    throw new Error(`Could not fetch frames: ${detail}`)
  }

  return res.json()
}

/**
 * Build the absolute URL for a single extracted frame image.
 *
 * @param {string} videoId
 * @param {string} filename  – e.g. "frame_001.jpg"
 * @returns {string}
 */
export function getFrameImageUrl(videoId, filename) {
  return `${BASE_URL}/api/capture/frame/${videoId}/${filename}`
}
