/**
 * api.js
 * ------
 * All communication with the local FastAPI backend.
 * This module never talks to any external service — the backend runs
 * HMR2 inference directly, in-process, on the local machine's GPU/CPU.
 *
 * Base URL is controlled by the VITE_API_BASE_URL env variable.
 * When running via `npm run dev` with the Vite proxy configured,
 * you can leave VITE_API_BASE_URL unset and calls go to /api/...
 * which the proxy forwards to http://localhost:8000.
 */

const BASE_URL = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

/**
 * Check whether the FastAPI backend is reachable.
 * @returns {Promise<boolean>}
 */
export async function checkHealth() {
  try {
    const res = await fetch(`${BASE_URL}/health`, { signal: AbortSignal.timeout(5000) })
    if (!res.ok) return false
    const data = await res.json()
    return data?.status === 'ok'
  } catch {
    return false
  }
}

/**
 * Submit an image file for HMR2 3D reconstruction.
 *
 * @param {File} file  – The image File object from the browser.
 * @returns {Promise<{job_id: string, status: string}>}
 * @throws {Error} with a human-readable message on failure.
 */
export async function submitImage(file) {
  const formData = new FormData()
  formData.append('image', file, file.name)

  const res = await fetch(`${BASE_URL}/api/inference/submit`, {
    method: 'POST',
    body: formData,
  })

  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const err = await res.json()
      detail = err.detail || detail
    } catch { /* ignore */ }
    throw new Error(`Submission failed: ${detail}`)
  }

  return res.json()
}

/**
 * Poll the backend for the current status of an inference job.
 *
 * Response shapes:
 *   processing: { job_id, status: "processing" }
 *   completed:  { job_id, status: "completed", mesh_url: "/api/inference/mesh/<id>",
 *                 metadata: { vertices, faces, body_model } }
 *   failed:     { job_id, status: "failed", detail: "..." }
 *
 * @param {string} jobId
 * @returns {Promise<object>}
 * @throws {Error} on network failure or non-200 response.
 */
export async function getJobStatus(jobId) {
  const res = await fetch(`${BASE_URL}/api/inference/status/${jobId}`, {
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
 * Return the absolute URL for the OBJ mesh file.
 * The URL points to the FastAPI backend, which serves the OBJ file that
 * was generated locally by the in-process HMR2 inference engine.
 *
 * @param {string} meshPath  – relative path like "/api/inference/mesh/<id>"
 *                             or an absolute URL.
 * @returns {string}
 */
export function getMeshUrl(meshPath) {
  if (!meshPath) return ''
  if (meshPath.startsWith('http')) return meshPath
  return `${BASE_URL}${meshPath}`
}
