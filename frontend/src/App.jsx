/**
 * App.jsx
 * -------
 * Root application component.
 *
 * Orchestrates the full inference workflow:
 *   1. Image selection  (ImageUploader)
 *   2. Submit to FastAPI backend  (api.submitImage)
 *   3. Poll for job status every POLL_INTERVAL_MS  (api.getJobStatus)
 *   4. Render 3D mesh when completed  (MeshViewer)
 *   5. Show metadata + download link  (ResultPanel)
 *
 * State machine:
 *   idle → uploading → queued → processing → completed
 *                                          └→ failed
 */

import { useState, useRef, useEffect, useCallback } from 'react'
import Header from './components/Header.jsx'
import ImageUploader from './components/ImageUploader.jsx'
import ProcessingStatus from './components/ProcessingStatus.jsx'
import MeshViewer from './components/MeshViewer.jsx'
import ResultPanel from './components/ResultPanel.jsx'
import CaptureWorkspace from './components/CaptureWorkspace.jsx'
import { checkHealth, submitImage, getJobStatus, getMeshUrl } from './services/api.js'

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const POLL_INTERVAL_MS = 3000     // 3 s between status polls
const MAX_POLL_ATTEMPTS = 40      // ~2 min before giving up

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------
export default function App() {
  // Active workspace tab: "photo" (existing single-image flow) or
  // "capture" (new 360° video capture + frame extraction flow).
  const [activeTab, setActiveTab] = useState('photo')

  // Backend health
  const [backendOnline, setBackendOnline] = useState(null)  // null = checking

  // Selected image file
  const [imageFile, setImageFile] = useState(null)

  // Inference workflow state
  const [phase, setPhase] = useState('idle')       // see state machine above
  const [jobId, setJobId] = useState(null)
  const [attempt, setAttempt] = useState(0)
  const [error, setError] = useState(null)

  // Result
  const [meshUrl, setMeshUrl] = useState(null)
  const [metadata, setMetadata] = useState({})
  const [meshInfo, setMeshInfo] = useState(null)   // from Three.js loader

  // Poll timer ref – lets us clear the interval on cleanup
  const pollTimerRef = useRef(null)
  const pollAttemptRef = useRef(0)
  const jobIdRef = useRef(null)

  // ---------------------------------------------------------------------------
  // Backend health check – runs once on mount, then every 30 s
  // ---------------------------------------------------------------------------
  useEffect(() => {
    const check = async () => setBackendOnline(await checkHealth())
    check()
    const interval = setInterval(check, 30_000)
    return () => clearInterval(interval)
  }, [])

  // ---------------------------------------------------------------------------
  // Polling logic
  // ---------------------------------------------------------------------------
  const stopPolling = useCallback(() => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current)
      pollTimerRef.current = null
    }
  }, [])

  const startPolling = useCallback(
    (id) => {
      jobIdRef.current = id
      pollAttemptRef.current = 0

      pollTimerRef.current = setInterval(async () => {
        pollAttemptRef.current += 1
        setAttempt(pollAttemptRef.current)

        if (pollAttemptRef.current > MAX_POLL_ATTEMPTS) {
          stopPolling()
          setPhase('failed')
          setError(`Inference did not complete within ${(MAX_POLL_ATTEMPTS * POLL_INTERVAL_MS) / 1000}s.`)
          return
        }

        try {
          const status = await getJobStatus(jobIdRef.current)
          const s = status.status

          if (s === 'queued') {
            setPhase('queued')
          } else if (s === 'processing') {
            setPhase('processing')
          } else if (s === 'completed') {
            stopPolling()
            setPhase('completed')
            const url = getMeshUrl(status.mesh_url)
            setMeshUrl(url)
            setMetadata(status.metadata || {})
          } else if (s === 'failed' || s === 'error') {
            stopPolling()
            setPhase('failed')
            setError(status.detail || 'Inference failed on the server.')
          }
        } catch (err) {
          // Network hiccup – keep polling, don't abort
          console.warn('Polling error (will retry):', err.message)
        }
      }, POLL_INTERVAL_MS)
    },
    [stopPolling],
  )

  // Cleanup on unmount
  useEffect(() => () => stopPolling(), [stopPolling])

  // ---------------------------------------------------------------------------
  // Submit handler
  // ---------------------------------------------------------------------------
  const handleGenerate = useCallback(async () => {
    if (!imageFile) return

    // Reset result state
    setError(null)
    setMeshUrl(null)
    setMetadata({})
    setMeshInfo(null)
    setAttempt(0)
    setJobId(null)

    setPhase('uploading')

    try {
      const data = await submitImage(imageFile)
      const id = data.job_id
      setJobId(id)
      setPhase(data.status === 'queued' ? 'queued' : 'processing')
      startPolling(id)
    } catch (err) {
      setPhase('failed')
      setError(err.message || 'Failed to submit image.')
    }
  }, [imageFile, startPolling])

  // ---------------------------------------------------------------------------
  // Reset / try again
  // ---------------------------------------------------------------------------
  const handleReset = useCallback(() => {
    stopPolling()
    setImageFile(null)
    setPhase('idle')
    setJobId(null)
    setAttempt(0)
    setError(null)
    setMeshUrl(null)
    setMetadata({})
    setMeshInfo(null)
  }, [stopPolling])

  // ---------------------------------------------------------------------------
  // Derived flags
  // ---------------------------------------------------------------------------
  const isProcessing = ['uploading', 'queued', 'processing'].includes(phase)
  const hasResult = phase === 'completed' && meshUrl

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      <Header backendOnline={backendOnline === true} />

      <main className="flex-1 max-w-7xl mx-auto w-full px-4 sm:px-6 lg:px-8 py-8">
        {/* Backend offline warning */}
        {backendOnline === false && (
          <div className="mb-6 flex items-start gap-3 p-4 rounded-xl bg-amber-50 border border-amber-200 text-sm text-amber-800">
            <svg className="w-5 h-5 shrink-0 mt-0.5 text-amber-500" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd"
                d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z"
                clipRule="evenodd" />
            </svg>
            <div>
              <strong className="font-semibold">Backend is offline.</strong>{' '}
              Start the FastAPI server before running inference:
              <code className="ml-1 px-1.5 py-0.5 rounded bg-amber-100 font-mono text-xs">
                uvicorn backend.main:app --reload --port 8000
              </code>
            </div>
          </div>
        )}

        {/* Tab switcher */}
        <div className="flex gap-2 mb-6 border-b border-gray-200">
          <button
            onClick={() => setActiveTab('photo')}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'photo'
                ? 'border-teal-600 text-teal-700'
                : 'border-transparent text-gray-400 hover:text-gray-600'
            }`}
          >
            Single Photo
          </button>
          <button
            onClick={() => setActiveTab('capture')}
            className={`px-4 py-2 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'capture'
                ? 'border-teal-600 text-teal-700'
                : 'border-transparent text-gray-400 hover:text-gray-600'
            }`}
          >
            360° Video Capture
          </button>
        </div>

        {activeTab === 'capture' && <CaptureWorkspace />}

        {/* Two-column layout on large screens */}
        {activeTab === 'photo' && (
        <div className="grid grid-cols-1 lg:grid-cols-[380px_1fr] gap-6">

          {/* ── Left panel ──────────────────────────────────────────────── */}
          <div className="space-y-5">
            {/* Upload */}
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-5">
              <ImageUploader
                onFileSelected={setImageFile}
                disabled={isProcessing}
              />
            </div>

            {/* Generate button */}
            <button
              onClick={handleGenerate}
              disabled={!imageFile || isProcessing || backendOnline === false}
              className="w-full flex items-center justify-center gap-2 px-5 py-3 rounded-xl text-sm font-semibold
                bg-teal-600 hover:bg-teal-700 disabled:bg-gray-200 disabled:text-gray-400
                text-white transition-colors shadow-sm disabled:cursor-not-allowed"
            >
              {isProcessing ? (
                <>
                  <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Processing…
                </>
              ) : (
                <>
                  <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round"
                      d="M5.25 5.653c0-.856.917-1.398 1.667-.986l11.54 6.348a1.125 1.125 0 010 1.971l-11.54 6.347a1.125 1.125 0 01-1.667-.985V5.653z" />
                  </svg>
                  Generate 3D Mesh
                </>
              )}
            </button>

            {/* Processing status */}
            {phase !== 'idle' && (
              <ProcessingStatus
                phase={phase}
                jobId={jobId}
                attempt={attempt}
                maxAttempts={MAX_POLL_ATTEMPTS}
                error={error}
              />
            )}

            {/* Result panel (shown alongside viewer) */}
            {hasResult && (
              <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-5">
                <h3 className="text-sm font-semibold text-gray-700 mb-3">Step 3 — Mesh results</h3>
                <ResultPanel
                  meshUrl={meshUrl}
                  metadata={metadata}
                  meshInfo={meshInfo}
                  jobId={jobId}
                />
              </div>
            )}

            {/* Try again */}
            {(phase === 'completed' || phase === 'failed') && (
              <button
                onClick={handleReset}
                className="w-full px-4 py-2 rounded-xl border border-gray-300 hover:border-gray-400
                  text-sm text-gray-600 hover:text-gray-800 transition-colors bg-white"
              >
                Start over
              </button>
            )}
          </div>

          {/* ── Right panel: 3D viewer ───────────────────────────────────── */}
          <div className="space-y-3">
            <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
              <div className="px-5 py-4 border-b border-gray-100">
                <h2 className="text-base font-semibold text-gray-800">
                  Step 2 — 3D body mesh viewer
                </h2>
                <p className="text-xs text-gray-400 mt-0.5">
                  HMR2 / SMPL reconstruction from a single RGB photograph
                </p>
              </div>

              {/* Viewer or placeholder */}
              {hasResult ? (
                <MeshViewer
                  meshUrl={meshUrl}
                  onLoad={setMeshInfo}
                  onError={(msg) => console.error('MeshViewer error:', msg)}
                />
              ) : (
                <div className="flex flex-col items-center justify-center gap-4 bg-gray-900 text-white/30"
                  style={{ height: 520 }}>
                  <svg className="w-16 h-16 opacity-20" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={0.8}>
                    <path strokeLinecap="round" strokeLinejoin="round"
                      d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z" />
                  </svg>
                  <p className="text-sm">
                    {isProcessing
                      ? 'Generating mesh — please wait…'
                      : 'Upload a photograph and click Generate to see the 3D mesh here.'}
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-gray-200 bg-white py-4">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <p className="text-xs text-gray-400 text-center">
            Tanita Human Digital Twin — Proof of Concept &nbsp;·&nbsp;
            HMR2 inference runs locally on your GPU &nbsp;·&nbsp;
            Not for clinical use
          </p>
        </div>
      </footer>
    </div>
  )
}
