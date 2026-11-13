/**
 * CaptureWorkspace.jsx
 * ---------------------
 * Orchestrates the full 360° video capture + frame extraction +
 * viewpoint-coverage workflow (Modules 1, 2, and 3's viewpoint
 * estimate):
 *
 *   1. Video selection (VideoCapture)
 *   2. Upload to FastAPI backend  (captureApi.uploadVideo)
 *   3. Trigger frame extraction  (captureApi.processVideo)
 *   4. Poll extraction status  (captureApi.getCaptureStatus)
 *   5. Once extraction completes, auto-trigger pose processing
 *      (poseApi.processPose) -- reuses the existing MediaPipe
 *      pipeline, no duplicate processing is started if a pose job is
 *      already running/completed for this video_id.
 *   6. Poll pose-processing status  (poseApi.getPoseStatus)
 *   7. Fetch viewpoint coverage  (poseApi.getViewpointCoverage)
 *   8. Render the extracted frame gallery + coverage report
 *      (FrameGallery)
 *
 * State machine:
 *   idle → uploading → uploaded → extracting → pose_processing → completed
 *                                                              └→ failed
 *
 * This component does not interact with the existing single-image
 * HMR2 inference flow in App.jsx -- it is rendered as a separate tab
 * so the two workflows never interfere with each other.
 */

import { useState, useRef, useEffect, useCallback } from 'react'
import VideoCapture from './VideoCapture.jsx'
import FrameGallery from './FrameGallery.jsx'
import VideoMeshPanel from './VideoMeshPanel.jsx'
import CaptureQualityPanel from './CaptureQualityPanel.jsx'
import SubjectReviewPanel from './SubjectReviewPanel.jsx'
import {
  uploadVideo,
  processVideo,
  getCaptureStatus,
  getFrameMetadata,
} from '../services/captureApi.js'
import {
  processPose,
  getPoseStatus,
  getViewpointCoverage,
  getPoseResults,
} from '../services/poseApi.js'

const POLL_INTERVAL_MS = 2000
const MAX_POLL_ATTEMPTS = 60 // ~2 min before giving up per stage

export default function CaptureWorkspace() {
  const [videoFile, setVideoFile] = useState(null)
  // idle|uploading|uploaded|extracting|pose_processing|completed|failed
  const [phase, setPhase] = useState('idle')
  const [videoId, setVideoId] = useState(null)
  const [attempt, setAttempt] = useState(0)
  const [error, setError] = useState(null)
  const [frameMetadata, setFrameMetadata] = useState(null)
  const [viewpointData, setViewpointData] = useState(null)
  const [viewpointError, setViewpointError] = useState(null)
  const [poseResults, setPoseResults] = useState(null)
  const [qualityReport, setQualityReport] = useState(null)
  const [reviewSession, setReviewSession] = useState(null)
  const [reviewDecisions, setReviewDecisions] = useState({})
  const [reviewFrame, setReviewFrame] = useState(null)
  const [savedVideoId, setSavedVideoId] = useState('')
  const [savedRunId, setSavedRunId] = useState('')
  const [restoring, setRestoring] = useState(false)
  const [restoreVersion, setRestoreVersion] = useState(0)
  const [uploadAuthorized, setUploadAuthorized] = useState(false)
  const [uploadConsent, setUploadConsent] = useState(false)
  const [savedAuthorized, setSavedAuthorized] = useState(false)
  const [savedConsent, setSavedConsent] = useState(false)

  const pollTimerRef = useRef(null)
  const pollAttemptRef = useRef(0)
  const videoIdRef = useRef(null)

  const stopPolling = useCallback(() => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current)
      pollTimerRef.current = null
    }
  }, [])

  // -----------------------------------------------------------------
  // Stage 2: poll pose-processing status, then fetch viewpoint coverage
  // -----------------------------------------------------------------
  const startPosePolling = useCallback((id) => {
    pollAttemptRef.current = 0

    pollTimerRef.current = setInterval(async () => {
      pollAttemptRef.current += 1
      setAttempt(pollAttemptRef.current)

      if (pollAttemptRef.current > MAX_POLL_ATTEMPTS) {
        stopPolling()
        setPhase('completed') // frames are still usable even without viewpoints
        setViewpointError(`Pose processing did not complete within ${(MAX_POLL_ATTEMPTS * POLL_INTERVAL_MS) / 1000}s.`)
        return
      }

      try {
        const status = await getPoseStatus(id)
        const s = status.status

        if (s === 'processing') {
          setPhase('pose_processing')
        } else if (s === 'completed') {
          stopPolling()
          try {
            const [coverage, results] = await Promise.all([
              getViewpointCoverage(id),
              getPoseResults(id),
            ])
            setViewpointData(coverage)
            setPoseResults(results)
          } catch (err) {
            setViewpointError(err.message || 'Failed to fetch viewpoint coverage.')
          }
          setPhase('completed')
        } else if (s === 'failed') {
          stopPolling()
          setViewpointError(status.detail || 'Pose processing failed on the server.')
          setPhase('completed') // frame extraction result is still valid/usable
        }
      } catch (err) {
        console.warn('Pose polling error (will retry):', err.message)
      }
    }, POLL_INTERVAL_MS)
  }, [stopPolling])

  // -----------------------------------------------------------------
  // Stage 1: poll frame-extraction status, then auto-trigger pose
  // -----------------------------------------------------------------
  const startExtractionPolling = useCallback(
    (id) => {
      videoIdRef.current = id
      pollAttemptRef.current = 0

      pollTimerRef.current = setInterval(async () => {
        pollAttemptRef.current += 1
        setAttempt(pollAttemptRef.current)

        if (pollAttemptRef.current > MAX_POLL_ATTEMPTS) {
          stopPolling()
          setPhase('failed')
          setError(`Frame extraction did not complete within ${(MAX_POLL_ATTEMPTS * POLL_INTERVAL_MS) / 1000}s.`)
          return
        }

        try {
          const status = await getCaptureStatus(videoIdRef.current)
          const s = status.status

          if (s === 'processing') {
            setPhase('extracting')
          } else if (s === 'completed') {
            stopPolling()
            try {
              const meta = await getFrameMetadata(videoIdRef.current)
              setFrameMetadata(meta)
            } catch (err) {
              setPhase('failed')
              setError(err.message || 'Failed to fetch extracted frames.')
              return
            }

            // Auto-trigger pose processing (reuses existing Module 3
            // pipeline; backend no-ops if already processing/completed).
            try {
              await processPose(videoIdRef.current)
              setPhase('pose_processing')
              startPosePolling(videoIdRef.current)
            } catch (err) {
              // Pose processing failing to start shouldn't hide the
              // already-successful frame extraction result.
              setViewpointError(err.message || 'Failed to start pose processing.')
              setPhase('completed')
            }
          } else if (s === 'failed') {
            stopPolling()
            setPhase('failed')
            setError(status.detail || 'Frame extraction failed on the server.')
          }
        } catch (err) {
          console.warn('Capture polling error (will retry):', err.message)
        }
      }, POLL_INTERVAL_MS)
    },
    [stopPolling, startPosePolling],
  )

  useEffect(() => () => stopPolling(), [stopPolling])

  async function reopenSaved(event) {
    event.preventDefault()
    setRestoring(true)
    setError(null)
    try {
      const id = savedVideoId.trim()
      const runId = savedRunId.trim()
      const uuid = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/
      if (!uuid.test(id) || !uuid.test(runId)) throw new Error('Enter valid recording and report IDs')
      if (!savedAuthorized || !savedConsent) throw new Error('Confirm authorization and consent before opening this recording')
      const base = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')
      const [metadata, response] = await Promise.all([
        getFrameMetadata(id),
        fetch(`${base}/api/capture-quality/runs/${encodeURIComponent(runId)}/report`),
      ])
      if (!response.ok) throw new Error(`Quality report unavailable (HTTP ${response.status})`)
      const report = await response.json()
      if (metadata.status !== 'completed' || metadata.video_id !== id ||
          report.video_id !== id || report.run_id !== runId || !Array.isArray(report.frames)) {
        throw new Error('Capture and completed quality report do not match')
      }
      const confirmation = await fetch(`${base}/api/capture/${encodeURIComponent(id)}/authorization`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ authorized: savedAuthorized, consent: savedConsent }),
      })
      if (!confirmation.ok) {
        const detail = await confirmation.json().catch(() => ({}))
        throw new Error(detail.detail || `Authorization unavailable (HTTP ${confirmation.status})`)
      }
      stopPolling()
      setReviewSession(null); setReviewDecisions({}); setReviewFrame(null)
      setRestoreVersion(version => version + 1)
      setVideoId(id); setFrameMetadata(metadata); setQualityReport(report)
      setPoseResults(null); setViewpointData(null); setViewpointError(null)
      setPhase('completed')
    } catch (err) {
      setError(`Could not reopen capture: ${err.message}`)
    } finally {
      setRestoring(false)
    }
  }

  const handleProcess = useCallback(async () => {
    if (!videoFile || !uploadAuthorized || !uploadConsent) return

    setError(null)
    setViewpointError(null)
    setFrameMetadata(null)
    setViewpointData(null)
    setPoseResults(null)
    setQualityReport(null)
    setReviewSession(null); setReviewDecisions({}); setReviewFrame(null)
    setAttempt(0)
    setVideoId(null)
    setPhase('uploading')

    try {
      const uploadResult = await uploadVideo(videoFile, { authorized: uploadAuthorized, consent: uploadConsent })
      const id = uploadResult.video_id
      setVideoId(id)
      setPhase('uploaded')

      await processVideo(id)
      setPhase('extracting')
      startExtractionPolling(id)
    } catch (err) {
      setPhase('failed')
      setError(err.message || 'Failed to upload/process video.')
    }
  }, [videoFile, uploadAuthorized, uploadConsent, startExtractionPolling])

  const handleReset = useCallback(() => {
    stopPolling()
    setVideoFile(null)
    setUploadAuthorized(false); setUploadConsent(false); setSavedAuthorized(false); setSavedConsent(false)
    setPhase('idle')
    setVideoId(null)
    setAttempt(0)
    setError(null)
    setFrameMetadata(null)
    setViewpointData(null)
    setViewpointError(null)
    setPoseResults(null)
    setQualityReport(null)
    setReviewSession(null); setReviewDecisions({}); setReviewFrame(null)
  }, [stopPolling])

  const isProcessing = ['uploading', 'uploaded', 'extracting', 'pose_processing'].includes(phase)
  const hasResult = phase === 'completed' && frameMetadata

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[380px_1fr] gap-6">
      {/* Left panel — capture */}
      <div className="space-y-5">
        <div className="bg-white rounded-2xl border border-gray-200 shadow-sm p-5">
          <VideoCapture onFileSelected={file => { setVideoFile(file); setUploadAuthorized(false); setUploadConsent(false) }} disabled={isProcessing || restoring} />
        </div>

        <div className="bg-white rounded-xl border p-4 text-xs space-y-2">
          <h3 className="font-semibold text-sm">Authorization and consent for selected video</h3>
          <p>Operator assertion only; this application cannot verify consent or identity. Confirm for this specific video before uploading and processing it.</p>
          <label className="flex gap-2"><input type="checkbox" checked={uploadAuthorized} onChange={e => setUploadAuthorized(e.target.checked)} disabled={!videoFile || isProcessing || restoring} /> I am authorized to process this recording and generate meshes.</label>
          <label className="flex gap-2"><input type="checkbox" checked={uploadConsent} onChange={e => setUploadConsent(e.target.checked)} disabled={!videoFile || isProcessing || restoring} /> Everyone depicted has consented to this processing and mesh generation.</label>
        </div>

        <button
          onClick={handleProcess}
          disabled={!videoFile || !uploadAuthorized || !uploadConsent || isProcessing || restoring}
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
              {phase === 'uploading' && 'Uploading…'}
              {phase === 'uploaded' && 'Starting extraction…'}
              {phase === 'extracting' && 'Extracting frames…'}
              {phase === 'pose_processing' && 'Analyzing pose & viewpoints…'}
            </>
          ) : (
            'Process Video'
          )}
        </button>

        <form onSubmit={reopenSaved} className="bg-white rounded-xl border p-4 space-y-2">
          <h3 className="font-semibold text-sm">Reopen saved capture</h3>
          <p className="text-xs text-gray-600">Enter the existing recording and quality-report IDs. This does not approve frames.</p>
          <label className="block text-xs">Recording ID
            <input required disabled={restoring || isProcessing} value={savedVideoId} onChange={e => { setSavedVideoId(e.target.value); setSavedAuthorized(false); setSavedConsent(false) }} className="block w-full border rounded p-1" />
          </label>
          <label className="block text-xs">Quality report ID
            <input required disabled={restoring || isProcessing} value={savedRunId} onChange={e => setSavedRunId(e.target.value)} className="block w-full border rounded p-1" />
          </label>
          <p className="text-xs">Operator assertion only; the application cannot verify consent or identity. Confirm for this specific recording.</p>
          <label className="flex gap-2 text-xs"><input type="checkbox" checked={savedAuthorized} onChange={e => setSavedAuthorized(e.target.checked)} disabled={restoring || isProcessing} /> I am authorized to process this recording and generate meshes.</label>
          <label className="flex gap-2 text-xs"><input type="checkbox" checked={savedConsent} onChange={e => setSavedConsent(e.target.checked)} disabled={restoring || isProcessing} /> Everyone depicted has consented to this processing and mesh generation.</label>
          <button disabled={restoring || isProcessing || !savedAuthorized || !savedConsent} className="border rounded px-3 py-2 text-sm disabled:text-gray-400">
            {restoring ? 'Opening…' : 'Open saved capture'}
          </button>
        </form>

        {/* Status */}
        {phase !== 'idle' && (
          <div className="p-4 rounded-xl bg-white border border-gray-200 shadow-sm text-sm">
            <StatusRow phase={phase} videoId={videoId} attempt={attempt} maxAttempts={MAX_POLL_ATTEMPTS} />
            {error && (
              <p className="mt-2 text-red-600 text-xs">{error}</p>
            )}
            {viewpointError && (
              <p className="mt-2 text-amber-600 text-xs">
                Viewpoint analysis: {viewpointError}
              </p>
            )}
          </div>
        )}

        {phase === 'idle' && error && <p role="alert" className="text-red-700 text-xs">{error}</p>}
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

      {/* Right panel — gallery */}
      <div className="space-y-3 min-w-0">
        <div className="bg-white rounded-2xl border border-gray-200 shadow-sm overflow-hidden">
          <div className="px-5 py-4 border-b border-gray-100">
            <h2 className="text-base font-semibold text-gray-800">Step 2 — Extracted frames & viewpoint coverage</h2>
            <p className="text-xs text-gray-400 mt-0.5">
              Representative frames sampled from the 360° scan, with quality filtering and
              pose-based viewpoint estimation
            </p>
          </div>

          <div className="p-5">
            {hasResult ? (
              <FrameGallery
                videoId={videoId}
                metadata={frameMetadata}
                viewpointData={viewpointData}
                poseResults={poseResults}
                posePhase={phase}
              />
            ) : (
              <div className="flex flex-col items-center justify-center gap-3 text-gray-300" style={{ height: 320 }}>
                <svg className="w-14 h-14 opacity-30" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={0.8}>
                  <path strokeLinecap="round" strokeLinejoin="round"
                    d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909m-18-3.75h16.5a1.5 1.5 0 001.5-1.5V6a1.5 1.5 0 00-1.5-1.5H3.75A1.5 1.5 0 002.25 6v12a1.5 1.5 0 001.5 1.5zm10.5-11.25h.008v.008h-.008V8.25zm.375 0a.375.375 0 11-.75 0 .375.375 0 01.75 0z" />
                </svg>
                <p className="text-sm text-gray-400">
                  {isProcessing ? 'Processing — please wait…' : 'Upload or record a video and click Process to see extracted frames here.'}
                </p>
              </div>
            )}
          </div>
        </div>
        {hasResult && <CaptureQualityPanel key={`quality-${videoId}`} videoId={videoId} onReport={setQualityReport} restoredReport={qualityReport}
          reviewSession={reviewSession?.quality_run_id === qualityReport?.run_id ? reviewSession : null} reviewDecisions={reviewDecisions} />}
        {hasResult && qualityReport && (
          <SubjectReviewPanel key={`${qualityReport.run_id}-${restoreVersion}`} videoId={videoId} report={qualityReport}
            requestedFrame={reviewFrame} decisions={reviewDecisions}
            onSession={(session, restoredDecisions = {}) => { setReviewSession(session); setReviewDecisions(restoredDecisions) }}
            onDecision={decision => setReviewDecisions(old => ({ ...old, [decision.frame_id]: decision }))} />
        )}
        {hasResult && (
          <VideoMeshPanel
            key={`${videoId}-${restoreVersion}`}
            videoId={videoId}
            metadata={frameMetadata}
            viewpointData={viewpointData}
            poseResults={poseResults}
            qualityReport={qualityReport}
            reviewSession={reviewSession?.quality_run_id === qualityReport?.run_id ? reviewSession : null}
            reviewDecisions={reviewDecisions}
            onReviewFrame={name => {
              setReviewFrame({ name })
              document.getElementById('subject-review')?.scrollIntoView({ behavior: 'smooth', block: 'start' })
            }}
          />
        )}
      </div>
    </div>
  )
}

function StatusRow({ phase, videoId, attempt, maxAttempts }) {
  const labelMap = {
    uploading: 'Uploading video…',
    uploaded: 'Video uploaded, starting extraction…',
    extracting: 'Extracting frames…',
    pose_processing: 'Analyzing pose & viewpoints…',
    completed: 'Processing completed.',
    failed: 'Processing failed.',
  }

  const colorMap = {
    uploading: 'text-blue-600',
    uploaded: 'text-blue-600',
    extracting: 'text-amber-600',
    pose_processing: 'text-amber-600',
    completed: 'text-emerald-600',
    failed: 'text-red-600',
  }

  return (
    <div className="flex items-center justify-between">
      <span className={`font-medium ${colorMap[phase] || 'text-gray-600'}`}>
        {labelMap[phase] || phase}
      </span>
      {(phase === 'extracting' || phase === 'pose_processing') && (
        <span className="text-xs text-gray-400">attempt {attempt}/{maxAttempts}</span>
      )}
      {videoId && (
        <span className="text-[10px] text-gray-300 font-mono truncate ml-2" title={videoId}>
          {videoId.slice(0, 8)}
        </span>
      )}
    </div>
  )
}
