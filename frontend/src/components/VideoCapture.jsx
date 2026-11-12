/**
 * VideoCapture.jsx
 * ----------------
 * Module 1 — 360° Video Capture / Upload.
 *
 * Lets the user either:
 *   - upload an existing MP4/MOV video, or
 *   - record a new one live using getUserMedia + MediaRecorder
 *     (when a camera is available in the browser),
 * and shows a preview plus the standard 360°-scan capture instructions.
 *
 * This component is purely about acquiring a video File object — it
 * does NOT talk to the backend. The parent (CaptureWorkspace) owns the
 * upload/process/poll workflow, mirroring how ImageUploader.jsx only
 * produces a File and App.jsx owns the submit/poll logic for images.
 *
 * Props:
 *   onFileSelected(file: File | null)  – called whenever selection changes
 *   disabled: boolean                   – block interaction while processing
 */

import { useRef, useState, useCallback, useEffect } from 'react'

const MAX_SIZE_MB = 200
// video/webm is included because browser MediaRecorder output is WebM in
// virtually all browsers (Chrome, Firefox, Edge) -- there is no standard
// way to record directly to MP4 client-side, so the "record with camera"
// path produces WebM files that must pass validation here too.
const ALLOWED_TYPES = ['video/mp4', 'video/quicktime', 'video/x-m4v', 'video/webm']
const ALLOWED_EXT_LABEL = 'MP4, MOV, WEBM'

export default function VideoCapture({ onFileSelected, disabled = false }) {
  const [preview, setPreview] = useState(null)       // object URL string
  const [fileName, setFileName] = useState('')
  const [fileMeta, setFileMeta] = useState(null)      // { size }
  const [dragOver, setDragOver] = useState(false)
  const [error, setError] = useState('')
  const [mode, setMode] = useState('upload')          // 'upload' | 'record'

  // Recording state
  const [cameraAvailable, setCameraAvailable] = useState(null) // null = unknown
  const [isRecording, setIsRecording] = useState(false)
  const [recordSeconds, setRecordSeconds] = useState(0)

  const inputRef = useRef(null)
  const videoPreviewRef = useRef(null)
  const liveVideoRef = useRef(null)
  const mediaStreamRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const recordedChunksRef = useRef([])
  const recordTimerRef = useRef(null)

  // ---------------------------------------------------------------------
  // Detect camera availability once on mount
  // ---------------------------------------------------------------------
  useEffect(() => {
    setCameraAvailable(
      typeof navigator !== 'undefined' &&
      !!navigator.mediaDevices &&
      typeof navigator.mediaDevices.getUserMedia === 'function',
    )
  }, [])

  // ---------------------------------------------------------------------
  // Shared file-processing (used by both upload and record paths)
  // ---------------------------------------------------------------------
  const processFile = useCallback(
    (file) => {
      setError('')

      if (!file) {
        setPreview(null)
        setFileName('')
        setFileMeta(null)
        onFileSelected(null)
        return
      }

      // Match on MIME type OR filename extension. MediaRecorder output
      // often has a MIME type like "video/webm;codecs=av01" (with a
      // codecs parameter), so compare only the type's base (before ';').
      const baseType = (file.type || '').split(';')[0].trim()
      if (!ALLOWED_TYPES.includes(baseType) && !/\.(mp4|mov|m4v|webm)$/i.test(file.name)) {
        setError(`Unsupported file type "${file.type || 'unknown'}". Please use MP4, MOV, or WEBM.`)
        return
      }

      const sizeMB = file.size / (1024 * 1024)
      if (sizeMB > MAX_SIZE_MB) {
        setError(`File is ${sizeMB.toFixed(1)} MB — maximum allowed is ${MAX_SIZE_MB} MB.`)
        return
      }

      const objectUrl = URL.createObjectURL(file)
      setPreview(objectUrl)
      setFileName(file.name)
      setFileMeta({ size: file.size })
      onFileSelected(file)
    },
    [onFileSelected],
  )

  // ---------------------------------------------------------------------
  // Upload mode handlers
  // ---------------------------------------------------------------------
  const handleInputChange = (e) => {
    processFile(e.target.files?.[0] || null)
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    if (!disabled) setDragOver(true)
  }
  const handleDragLeave = () => setDragOver(false)
  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    if (disabled) return
    processFile(e.dataTransfer.files?.[0] || null)
  }

  const handleClear = () => {
    if (inputRef.current) inputRef.current.value = ''
    if (preview) URL.revokeObjectURL(preview)
    processFile(null)
  }

  // ---------------------------------------------------------------------
  // Record mode handlers
  // ---------------------------------------------------------------------
  const startCameraPreview = useCallback(async () => {
    setError('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      })
      mediaStreamRef.current = stream
      if (liveVideoRef.current) {
        liveVideoRef.current.srcObject = stream
        await liveVideoRef.current.play().catch(() => {})
      }
    } catch (err) {
      setError(`Could not access camera: ${err.message || err.name}`)
    }
  }, [])

  const stopCameraPreview = useCallback(() => {
    mediaStreamRef.current?.getTracks().forEach((t) => t.stop())
    mediaStreamRef.current = null
    if (liveVideoRef.current) liveVideoRef.current.srcObject = null
  }, [])

  const startRecording = useCallback(() => {
    if (!mediaStreamRef.current) return
    setError('')
    recordedChunksRef.current = []
    setRecordSeconds(0)

    let recorder
    try {
      recorder = new MediaRecorder(mediaStreamRef.current, { mimeType: 'video/webm' })
    } catch {
      try {
        recorder = new MediaRecorder(mediaStreamRef.current)
      } catch (err) {
        setError(`Recording is not supported in this browser: ${err.message}`)
        return
      }
    }

    recorder.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) recordedChunksRef.current.push(e.data)
    }
    recorder.onstop = () => {
      const blob = new Blob(recordedChunksRef.current, { type: recorder.mimeType || 'video/webm' })
      const ext = (recorder.mimeType || '').includes('mp4') ? 'mp4' : 'webm'
      const file = new File([blob], `recorded-360-scan.${ext}`, { type: blob.type })
      processFile(file)
      stopCameraPreview()
    }

    mediaRecorderRef.current = recorder
    recorder.start()
    setIsRecording(true)

    recordTimerRef.current = setInterval(() => {
      setRecordSeconds((s) => s + 1)
    }, 1000)
  }, [processFile, stopCameraPreview])

  const stopRecording = useCallback(() => {
    mediaRecorderRef.current?.stop()
    setIsRecording(false)
    if (recordTimerRef.current) {
      clearInterval(recordTimerRef.current)
      recordTimerRef.current = null
    }
  }, [])

  const switchToRecordMode = useCallback(() => {
    setMode('record')
    startCameraPreview()
  }, [startCameraPreview])

  const switchToUploadMode = useCallback(() => {
    if (isRecording) stopRecording()
    stopCameraPreview()
    setMode('upload')
  }, [isRecording, stopRecording, stopCameraPreview])

  // Cleanup camera/recorder on unmount
  useEffect(() => {
    return () => {
      mediaRecorderRef.current?.stop()
      mediaStreamRef.current?.getTracks().forEach((t) => t.stop())
      if (recordTimerRef.current) clearInterval(recordTimerRef.current)
      if (preview) URL.revokeObjectURL(preview)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const formatSeconds = (s) => {
    const m = Math.floor(s / 60)
    const r = s % 60
    return `${m}:${r.toString().padStart(2, '0')}`
  }

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-base font-semibold text-gray-800">Step 1 — Capture a 360° body scan</h2>
        <p className="text-sm text-gray-500 mt-0.5">
          Upload an existing video, or record one live if your device has a camera.
        </p>
      </div>

      {/* Mode toggle */}
      {!preview && (
        <div className="flex gap-2">
          <button
            type="button"
            onClick={switchToUploadMode}
            disabled={disabled}
            className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium border transition-colors
              ${mode === 'upload' ? 'bg-teal-600 text-white border-teal-600' : 'bg-white text-gray-600 border-gray-300 hover:border-teal-400'}
              ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            Upload video
          </button>
          <button
            type="button"
            onClick={switchToRecordMode}
            disabled={disabled || cameraAvailable === false}
            title={cameraAvailable === false ? 'No camera detected on this device/browser' : ''}
            className={`flex-1 px-3 py-2 rounded-lg text-sm font-medium border transition-colors
              ${mode === 'record' ? 'bg-teal-600 text-white border-teal-600' : 'bg-white text-gray-600 border-gray-300 hover:border-teal-400'}
              ${disabled || cameraAvailable === false ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            Record with camera
          </button>
        </div>
      )}

      {/* --- Upload drop zone --- */}
      {!preview && mode === 'upload' && (
        <div
          onClick={() => !disabled && inputRef.current?.click()}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          className={`relative flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-6 py-12 cursor-pointer transition-colors
            ${disabled ? 'opacity-50 cursor-not-allowed border-gray-200 bg-gray-50'
              : dragOver
                ? 'border-teal-500 bg-teal-50'
                : 'border-gray-300 bg-white hover:border-teal-400 hover:bg-teal-50/30'
            }`}
        >
          <div className="flex items-center justify-center w-12 h-12 rounded-full bg-teal-50 text-teal-600">
            <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
            </svg>
          </div>
          <div className="text-center">
            <p className="text-sm font-medium text-gray-700">
              {dragOver ? 'Drop your video here' : 'Click or drag & drop to upload'}
            </p>
            <p className="text-xs text-gray-400 mt-1">{ALLOWED_EXT_LABEL} · max {MAX_SIZE_MB} MB</p>
          </div>
          <input
            ref={inputRef}
            type="file"
            accept=".mp4,.mov,.m4v,.webm,video/mp4,video/quicktime,video/webm"
            className="sr-only"
            onChange={handleInputChange}
            disabled={disabled}
          />
        </div>
      )}

      {/* --- Record mode: live camera preview + controls --- */}
      {!preview && mode === 'record' && (
        <div className="space-y-3">
          <div className="relative rounded-xl overflow-hidden bg-gray-900" style={{ height: 320 }}>
            <video
              ref={liveVideoRef}
              className="w-full h-full object-cover"
              muted
              playsInline
            />
            {isRecording && (
              <div className="absolute top-3 left-3 flex items-center gap-2 px-2.5 py-1 rounded-full bg-red-600/90 text-white text-xs font-semibold">
                <span className="w-2 h-2 rounded-full bg-white animate-pulse" />
                REC {formatSeconds(recordSeconds)}
              </div>
            )}
          </div>
          <div className="flex justify-center gap-3">
            {!isRecording ? (
              <button
                type="button"
                onClick={startRecording}
                disabled={disabled}
                className="px-5 py-2 rounded-xl bg-red-600 hover:bg-red-700 text-white text-sm font-semibold transition-colors disabled:opacity-50"
              >
                ● Start recording
              </button>
            ) : (
              <button
                type="button"
                onClick={stopRecording}
                className="px-5 py-2 rounded-xl bg-gray-800 hover:bg-gray-900 text-white text-sm font-semibold transition-colors"
              >
                ■ Stop recording
              </button>
            )}
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="flex items-start gap-2 p-3 rounded-lg bg-red-50 border border-red-200 text-sm text-red-700">
          <svg className="w-4 h-4 mt-0.5 shrink-0" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd"
              d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.28 7.22a.75.75 0 00-1.06 1.06L8.94 10l-1.72 1.72a.75.75 0 101.06 1.06L10 11.06l1.72 1.72a.75.75 0 101.06-1.06L11.06 10l1.72-1.72a.75.75 0 00-1.06-1.06L10 8.94 8.28 7.22z"
              clipRule="evenodd" />
          </svg>
          {error}
        </div>
      )}

      {/* Video preview panel */}
      {preview && (
        <div className="space-y-3">
          <div className="rounded-xl overflow-hidden bg-gray-900" style={{ height: 320 }}>
            <video
              ref={videoPreviewRef}
              src={preview}
              className="w-full h-full object-contain"
              controls
            />
          </div>
          <div className="flex items-center justify-between p-3 rounded-lg bg-white border border-gray-200">
            <div className="min-w-0">
              <p className="text-sm font-medium text-gray-800 truncate" title={fileName}>
                {fileName}
              </p>
              {fileMeta && (
                <p className="text-xs text-gray-400 mt-0.5">
                  {(fileMeta.size / (1024 * 1024)).toFixed(1)} MB
                </p>
              )}
            </div>
            {!disabled && (
              <button
                onClick={handleClear}
                className="shrink-0 ml-3 text-xs text-gray-400 hover:text-red-500 transition-colors"
              >
                Remove video
              </button>
            )}
          </div>
        </div>
      )}

      {/* Capture instructions */}
      <div className="p-4 rounded-xl bg-teal-50 border border-teal-200">
        <h3 className="text-sm font-semibold text-teal-800 mb-2">360° capture instructions</h3>
        <ul className="space-y-1 text-xs text-teal-700 list-disc list-inside">
          <li>Keep the camera stationary (use a tripod or stand if possible).</li>
          <li>Ensure the entire body is visible in frame, head to feet.</li>
          <li>Person should slowly rotate a full 360° in place.</li>
          <li>Capture front, side, back, and intermediate views evenly.</li>
          <li>Maintain consistent lighting and a plain, uncluttered background.</li>
        </ul>
      </div>
    </div>
  )
}
