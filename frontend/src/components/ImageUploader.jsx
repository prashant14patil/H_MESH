/**
 * ImageUploader.jsx
 * -----------------
 * Drag-and-drop + file-picker image upload panel with inline preview.
 * Accepts JPG, JPEG, PNG up to 10 MB.
 *
 * Props:
 *   onFileSelected(file: File | null)  – called whenever selection changes
 *   disabled: boolean                  – block interaction while processing
 */

import { useRef, useState, useCallback } from 'react'

const MAX_SIZE_MB = 10
const ALLOWED_TYPES = ['image/jpeg', 'image/jpg', 'image/png']
const ALLOWED_EXT_LABEL = 'JPG, JPEG, PNG'

export default function ImageUploader({ onFileSelected, disabled = false }) {
  const [preview, setPreview] = useState(null)     // object URL string
  const [fileName, setFileName] = useState('')
  const [fileMeta, setFileMeta] = useState(null)   // { size, width, height }
  const [dragOver, setDragOver] = useState(false)
  const [error, setError] = useState('')
  const inputRef = useRef(null)

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

      // Validate type
      if (!ALLOWED_TYPES.includes(file.type)) {
        setError(`Unsupported file type "${file.type}". Please upload a JPEG or PNG.`)
        return
      }

      // Validate size
      const sizeMB = file.size / (1024 * 1024)
      if (sizeMB > MAX_SIZE_MB) {
        setError(`File is ${sizeMB.toFixed(1)} MB — maximum allowed is ${MAX_SIZE_MB} MB.`)
        return
      }

      // Build preview
      const objectUrl = URL.createObjectURL(file)
      const img = new Image()
      img.onload = () => {
        setFileMeta({ size: file.size, width: img.naturalWidth, height: img.naturalHeight })
        URL.revokeObjectURL(objectUrl)
      }
      img.src = objectUrl

      setPreview(URL.createObjectURL(file))
      setFileName(file.name)
      onFileSelected(file)
    },
    [onFileSelected],
  )

  // File input change
  const handleInputChange = (e) => {
    processFile(e.target.files?.[0] || null)
  }

  // Drag events
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
    processFile(null)
  }

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-base font-semibold text-gray-800">Step 1 — Upload a photograph</h2>
        <p className="text-sm text-gray-500 mt-0.5">
          Front-facing, full-body RGB photo for best results. Plain background recommended.
        </p>
      </div>

      {/* Drop zone */}
      {!preview && (
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
              {dragOver ? 'Drop your image here' : 'Click or drag & drop to upload'}
            </p>
            <p className="text-xs text-gray-400 mt-1">{ALLOWED_EXT_LABEL} · max {MAX_SIZE_MB} MB</p>
          </div>
          <input
            ref={inputRef}
            type="file"
            accept=".jpg,.jpeg,.png,image/jpeg,image/png"
            className="sr-only"
            onChange={handleInputChange}
            disabled={disabled}
          />
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

      {/* Preview panel */}
      {preview && (
        <div className="flex gap-4 p-4 rounded-xl bg-white border border-gray-200 shadow-sm">
          {/* Thumbnail */}
          <div className="shrink-0 w-28 h-36 rounded-lg overflow-hidden bg-gray-100 border border-gray-200">
            <img src={preview} alt="Preview" className="w-full h-full object-cover" />
          </div>

          {/* Meta + actions */}
          <div className="flex flex-col justify-between flex-1 min-w-0">
            <div>
              <p className="text-sm font-medium text-gray-800 truncate" title={fileName}>
                {fileName}
              </p>
              {fileMeta && (
                <p className="text-xs text-gray-400 mt-1">
                  {(fileMeta.size / 1024).toFixed(1)} KB
                  {fileMeta.width ? ` · ${fileMeta.width} × ${fileMeta.height} px` : ''}
                </p>
              )}
              <span className="inline-block mt-2 px-2 py-0.5 rounded-full bg-teal-50 text-teal-700 text-xs font-medium ring-1 ring-teal-200">
                Ready to process
              </span>
            </div>

            {!disabled && (
              <button
                onClick={handleClear}
                className="self-start mt-3 text-xs text-gray-400 hover:text-red-500 transition-colors"
              >
                Remove image
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
