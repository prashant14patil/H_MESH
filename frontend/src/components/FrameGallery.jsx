/**
 * FrameGallery.jsx
 * ----------------
 * Module 2 result display — renders the frames extracted from an
 * uploaded 360° video, along with per-frame quality indicators
 * (sharp / blurry) derived from the backend's Laplacian-variance
 * blur score, plus a summary of the extraction run.
 *
 * Props:
 *   videoId: string         – id of the processed video
 *   metadata: object        – response from GET /api/capture/frames/{video_id}
 */

import { getFrameImageUrl } from '../services/captureApi.js'

export default function FrameGallery({ videoId, metadata }) {
  if (!metadata) return null

  const { video_info: videoInfo, frames = [], accepted_frame_count, rejected_frame_count } = metadata

  return (
    <div className="space-y-4">
      {/* Summary */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <SummaryStat label="Duration" value={`${videoInfo?.duration_sec?.toFixed(1) ?? '—'}s`} />
        <SummaryStat label="Resolution" value={videoInfo ? `${videoInfo.width}×${videoInfo.height}` : '—'} />
        <SummaryStat label="FPS" value={videoInfo?.fps?.toFixed(1) ?? '—'} />
        <SummaryStat label="Frames extracted" value={accepted_frame_count ?? frames.length} />
      </div>

      {rejected_frame_count > 0 && (
        <p className="text-xs text-gray-400">
          {rejected_frame_count} candidate frame(s) were skipped as near-duplicates or unreadable.
        </p>
      )}

      {/* Gallery grid */}
      <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 gap-2">
        {frames.map((frame) => (
          <FrameThumb key={frame.filename} videoId={videoId} frame={frame} />
        ))}
      </div>
    </div>
  )
}

function SummaryStat({ label, value }) {
  return (
    <div className="p-3 rounded-lg bg-gray-50 border border-gray-200 text-center">
      <p className="text-sm font-semibold text-gray-800">{value}</p>
      <p className="text-[10px] uppercase tracking-wide text-gray-400 mt-0.5">{label}</p>
    </div>
  )
}

function FrameThumb({ videoId, frame }) {
  const isBlurry = !!frame.is_blurry

  return (
    <div className="relative rounded-lg overflow-hidden border border-gray-200 bg-gray-100 aspect-square group">
      <img
        src={getFrameImageUrl(videoId, frame.filename)}
        alt={frame.filename}
        className="w-full h-full object-cover"
        loading="lazy"
      />

      {/* Quality badge */}
      <span
        title={`Blur score: ${frame.blur_score}`}
        className={`absolute top-1 right-1 flex items-center justify-center w-5 h-5 rounded-full text-[10px] font-bold
          ${isBlurry ? 'bg-amber-500 text-white' : 'bg-emerald-500 text-white'}`}
      >
        {isBlurry ? '!' : '✓'}
      </span>

      {/* Timestamp overlay, shown on hover */}
      <div className="absolute inset-x-0 bottom-0 bg-black/60 text-white text-[10px] px-1.5 py-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
        {frame.timestamp_sec?.toFixed(2)}s
      </div>
    </div>
  )
}
