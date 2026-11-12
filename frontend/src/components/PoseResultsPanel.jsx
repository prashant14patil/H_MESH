/**
 * PoseResultsPanel.jsx
 * --------------------
 * Module 3 result display — shows per-frame pose/segmentation results:
 * pose visualization overlay, segmentation mask preview, detection
 * confidence, full-body visibility, and review-required flags.
 *
 * Props:
 *   videoId: string   – id of the processed video
 *   summary: object   – response from GET /api/pose/results/{video_id}
 */

import { useState } from 'react'
import { getPoseFrameAssetUrl } from '../services/poseApi.js'

export default function PoseResultsPanel({ videoId, summary }) {
  const [selectedFrame, setSelectedFrame] = useState(null)

  if (!summary) return null

  const {
    total_frames,
    frames_with_person_detected,
    frames_full_body_visible,
    frames_requiring_review,
    frames = [],
  } = summary

  return (
    <div className="space-y-4">
      {/* Summary stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <Stat label="Total frames" value={total_frames} />
        <Stat label="Person detected" value={frames_with_person_detected} tone="good" />
        <Stat label="Full body visible" value={frames_full_body_visible} tone="good" />
        <Stat label="Needs review" value={frames_requiring_review} tone={frames_requiring_review > 0 ? 'warn' : 'good'} />
      </div>

      {/* Grid of frame thumbnails (pose visualization) */}
      <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 gap-2">
        {frames.map((frame) => (
          <FrameThumb
            key={frame.frame}
            videoId={videoId}
            frame={frame}
            onClick={() => setSelectedFrame(frame)}
          />
        ))}
      </div>

      {/* Detail panel for selected frame */}
      {selectedFrame && (
        <FrameDetail
          videoId={videoId}
          frame={selectedFrame}
          onClose={() => setSelectedFrame(null)}
        />
      )}
    </div>
  )
}

function Stat({ label, value, tone = 'neutral' }) {
  const toneClass = {
    good: 'text-emerald-600',
    warn: 'text-amber-600',
    neutral: 'text-gray-800',
  }[tone]

  return (
    <div className="p-3 rounded-lg bg-gray-50 border border-gray-200 text-center">
      <p className={`text-sm font-semibold ${toneClass}`}>{value}</p>
      <p className="text-[10px] uppercase tracking-wide text-gray-400 mt-0.5">{label}</p>
    </div>
  )
}

function FrameThumb({ videoId, frame, onClick }) {
  const needsReview = frame.status === 'review_required'
  const isOk = frame.status === 'ok'

  return (
    <button
      onClick={onClick}
      className="relative rounded-lg overflow-hidden border border-gray-200 bg-gray-100 aspect-square group text-left"
    >
      <img
        src={getPoseFrameAssetUrl(videoId, frame.frame, 'pose_visualization.jpg')}
        alt={frame.frame}
        className="w-full h-full object-cover"
        loading="lazy"
      />
      <span
        title={frame.status}
        className={`absolute top-1 right-1 flex items-center justify-center w-5 h-5 rounded-full text-[10px] font-bold text-white
          ${isOk ? 'bg-emerald-500' : needsReview ? 'bg-red-500' : 'bg-amber-500'}`}
      >
        {isOk ? '✓' : needsReview ? '!' : '~'}
      </span>
      <div className="absolute inset-x-0 bottom-0 bg-black/60 text-white text-[10px] px-1.5 py-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
        {frame.frame}
      </div>
    </button>
  )
}

function FrameDetail({ videoId, frame, onClose }) {
  const landmarks = frame.landmarks || {}
  const landmarkEntries = Object.entries(landmarks)

  return (
    <div className="p-4 rounded-xl bg-white border border-gray-200 shadow-sm space-y-4">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-semibold text-gray-800">{frame.frame} — details</h4>
        <button
          onClick={onClose}
          className="text-xs text-gray-400 hover:text-gray-700"
        >
          Close ✕
        </button>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {/* Pose visualization */}
        <div>
          <p className="text-xs font-medium text-gray-500 mb-1.5">Pose visualization</p>
          <img
            src={getPoseFrameAssetUrl(videoId, frame.frame, 'pose_visualization.jpg')}
            alt="pose visualization"
            className="w-full rounded-lg border border-gray-200"
          />
        </div>

        {/* Segmentation mask */}
        <div>
          <p className="text-xs font-medium text-gray-500 mb-1.5">Segmentation mask</p>
          <img
            src={getPoseFrameAssetUrl(videoId, frame.frame, 'segmentation_mask.png')}
            alt="segmentation mask"
            className="w-full rounded-lg border border-gray-200 bg-gray-900"
          />
        </div>
      </div>

      {/* Metadata summary */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
        <InfoChip label="Detected" value={frame.person_detected ? 'Yes' : 'No'} />
        <InfoChip
          label="Confidence"
          value={typeof frame.detection_confidence === 'number' ? frame.detection_confidence.toFixed(2) : '—'}
        />
        <InfoChip label="Full body" value={frame.full_body_visible ? 'Yes' : 'No'} />
        <InfoChip label="View hint" value={frame.view_hint || 'unknown'} />
      </div>

      {frame.missing_regions && frame.missing_regions.length > 0 && (
        <p className="text-xs text-amber-700">
          Missing / low-visibility regions: {frame.missing_regions.join(', ')}
        </p>
      )}

      {/* Landmark table */}
      {landmarkEntries.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-gray-400 border-b border-gray-100">
                <th className="py-1 pr-2">Landmark</th>
                <th className="py-1 pr-2">Visibility</th>
                <th className="py-1">Status</th>
              </tr>
            </thead>
            <tbody>
              {landmarkEntries.map(([name, data]) => (
                <tr key={name} className="border-b border-gray-50">
                  <td className="py-1 pr-2 text-gray-700 capitalize">{name.replace(/_/g, ' ')}</td>
                  <td className="py-1 pr-2 text-gray-500">{data.visibility?.toFixed(2)}</td>
                  <td className="py-1">
                    <span className={data.visible ? 'text-emerald-600' : 'text-amber-600'}>
                      {data.visible ? 'visible' : 'low visibility'}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function InfoChip({ label, value }) {
  return (
    <div className="p-2 rounded-lg bg-gray-50 border border-gray-200 text-center">
      <p className="font-semibold text-gray-700">{value}</p>
      <p className="text-[10px] uppercase tracking-wide text-gray-400">{label}</p>
    </div>
  )
}
