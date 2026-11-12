/**
 * FrameGallery.jsx
 * ----------------
 * Module 2/3 result display — renders the frames extracted from an
 * uploaded 360° video, per-frame quality indicators (blur/duplicate),
 * an extraction summary (candidates/accepted/rejected + reasons,
 * duration/FPS/resolution), and -- once available -- the viewpoint
 * (orientation) classification + 360° coverage report derived from
 * the existing pose-processing pipeline.
 *
 * Props:
 *   videoId: string          – id of the processed video
 *   metadata: object         – response from GET /api/capture/frames/{video_id}
 *   viewpointData: object|null – response from GET /api/pose/viewpoints/{video_id}
 *   posePhase: string        – CaptureWorkspace phase, used to show a
 *                                "analyzing…" state while pose/viewpoint
 *                                data is still being computed
 */

import { useState } from 'react'
import { getFrameImageUrl } from '../services/captureApi.js'
import { getPoseFrameAssetUrl } from '../services/poseApi.js'

const VIEWPOINT_LABELS = {
  front: 'Front',
  back: 'Back',
  left_side: 'Left side',
  right_side: 'Right side',
  intermediate: 'Intermediate',
  front_or_back: 'Front or back (ambiguous)',
  unknown: 'Unknown',
}

const VIEWPOINT_COLORS = {
  front: 'bg-emerald-500',
  back: 'bg-indigo-500',
  left_side: 'bg-sky-500',
  right_side: 'bg-purple-500',
  intermediate: 'bg-teal-500',
  front_or_back: 'bg-amber-500',
  unknown: 'bg-gray-400',
}

export default function FrameGallery({ videoId, metadata, viewpointData, poseResults, posePhase }) {
  const [selectedFrame, setSelectedFrame] = useState(null)

  if (!metadata) return null

  const {
    video_info: videoInfo,
    frames = [],
    rejected_frames: rejectedFrames = [],
    accepted_frame_count,
    rejected_frame_count,
    candidate_frame_count,
  } = metadata

  // Build a lookup from frame filename -> viewpoint classification.
  const viewpointByFrame = {}
  if (viewpointData?.frames) {
    for (const vf of viewpointData.frames) {
      // pose frame name is like "frame_001" (no extension); capture
      // filenames are "frame_001.jpg" -- normalize for lookup.
      viewpointByFrame[`${vf.frame}.jpg`] = vf
    }
  }

  // Build a lookup from frame filename -> full pose/landmark result
  // (reused from the existing Module 3 pose_processor.py output, no
  // duplicate MediaPipe inference).
  const poseByFrame = {}
  if (poseResults?.frames) {
    for (const pf of poseResults.frames) {
      poseByFrame[`${pf.frame}.jpg`] = pf
    }
  }

  const coverage = viewpointData?.coverage
  const isAnalyzingViewpoints = posePhase === 'extracting' || posePhase === 'pose_processing'

  return (
    <div className="space-y-5">
      {/* Extraction summary */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <SummaryStat label="Duration" value={`${videoInfo?.duration_sec?.toFixed(1) ?? '—'}s`} />
        <SummaryStat label="Resolution" value={videoInfo ? `${videoInfo.width}×${videoInfo.height}` : '—'} />
        <SummaryStat label="FPS" value={videoInfo?.fps?.toFixed(1) ?? '—'} />
        <SummaryStat label="Frames extracted" value={accepted_frame_count ?? frames.length} />
      </div>

      <div className="flex flex-wrap gap-3 text-xs text-gray-500">
        <span>Candidates sampled: <strong className="text-gray-700">{candidate_frame_count ?? '—'}</strong></span>
        <span>Rejected: <strong className="text-gray-700">{rejected_frame_count ?? 0}</strong></span>
        {videoInfo?.timing_source === 'measured_walk' && (
          <span className="text-amber-600" title={`Header reported FPS ${videoInfo.reported_fps}, frame count ${videoInfo.reported_total_frames}`}>
            ⚠ FPS/duration auto-corrected (unreliable container metadata detected)
          </span>
        )}
      </div>

      {/* Rejection reasons breakdown */}
      {rejectedFrames.length > 0 && (
        <RejectionBreakdown rejectedFrames={rejectedFrames} />
      )}

      {/* Viewpoint coverage summary */}
      {isAnalyzingViewpoints && (
        <div className="flex items-center gap-2 p-3 rounded-lg bg-blue-50 border border-blue-200 text-sm text-blue-700">
          <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
          Analyzing pose & viewpoint coverage…
        </div>
      )}

      {coverage && <CoverageSummary coverage={coverage} notes={coverage.notes} />}

      {/* Gallery grid */}
      <div className="grid grid-cols-3 sm:grid-cols-4 md:grid-cols-6 gap-2">
        {frames.map((frame) => (
          <FrameThumb
            key={frame.filename}
            videoId={videoId}
            frame={frame}
            viewpoint={viewpointByFrame[frame.filename]}
            onClick={() => setSelectedFrame({ frame, pose: poseByFrame[frame.filename], viewpoint: viewpointByFrame[frame.filename] })}
          />
        ))}
      </div>

      {/* Detail panel for selected frame -- reuses pose_visualization.jpg /
          landmark table concepts from PoseResultsPanel.jsx, sourced from
          the same already-fetched pose results (no duplicate requests). */}
      {selectedFrame && (
        <FrameDetail
          videoId={videoId}
          selection={selectedFrame}
          onClose={() => setSelectedFrame(null)}
        />
      )}
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

function RejectionBreakdown({ rejectedFrames }) {
  const counts = {}
  for (const r of rejectedFrames) {
    counts[r.reason] = (counts[r.reason] || 0) + 1
  }
  return (
    <div className="p-3 rounded-lg bg-gray-50 border border-gray-200">
      <p className="text-xs font-medium text-gray-500 mb-1.5">Rejected candidate frames (by reason)</p>
      <div className="flex flex-wrap gap-2">
        {Object.entries(counts).map(([reason, count]) => (
          <span key={reason} className="px-2 py-0.5 rounded-full bg-gray-200 text-gray-700 text-[11px]">
            {reason.replace(/_/g, ' ')}: {count}
          </span>
        ))}
      </div>
    </div>
  )
}

function CoverageSummary({ coverage, notes }) {
  const { counts = {}, missing_buckets = [], sufficient_rotation_coverage } = coverage

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap gap-2">
        {Object.entries(VIEWPOINT_LABELS).map(([key, label]) => {
          const count = counts[key] || 0
          const covered = count > 0
          return (
            <span
              key={key}
              className={`px-2.5 py-1 rounded-full text-[11px] font-medium
                ${covered ? `${VIEWPOINT_COLORS[key]} text-white` : 'bg-gray-100 text-gray-400'}`}
            >
              {label}: {count}
            </span>
          )
        })}
      </div>

      {!sufficient_rotation_coverage && (
        <div className="flex items-start gap-2 p-3 rounded-lg bg-amber-50 border border-amber-200 text-xs text-amber-800">
          <svg className="w-4 h-4 mt-0.5 shrink-0" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd"
              d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495zM10 5a.75.75 0 01.75.75v3.5a.75.75 0 01-1.5 0v-3.5A.75.75 0 0110 5zm0 9a1 1 0 100-2 1 1 0 000 2z"
              clipRule="evenodd" />
          </svg>
          <div>
            <strong className="font-semibold">Insufficient viewpoint coverage.</strong>{' '}
            This video does not appear to contain enough of a full 360° rotation
            {missing_buckets.length > 0 && (
              <> (no frames classified as: {missing_buckets.map((b) => VIEWPOINT_LABELS[b] || b).join(', ')})</>
            )}
            . Multi-view 3D reconstruction quality may be degraded — consider
            re-capturing with a slower, more complete rotation.
          </div>
        </div>
      )}

      {notes && (
        <p className="text-[11px] text-gray-400 italic">{notes}</p>
      )}
    </div>
  )
}

function FrameThumb({ videoId, frame, viewpoint, onClick }) {
  const isBlurry = !!frame.is_blurry
  const vpKey = viewpoint?.viewpoint
  const vpLabel = vpKey ? (VIEWPOINT_LABELS[vpKey] || vpKey) : null
  const vpColor = vpKey ? (VIEWPOINT_COLORS[vpKey] || 'bg-gray-400') : null

  return (
    <button
      onClick={onClick}
      className="relative rounded-lg overflow-hidden border border-gray-200 bg-gray-100 aspect-square group text-left"
    >
      <img
        src={getFrameImageUrl(videoId, frame.filename)}
        alt={frame.filename}
        className="w-full h-full object-cover"
        loading="lazy"
      />

      {/* Quality badge (blur) */}
      <span
        title={`Blur score: ${frame.blur_score}`}
        className={`absolute top-1 right-1 flex items-center justify-center w-5 h-5 rounded-full text-[10px] font-bold
          ${isBlurry ? 'bg-amber-500 text-white' : 'bg-emerald-500 text-white'}`}
      >
        {isBlurry ? '!' : '✓'}
      </span>

      {/* Viewpoint badge */}
      {vpLabel && (
        <span
          title={`Confidence: ${viewpoint.confidence}`}
          className={`absolute top-1 left-1 px-1.5 py-0.5 rounded-full text-[9px] font-semibold text-white ${vpColor}`}
        >
          {vpLabel}
        </span>
      )}

      {/* Timestamp + confidence overlay, shown on hover */}
      <div className="absolute inset-x-0 bottom-0 bg-black/60 text-white text-[10px] px-1.5 py-0.5 opacity-0 group-hover:opacity-100 transition-opacity flex justify-between">
        <span>{frame.timestamp_sec?.toFixed(2)}s</span>
        {viewpoint && <span>conf {viewpoint.confidence}</span>}
      </div>
    </button>
  )
}

function FrameDetail({ videoId, selection, onClose }) {
  const { frame, pose, viewpoint } = selection
  const landmarks = pose?.landmarks || {}
  const landmarkEntries = Object.entries(landmarks)
  const frameStem = frame.filename.replace(/\.jpg$/i, '')

  return (
    <div className="p-4 rounded-xl bg-white border border-gray-200 shadow-sm space-y-4">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-semibold text-gray-800">
          {frame.filename} — {frame.timestamp_sec?.toFixed(2)}s
        </h4>
        <button onClick={onClose} className="text-xs text-gray-400 hover:text-gray-700">
          Close ✕
        </button>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div>
          <p className="text-xs font-medium text-gray-500 mb-1.5">Extracted frame</p>
          <img
            src={getFrameImageUrl(videoId, frame.filename)}
            alt={frame.filename}
            className="w-full rounded-lg border border-gray-200"
          />
        </div>
        <div>
          <p className="text-xs font-medium text-gray-500 mb-1.5">Pose visualization (Module 3)</p>
          {pose ? (
            <img
              src={getPoseFrameAssetUrl(videoId, frameStem, 'pose_visualization.jpg')}
              alt="pose visualization"
              className="w-full rounded-lg border border-gray-200"
            />
          ) : (
            <div className="flex items-center justify-center h-full min-h-[120px] rounded-lg border border-dashed border-gray-200 text-xs text-gray-400">
              Pose data not available for this frame
            </div>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
        <InfoChip label="Blur score" value={frame.blur_score ?? '—'} />
        <InfoChip label="Person detected" value={pose?.person_detected ? 'Yes' : pose ? 'No' : '—'} />
        <InfoChip
          label="Viewpoint"
          value={viewpoint ? (VIEWPOINT_LABELS[viewpoint.viewpoint] || viewpoint.viewpoint) : '—'}
        />
        <InfoChip label="Confidence" value={viewpoint ? viewpoint.confidence : '—'} />
      </div>

      {viewpoint?.reason && (
        <p className="text-xs text-gray-500 italic">Classification note: {viewpoint.reason.replace(/_/g, ' ')}</p>
      )}

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
      <p className="font-semibold text-gray-700">{String(value)}</p>
      <p className="text-[10px] uppercase tracking-wide text-gray-400">{label}</p>
    </div>
  )
}
