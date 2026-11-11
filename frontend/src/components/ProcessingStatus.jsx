/**
 * ProcessingStatus.jsx
 * --------------------
 * Displays the current state of the inference pipeline with animated
 * progress indicators.
 *
 * Props:
 *   phase: 'idle' | 'uploading' | 'queued' | 'processing' | 'completed' | 'failed'
 *   jobId: string | null
 *   attempt: number        – current poll attempt
 *   maxAttempts: number    – max poll attempts
 *   error: string | null   – error message when phase === 'failed'
 */

const PHASE_CONFIG = {
  idle: {
    label: 'Waiting for image',
    color: 'gray',
    icon: 'clock',
  },
  uploading: {
    label: 'Uploading image…',
    color: 'teal',
    icon: 'spinner',
  },
  queued: {
    label: 'Job queued — waiting for GPU…',
    color: 'teal',
    icon: 'spinner',
  },
  processing: {
    label: 'Running HMR2 reconstruction…',
    color: 'teal',
    icon: 'spinner',
  },
  completed: {
    label: 'Reconstruction complete',
    color: 'green',
    icon: 'check',
  },
  failed: {
    label: 'Reconstruction failed',
    color: 'red',
    icon: 'error',
  },
}

function Spinner({ className = '' }) {
  return (
    <svg className={`animate-spin ${className}`} fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path
        className="opacity-75"
        fill="currentColor"
        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
      />
    </svg>
  )
}

function StatusIcon({ icon, color }) {
  const cls = `w-5 h-5`
  if (icon === 'spinner') {
    return <Spinner className={`${cls} text-${color}-600`} />
  }
  if (icon === 'check') {
    return (
      <svg className={`${cls} text-${color}-600`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
      </svg>
    )
  }
  if (icon === 'error') {
    return (
      <svg className={`${cls} text-${color}-600`} fill="currentColor" viewBox="0 0 20 20">
        <path fillRule="evenodd"
          d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z"
          clipRule="evenodd" />
      </svg>
    )
  }
  // clock
  return (
    <svg className={`${cls} text-${color}-400`} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
      <path strokeLinecap="round" strokeLinejoin="round" d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z" />
    </svg>
  )
}

export default function ProcessingStatus({
  phase = 'idle',
  jobId = null,
  attempt = 0,
  maxAttempts = 40,
  error = null,
}) {
  const config = PHASE_CONFIG[phase] || PHASE_CONFIG.idle
  const isActive = ['uploading', 'queued', 'processing'].includes(phase)
  const progress = isActive ? Math.min((attempt / maxAttempts) * 100, 95) : phase === 'completed' ? 100 : 0

  const bgClass = {
    gray: 'bg-gray-50 border-gray-200',
    teal: 'bg-teal-50 border-teal-200',
    green: 'bg-green-50 border-green-200',
    red: 'bg-red-50 border-red-200',
  }[config.color]

  return (
    <div className={`rounded-xl border p-4 space-y-3 ${bgClass}`}>
      {/* Status row */}
      <div className="flex items-center gap-3">
        <StatusIcon icon={config.icon} color={config.color} />
        <div className="flex-1 min-w-0">
          <p className={`text-sm font-medium text-${config.color}-800`}>
            {config.label}
          </p>
          {jobId && (
            <p className="text-xs text-gray-400 truncate mt-0.5">
              Job ID: <span className="font-mono">{jobId}</span>
            </p>
          )}
        </div>
        {isActive && attempt > 0 && (
          <span className="text-xs text-gray-400 shrink-0">
            {attempt}/{maxAttempts}
          </span>
        )}
      </div>

      {/* Progress bar – visible during active phases */}
      {(isActive || phase === 'completed') && (
        <div className="w-full bg-white/60 rounded-full h-1.5 overflow-hidden">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              phase === 'completed' ? 'bg-green-500' : 'bg-teal-500'
            }`}
            style={{ width: `${progress}%` }}
          />
        </div>
      )}

      {/* Error detail */}
      {phase === 'failed' && error && (
        <p className="text-xs text-red-600 bg-red-100 rounded-lg px-3 py-2">
          {error}
        </p>
      )}

      {/* Step pipeline */}
      <div className="flex items-center gap-1 pt-1">
        {['uploading', 'queued', 'processing', 'completed'].map((step, i) => {
          const stepOrder = ['uploading', 'queued', 'processing', 'completed']
          const currentIdx = stepOrder.indexOf(phase)
          const stepIdx = stepOrder.indexOf(step)
          const done = currentIdx > stepIdx || phase === 'completed'
          const active = currentIdx === stepIdx
          return (
            <div key={step} className="flex items-center gap-1 flex-1">
              <div
                className={`flex-1 h-0.5 rounded-full transition-colors ${
                  done || active ? 'bg-teal-400' : 'bg-gray-200'
                }`}
              />
              <div
                className={`w-2 h-2 rounded-full shrink-0 transition-colors ${
                  done ? 'bg-teal-500' : active ? 'bg-teal-400 animate-pulse' : 'bg-gray-200'
                }`}
                title={step}
              />
            </div>
          )
        })}
      </div>
    </div>
  )
}
