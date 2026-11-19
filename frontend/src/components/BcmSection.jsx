/**
 * BcmSection.jsx
 * --------------
 * Optional BCM (body-composition monitor) data of one assessment: empty state, view, manual entry / edit,
 * DEMO DATA, bcm-json-v1 import, removal and revision history.
 *
 * The camera assessment never depends on this section. No device connection is offered here: Tanita device
 * integration is not implemented (see DeviceIntegrationCard).
 */
import { useEffect, useState } from 'react'
import {
  formFromRecord, payloadFromForm, profileParameters, groupParameters, countEntered, formatBcmValue, originalReading,
  sourceBadge, fieldErrors, formatWhen,
} from './bcmModel.js'
import { getAssessment, getBcmSchema, saveBcm, saveDemoBcm, importBcm, removeBcm, bcmHistory, getDeviceStatus } from '../services/assessmentApi.js'

const TONES = {
  demo: 'bg-amber-100 text-amber-800 ring-amber-300',
  manual: 'bg-indigo-50 text-indigo-700 ring-indigo-200',
  import: 'bg-sky-50 text-sky-700 ring-sky-200',
  device: 'bg-emerald-50 text-emerald-700 ring-emerald-200',
}
const field = 'w-full rounded-lg border border-slate-200 bg-white px-2 py-1.5 text-sm text-slate-800 focus:border-indigo-400 focus:outline-none focus:ring-1 focus:ring-indigo-300'
const btn = 'rounded-lg px-3 py-1.5 text-sm font-medium transition disabled:opacity-50'

export function SourceTag({ kind }) {
  return kind === 'camera'
    ? <span className="rounded-md bg-teal-50 px-2 py-0.5 text-[11px] font-semibold tracking-wider text-teal-700 ring-1 ring-teal-200">CAMERA</span>
    : <span className="rounded-md bg-indigo-50 px-2 py-0.5 text-[11px] font-semibold tracking-wider text-indigo-700 ring-1 ring-indigo-200">BCM</span>
}

export default function BcmSection({ assessmentId, record, onChange }) {
  const [schema, setSchema] = useState(null)
  const [schemaError, setSchemaError] = useState(null)
  const [editing, setEditing] = useState(false)
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState(null)
  const [demoSet, setDemoSet] = useState(1)
  const [showHistory, setShowHistory] = useState(false)

  useEffect(() => {
    getBcmSchema().then(setSchema).catch((e) => setSchemaError(e.message))
  }, [])
  useEffect(() => { setEditing(false); setMsg(null); setShowHistory(false) }, [assessmentId])

  const run = async (fn, okMsg) => {
    setBusy(true); setMsg(null)
    try {
      const rec = await fn()
      onChange?.(rec)
      if (okMsg) setMsg({ ok: true, text: okMsg })
      return rec
    } catch (e) {
      setMsg({ ok: false, text: e.message })
      return undefined
    } finally { setBusy(false) }
  }

  const useDemo = () => run(() => saveDemoBcm(assessmentId, demoSet, 'generic_segmental'), `Sample set ${demoSet} attached as DEMO DATA.`)
  const remove = () => {
    if (!window.confirm('Remove the BCM data from this assessment? The camera assessment is not affected; the removed values stay in the history log.')) return
    run(async () => { await removeBcm(assessmentId); return null }, 'BCM data removed.')
  }
  const onImport = async (e) => {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (!file) return
    if (file.size > 200_000) { setMsg({ ok: false, text: 'File is too large for a BCM record (max 200 kB).' }); return }
    let doc
    try { doc = JSON.parse(await file.text()) } catch { setMsg({ ok: false, text: 'The file is not valid JSON (expected bcm-json-v1).' }); return }
    run(() => importBcm(assessmentId, doc, file.name), `Imported ${file.name}.`)
  }

  if (schemaError) return <Card><p className="text-sm text-red-600">BCM schema unavailable: {schemaError}</p></Card>
  if (!schema) return <Card><p className="text-sm text-slate-400">Loading BCM section…</p></Card>

  if (editing) {
    return (
      <Card>
        <BcmForm schema={schema} record={record} busy={busy}
          onCancel={() => setEditing(false)}
          onSave={async (payload, setErrors) => {
            setBusy(true); setMsg(null)
            try {
              const rec = await saveBcm(assessmentId, payload)
              onChange?.(rec); setEditing(false)
              setMsg({ ok: true, text: record ? `Saved revision ${rec.revision}.` : 'BCM data saved.' })
            } catch (e) {
              setErrors(fieldErrors(e.detail))
              setMsg({ ok: false, text: e.message })
            } finally { setBusy(false) }
          }} />
        <Status msg={msg} />
      </Card>
    )
  }

  if (!record) {
    return (
      <Card>
        <Header />
        <div className="mt-4 rounded-xl border border-dashed border-slate-300 bg-slate-50/60 px-4 py-6 text-center" data-testid="bcm-empty">
          <div className="mx-auto flex h-10 w-10 items-center justify-center rounded-full bg-white text-slate-400 ring-1 ring-slate-200">
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
              <rect x="4" y="3" width="16" height="18" rx="3" /><path d="M8 8h8M8 12h5" strokeLinecap="round" />
            </svg>
          </div>
          <p className="mt-3 text-sm font-medium text-slate-700">No BCM data for this assessment</p>
          <p className="mx-auto mt-1 max-w-sm text-xs text-slate-500">
            The camera measurements are complete without it. Add body-composition values when a BCM reading for the
            same session is available.
          </p>
          <div className="mt-4 flex flex-wrap justify-center gap-2">
            <button type="button" onClick={() => setEditing(true)} className={`${btn} bg-indigo-600 text-white hover:bg-indigo-700`}>
              Add BCM data
            </button>
            <label className={`${btn} cursor-pointer bg-white text-slate-700 ring-1 ring-slate-200 hover:bg-slate-50`}>
              Import file
              <input type="file" accept="application/json,.json" className="hidden" onChange={onImport} disabled={busy} />
            </label>
          </div>
          <div className="mt-3 flex flex-wrap items-center justify-center gap-2 text-xs text-slate-500">
            <span>or show how it will look:</span>
            <select value={demoSet} onChange={(e) => setDemoSet(Number(e.target.value))} className="rounded-md border border-slate-200 bg-white px-1.5 py-0.5"
              aria-label="Sample set">
              {[1, 2, 3].map((n) => <option key={n} value={n}>Sample set {n}</option>)}
            </select>
            <button type="button" onClick={useDemo} disabled={busy}
              className="rounded-md bg-amber-50 px-2 py-0.5 font-semibold text-amber-800 ring-1 ring-amber-300 hover:bg-amber-100 disabled:opacity-50">
              Use DEMO DATA
            </button>
          </div>
        </div>
        <Status msg={msg} />
      </Card>
    )
  }

  const badge = sourceBadge(record)
  const params = Object.fromEntries(schema.parameters.map((p) => [p.key, p]))
  const whole = schema.parameters.filter((p) => p.group === 'whole_body' && record.values[p.key])
  const segKeys = Object.keys(record.values).filter((k) => k.startsWith('segment_'))
  return (
    <Card>
      <Header badge={badge} />
      {record.source.is_demo && (
        <div className="mt-3 rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-900" role="note" data-testid="bcm-demo-banner">
          <strong className="font-semibold tracking-wide">DEMO DATA.</strong> Invented sample values that show how BCM data will
          appear. Not a Tanita measurement and not a measurement of this person.
        </div>
      )}
      <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 text-xs text-slate-500 sm:grid-cols-4">
        <Meta k="Measured" v={formatWhen(record.measured_at)} />
        <Meta k="Profile" v={record.profile?.label} />
        <Meta k="Device model" v={record.source.device_model || 'Not recorded'} />
        <Meta k="Revision" v={`${record.revision} · ${formatWhen(record.updated_at)}`} />
      </dl>

      {whole.length > 0 && (
        <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-3">
          {whole.map((p) => {
            const e = record.values[p.key]
            const orig = originalReading(e)
            return (
              <div key={p.key} className="rounded-xl bg-indigo-50/40 px-3 py-2 ring-1 ring-indigo-100">
                <div className="text-[11px] uppercase tracking-wide text-slate-500">{p.label}</div>
                <div className="text-lg font-semibold text-slate-900">
                  {formatBcmValue(e, p)} <span className="text-xs font-normal text-slate-500">{p.unit === 'rating' ? '' : p.unit}</span>
                </div>
                {orig && <div className="text-[11px] text-slate-400">entered as {orig}</div>}
              </div>
            )
          })}
        </div>
      )}

      {segKeys.length > 0 && <SegmentalTable record={record} schema={schema} />}

      {record.warnings?.length > 0 && (
        <ul className="mt-3 space-y-1 rounded-lg bg-orange-50 px-3 py-2 text-xs text-orange-800">
          {record.warnings.map((w) => <li key={w.code}>⚠ {w.message} Values are shown as entered.</li>)}
        </ul>
      )}
      {record.missing?.length > 0 && (
        <details className="mt-3 text-xs text-slate-500">
          <summary className="cursor-pointer">{record.missing.length} parameter(s) of this profile not provided</summary>
          <p className="mt-1">{record.missing.map((k) => params[k]?.label || k).join(', ')}. Missing values are left blank, never estimated.</p>
        </details>
      )}
      {record.note && !record.source.is_demo && <p className="mt-2 text-xs text-slate-500">Note: {record.note}</p>}

      <div className="mt-4 flex flex-wrap gap-2">
        <button type="button" onClick={() => setEditing(true)} className={`${btn} bg-indigo-600 text-white hover:bg-indigo-700`}>
          {record.source.is_demo ? 'Replace with real values' : 'Edit BCM data'}
        </button>
        <button type="button" onClick={() => setShowHistory((s) => !s)} className={`${btn} bg-white text-slate-700 ring-1 ring-slate-200 hover:bg-slate-50`}>
          {showHistory ? 'Hide history' : 'Revision history'}
        </button>
        <button type="button" onClick={remove} disabled={busy} className={`${btn} text-red-600 hover:bg-red-50`}>Remove</button>
      </div>
      {showHistory && <RevisionHistory assessmentId={assessmentId} key={record.updated_at} />}
      <Status msg={msg} />
    </Card>
  )
}

function SegmentalTable({ record, schema }) {
  const v = record.values
  const cols = [['segment_fat_pct', 'Fat', '%'], ['segment_fat_mass', 'Fat mass', 'kg'], ['segment_muscle_mass', 'Muscle', 'kg']]
    .filter(([pre]) => schema.segments.some((s) => v[`${pre}_${s.key}`]))
  return (
    <div className="mt-4 overflow-x-auto">
      <table className="w-full text-sm">
        <thead><tr className="text-left text-[11px] uppercase tracking-wide text-slate-500">
          <th className="py-1 font-medium">Segment</th>
          {cols.map(([pre, l, u]) => <th key={pre} className="py-1 text-right font-medium">{l} ({u})</th>)}
        </tr></thead>
        <tbody>
          {schema.segments.map((s) => (
            <tr key={s.key} className="border-t border-slate-100">
              <td className="py-1 text-slate-700">{s.label}</td>
              {cols.map(([pre]) => <td key={pre} className="py-1 text-right tabular-nums text-slate-900">{formatBcmValue(v[`${pre}_${s.key}`], { decimals: 1 })}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function RevisionHistory({ assessmentId }) {
  const [events, setEvents] = useState(null)
  useEffect(() => { bcmHistory(assessmentId).then(setEvents).catch(() => setEvents([])) }, [assessmentId])
  if (!events) return <p className="mt-2 text-xs text-slate-400">Loading history…</p>
  return (
    <ol className="mt-3 space-y-1 border-l-2 border-slate-100 pl-3 text-xs text-slate-600">
      {[...events].reverse().map((e, i) => (
        <li key={i}>
          <span className="font-medium">{e.event === 'removed' ? 'Removed' : `Revision ${e.record.revision}`}</span>
          {' · '}{formatWhen(e.at)} · {sourceBadge(e.record)?.text} · {Object.keys(e.record.values || {}).length} value(s)
          {e.record.values?.weight && <> · weight {e.record.values.weight.value} kg</>}
        </li>
      ))}
    </ol>
  )
}

function BcmForm({ schema, record, busy, onSave, onCancel }) {
  // Sample values are never carried into a real record: editing DEMO DATA starts from an empty form.
  const [form, setForm] = useState(() => formFromRecord(record?.source?.is_demo ? null : record, schema))
  const [errors, setErrors] = useState({})
  const params = profileParameters(schema, form.profile)
  const groups = groupParameters(params.filter((p) => p.group === 'whole_body'))
  const segmental = params.some((p) => p.segment)
  const profile = schema.profiles.find((p) => p.key === form.profile)
  const setValue = (key, patch) => setForm((f) => ({ ...f, values: { ...f.values, [key]: { ...f.values[key], ...patch } } }))
  const set = (k) => (e) => setForm((f) => ({ ...f, [k]: e.target.value }))
  const entered = countEntered(form, schema)

  return (
    <form onSubmit={(e) => { e.preventDefault(); setErrors({}); onSave(payloadFromForm(form, schema), setErrors) }} data-testid="bcm-form">
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-base font-semibold text-slate-900">{record ? 'Edit BCM data' : 'Add BCM data'}</h3>
        <SourceTag kind="bcm" />
      </div>
      {record?.source?.is_demo && (
        <p className="mt-2 rounded-md bg-amber-50 px-2 py-1 text-xs text-amber-800">
          The sample values are not pre-filled. Saving replaces the DEMO DATA with your entries.
        </p>
      )}
      <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-3">
        <label className="text-xs text-slate-600">Device / assessment profile
          <select value={form.profile} onChange={set('profile')} className={`${field} mt-1`}>
            {schema.profiles.map((p) => <option key={p.key} value={p.key}>{p.label}</option>)}
          </select>
        </label>
        <label className="text-xs text-slate-600">Measured at
          <input type="datetime-local" value={form.measured_at} onChange={set('measured_at')} className={`${field} mt-1`} required />
          {errors.measured_at && <span className="text-red-600">{errors.measured_at}</span>}
        </label>
        <label className="text-xs text-slate-600">Device model (optional)
          <input value={form.device_model} onChange={set('device_model')} maxLength={60} placeholder="As printed on the device" className={`${field} mt-1`} />
        </label>
      </div>
      {profile && <p className="mt-2 text-xs text-slate-500">{profile.description} {profile.notes?.[0]}</p>}

      {groups.map((g) => (
        <fieldset key={g.key} className="mt-4">
          <legend className="text-xs font-semibold uppercase tracking-wide text-slate-500">{g.label}</legend>
          <div className="mt-2 grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
            {g.params.map((p) => (
              <NumberField key={p.key} p={p} v={form.values[p.key]} error={errors[p.key]} onChange={(patch) => setValue(p.key, patch)} />
            ))}
          </div>
        </fieldset>
      ))}

      {segmental && (
        <fieldset className="mt-4">
          <legend className="text-xs font-semibold uppercase tracking-wide text-slate-500">Segmental composition</legend>
          <div className="mt-2 overflow-x-auto">
            <table className="w-full min-w-[420px] text-sm">
              <thead><tr className="text-left text-[11px] text-slate-500">
                <th className="py-1 font-medium">Segment</th><th className="py-1 font-medium">Fat (%)</th>
                <th className="py-1 font-medium">Fat mass</th><th className="py-1 font-medium">Muscle mass</th>
              </tr></thead>
              <tbody>
                {schema.segments.map((s) => (
                  <tr key={s.key}>
                    <td className="py-1 pr-2 text-slate-700">{s.label}</td>
                    {['segment_fat_pct', 'segment_fat_mass', 'segment_muscle_mass'].map((pre) => {
                      const p = params.find((x) => x.key === `${pre}_${s.key}`)
                      return <td key={pre} className="py-1 pr-2">{p && <NumberField p={p} compact v={form.values[p.key]} error={errors[p.key]} onChange={(patch) => setValue(p.key, patch)} />}</td>
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </fieldset>
      )}

      <label className="mt-4 block text-xs text-slate-600">Note (optional)
        <input value={form.note} onChange={set('note')} maxLength={500} className={`${field} mt-1`} placeholder="e.g. measured after fasting" />
      </label>
      {errors.values && <p className="mt-2 text-xs text-red-600">{errors.values}</p>}
      {errors.profile && <p className="mt-2 text-xs text-red-600">{errors.profile}</p>}

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <button type="submit" disabled={busy || entered === 0} className={`${btn} bg-indigo-600 text-white hover:bg-indigo-700`}>
          {busy ? 'Saving…' : 'Save'}
        </button>
        <button type="button" onClick={onCancel} className={`${btn} text-slate-600 hover:bg-slate-100`}>Cancel</button>
        <span className="text-xs text-slate-500">{entered} value(s) entered · leave fields blank that were not measured</span>
      </div>
    </form>
  )
}

function NumberField({ p, v, error, onChange, compact = false }) {
  const input = (
    <div className="flex gap-1">
      <input inputMode="decimal" value={v?.value ?? ''} onChange={(e) => onChange({ value: e.target.value })}
        aria-label={p.label} aria-invalid={!!error} placeholder={compact ? '' : `${p.min}–${p.max}`}
        className={`${field} ${error ? 'border-red-400' : ''} tabular-nums`} />
      {p.units.length > 1 ? (
        <select value={v?.unit || p.unit} onChange={(e) => onChange({ unit: e.target.value })} aria-label={`${p.label} unit`}
          className="rounded-lg border border-slate-200 bg-white px-1 text-xs text-slate-700">
          {p.units.map((u) => <option key={u} value={u}>{u}</option>)}
        </select>
      ) : (
        p.unit !== 'rating' && !compact && <span className="self-center px-1 text-xs text-slate-500">{p.unit}</span>
      )}
    </div>
  )
  if (compact) return <div>{input}{error && <div className="text-[11px] text-red-600">{error}</div>}</div>
  return (
    <label className="text-xs text-slate-600" title={p.help || undefined}>
      {p.label}{p.help && <span className="text-slate-400"> ⓘ</span>}
      <div className="mt-1">{input}</div>
      {error && <span className="text-[11px] text-red-600">{error}</span>}
    </label>
  )
}

/** Planned Tanita device integration: shows the real status from the backend (no adapter = not connected). */
export function DeviceIntegrationCard() {
  const [status, setStatus] = useState(null)
  const [err, setErr] = useState(null)
  useEffect(() => { getDeviceStatus().then(setStatus).catch((e) => setErr(e.message)) }, [])
  const connected = status?.state === 'connected'
  return (
    <Card>
      <div className="flex items-center justify-between gap-2">
        <h3 className="text-sm font-semibold text-slate-900">Tanita device integration</h3>
        <span className={`rounded-full px-2.5 py-0.5 text-[11px] font-semibold ring-1 ${connected
          ? 'bg-emerald-50 text-emerald-700 ring-emerald-200' : 'bg-slate-100 text-slate-600 ring-slate-200'}`}>
          {status ? (connected ? 'Connected' : 'Planned · not connected') : '…'}
        </span>
      </div>
      {err && <p className="mt-2 text-xs text-red-600">{err}</p>}
      {status && (
        <>
          <p className="mt-2 text-xs text-slate-500">{status.message}</p>
          <ul className="mt-2 space-y-0.5 text-xs text-slate-500">
            <li>Available now: manual entry, bcm-json-v1 import, DEMO DATA.</li>
            <li>Adapter interface: <code className="rounded bg-slate-100 px-1">{status.adapter_contract.interface.split('.').pop()}</code> — {status.adapter_contract.methods.join(', ')}</li>
          </ul>
          <button type="button" disabled className="mt-3 cursor-not-allowed rounded-lg bg-slate-100 px-3 py-1.5 text-xs font-medium text-slate-400"
            title="No device adapter is installed">
            Read from device (not available)
          </button>
        </>
      )}
    </Card>
  )
}

function Header({ badge }) {
  return (
    <div className="flex items-center justify-between gap-2">
      <div className="flex items-center gap-2">
        <h3 className="text-base font-semibold text-slate-900">Body composition</h3>
        <SourceTag kind="bcm" />
      </div>
      {badge && <span className={`rounded-full px-2.5 py-0.5 text-[11px] font-semibold tracking-wide ring-1 ${TONES[badge.tone]}`}>{badge.text}</span>}
    </div>
  )
}

function Meta({ k, v }) {
  return <div><dt className="text-[10px] uppercase tracking-wide text-slate-400">{k}</dt><dd className="break-words text-slate-700">{v}</dd></div>
}

function Status({ msg }) {
  if (!msg) return null
  return <p className={`mt-2 text-xs ${msg.ok ? 'text-emerald-700' : 'text-red-600'}`} role="status">{msg.text}</p>
}

/** BCM section for a just-finished reconstruction (360° tab): loads the assessment's current record itself. */
export function RunBcmSection({ runId }) {
  const [state, setState] = useState({ loading: true, record: null, error: null })
  useEffect(() => {
    let alive = true
    getAssessment(runId)
      .then((a) => alive && setState({ loading: false, record: a.bcm, error: null }))
      .catch((e) => alive && setState({ loading: false, record: null, error: e.message }))
    return () => { alive = false }
  }, [runId])
  if (state.loading) return null
  if (state.error) return <Card><p className="text-xs text-slate-500">BCM section unavailable: {state.error}</p></Card>
  return <BcmSection assessmentId={runId} record={state.record} onChange={(record) => setState((s) => ({ ...s, record }))} />
}

export function Card({ children, className = '' }) {
  return <section className={`rounded-2xl border border-slate-200 bg-white p-4 shadow-sm sm:p-5 ${className}`}>{children}</section>
}
