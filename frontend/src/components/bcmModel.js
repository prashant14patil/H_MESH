/**
 * bcmModel.js
 * -----------
 * Pure helpers for the BCM (body-composition monitor) section: form <-> API payload, grouping, formatting and the
 * segment-by-segment pairing of camera girths with BCM composition. No React, no fetch — unit-tested with node:test.
 *
 * Camera values and BCM values are never merged into one number: they are different measurements of different
 * things (external shape vs. composition estimate) and are shown side by side with their source.
 */

export const CAMERA_SEGMENTS = [
  { key: 'trunk', label: 'Trunk', sites: ['chest', 'waist', 'abdomen', 'hip'] },
  { key: 'right_arm', label: 'Right arm', sites: ['upper_arm_right', 'forearm_right'] },
  { key: 'left_arm', label: 'Left arm', sites: ['upper_arm_left', 'forearm_left'] },
  { key: 'right_leg', label: 'Right leg', sites: ['thigh_right', 'calf_right'] },
  { key: 'left_leg', label: 'Left leg', sites: ['thigh_left', 'calf_left'] },
]

export const GROUP_LABELS = {
  whole_body: 'Whole body',
  segmental_fat: 'Segmental fat',
  segmental_muscle: 'Segmental muscle',
}

/** Parameters of the schema that a profile supports, in schema order, grouped. */
export function profileParameters(schema, profileKey) {
  const profile = schema?.profiles?.find((p) => p.key === profileKey)
  if (!profile) return []
  const allowed = new Set(profile.parameters)
  return (schema.parameters || []).filter((p) => allowed.has(p.key))
}

export function groupParameters(params) {
  const groups = []
  for (const p of params) {
    let g = groups.find((x) => x.key === p.group)
    if (!g) { g = { key: p.group, label: GROUP_LABELS[p.group] || p.group, params: [] }; groups.push(g) }
    g.params.push(p)
  }
  return groups
}

/** Local "YYYY-MM-DDTHH:MM" for a datetime-local input. */
export function localDateTimeInput(date = new Date()) {
  const pad = (n) => String(n).padStart(2, '0')
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`
}

/**
 * Form state from an existing record (edit) or empty (add). The ORIGINAL reading is shown for editing so a value
 * entered in lb stays in lb; untouched fields therefore round-trip exactly.
 */
export function formFromRecord(record, schema, now = new Date()) {
  const profile = record?.profile?.key || schema?.default_profile || 'generic_whole_body'
  const values = {}
  for (const p of schema?.parameters || []) {
    const v = record?.values?.[p.key]
    values[p.key] = v
      ? { value: String(v.original?.value ?? v.value), unit: v.original?.unit || v.unit }
      : { value: '', unit: p.unit }
  }
  return {
    profile,
    device_model: record?.source?.device_model || '',
    measured_at: record?.measured_at ? String(record.measured_at).slice(0, 16) : localDateTimeInput(now),
    note: record && !record.source?.is_demo ? record.note || '' : '',
    values,
  }
}

/** API payload: only parameters of the selected profile; blanks become null (= not measured). */
export function payloadFromForm(form, schema) {
  const allowed = new Set(profileParameters(schema, form.profile).map((p) => p.key))
  const values = {}
  for (const [key, v] of Object.entries(form.values || {})) {
    if (!allowed.has(key)) continue
    const text = String(v?.value ?? '').trim()
    values[key] = { value: text === '' ? null : text.replace(',', '.'), unit: v.unit }
  }
  return {
    profile: form.profile,
    device_model: form.device_model?.trim() || null,
    measured_at: form.measured_at || null,
    note: form.note || '',
    values,
  }
}

export function countEntered(form, schema) {
  const allowed = new Set(profileParameters(schema, form.profile).map((p) => p.key))
  return Object.entries(form.values || {}).filter(([k, v]) => allowed.has(k) && String(v?.value ?? '').trim() !== '').length
}

export function formatBcmValue(entry, param) {
  if (!entry || !Number.isFinite(entry.value)) return '—'
  const d = param?.decimals ?? 1
  return entry.value.toFixed(d)
}

/** "176.4 lb" when the reading was entered in a unit other than the stored one; otherwise null. */
export function originalReading(entry) {
  const o = entry?.original
  if (!o || !entry || o.unit === entry.unit) return null
  return `${o.value} ${o.unit}`
}

export function sourceBadge(record) {
  if (!record) return null
  const t = record.source?.type
  if (record.source?.is_demo || t === 'demo') return { text: 'DEMO DATA', tone: 'demo' }
  if (t === 'import') return { text: 'Imported', tone: 'import' }
  if (t === 'device') return { text: 'Device', tone: 'device' }
  return { text: 'Manual entry', tone: 'manual' }
}

/**
 * Rows pairing each body segment's camera girths with the same segment's BCM composition.
 * BCM cells are null when no record exists or the record lacks that segment — never estimated.
 */
export function segmentRows(camera, record) {
  const cm = camera?.measurements_cm
  const unit = cm ? 'cm' : 'ratio'
  const src = cm || camera?.measurements_ratio || {}
  const v = record?.values || {}
  return CAMERA_SEGMENTS.map((s) => ({
    key: s.key,
    label: s.label,
    camera: s.sites.map((site) => ({ site, value: Number.isFinite(src[site]) ? src[site] : null, valid: camera?.site_valid?.[site] !== false })),
    unit,
    bcm: {
      fat_pct: v[`segment_fat_pct_${s.key}`]?.value ?? null,
      fat_mass: v[`segment_fat_mass_${s.key}`]?.value ?? null,
      muscle_mass: v[`segment_muscle_mass_${s.key}`]?.value ?? null,
    },
  }))
}

export function hasSegmental(record) {
  return Object.keys(record?.values || {}).some((k) => k.startsWith('segment_'))
}

/** Signed change text; null when either side is missing (no change is invented from one value). */
export function formatChange(delta, decimals = 1) {
  if (!Number.isFinite(delta)) return null
  const r = Number(delta.toFixed(decimals))
  if (r === 0) return '±0'
  return `${r > 0 ? '+' : '−'}${Math.abs(r).toFixed(decimals)}`
}

/** Field-level error text from a 422 response detail ({message, errors}) or a plain string. */
export function fieldErrors(detail) {
  if (detail && typeof detail === 'object' && detail.errors) return detail.errors
  return {}
}

export function formatWhen(iso) {
  if (!iso) return '—'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return String(iso)
  return d.toLocaleString(undefined, { year: 'numeric', month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}
