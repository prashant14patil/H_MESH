import { qualityGroup } from './frameQuality.js'

// Review aids only: these hints do not approve frames or establish viewpoint coverage.
export function recommendFrames(frames, report, viewpoints, limit = 12) {
  const quality = Object.fromEntries((report?.frames || []).map(f => [f.filename, f]))
  const views = Object.fromEntries((viewpoints?.frames || []).map(f => [f.frame, f]))
  const groups = new Map()
  const excluded = {}
  for (const frame of frames || []) {
    const q = quality[frame.filename]
    if (qualityGroup(q) !== 'passed') {
      excluded[frame.filename] = 'Mandatory quality checks failed or are unknown.'
      continue
    }
    if (frame.is_blurry !== false) {
      excluded[frame.filename] = 'Extractor blur status is flagged or unknown.'
      continue
    }
    // The extractor's configured duplicate threshold is not present in metadata.
    // A zero difference is unambiguous; other similarity scores are not calibrated here.
    if (frame.diff_score === 0) {
      excluded[frame.filename] = 'Extractor reported no difference from a recent accepted frame.'
      continue
    }
    const view = views[frame.filename.replace(/\.[^.]+$/, '')]
    const label = view?.viewpoint || 'unknown'
    const bucket = ['front', 'back', 'left_side', 'right_side', 'side_ambiguous', 'intermediate'].includes(label) ? label : 'unknown'
    const item = { filename: frame.filename, bucket, score: Number.isFinite(frame.blur_score) ? frame.blur_score : -1,
      time: Number.isFinite(frame.timestamp_sec) ? frame.timestamp_sec : null }
    if (!groups.has(bucket)) groups.set(bucket, [])
    groups.get(bucket).push(item)
  }
  for (const items of groups.values()) items.sort((a, b) => b.score - a.score || a.filename.localeCompare(b.filename))
  const recommended = []
  const order = ['front', 'side_ambiguous', 'left_side', 'right_side', 'intermediate', 'back', 'unknown']
  while (recommended.length < limit && order.some(key => groups.get(key)?.length)) {
    for (const key of order) {
      if (recommended.length >= limit) break
      const item = groups.get(key)?.shift()
      if (!item) continue
      if (item.time !== null && recommended.some(other => other.time !== null && Math.abs(other.time - item.time) < 1.5)) {
        excluded[item.filename] = 'Close in time to another advisory candidate; visual duplication is not verified.'
      } else {
        recommended.push(item)
      }
    }
  }
  for (const items of groups.values()) for (const item of items) excluded[item.filename] = 'Not prioritized within the 12-frame advisory limit.'
  return { recommended, excluded }
}
