/**
 * sizeCharts.js
 * -------------
 * Available size charts: the built-in example, every JSON in src/sizeCharts/,
 * and charts the operator uploaded (kept in this browser's localStorage).
 */
import { EXAMPLE_CHART, validateChart } from './sizeRecommendation.js'

const CUSTOM_KEY = 'sizeCharts.custom'
const SELECTED_KEY = 'sizeCharts.selected'

const shipped = Object.values(import.meta.glob('../sizeCharts/*.json', { eager: true, import: 'default' }))
  .map((raw) => validateChart(raw).chart)
  .filter(Boolean)

export const BUILT_IN_CHARTS = [EXAMPLE_CHART, ...shipped.filter((c) => c.id !== EXAMPLE_CHART.id)]

export function loadCustomCharts() {
  try {
    const list = JSON.parse(localStorage.getItem(CUSTOM_KEY) || '[]')
    return Array.isArray(list) ? list.map((raw) => validateChart(raw).chart).filter(Boolean) : []
  } catch {
    return []
  }
}

export function saveCustomCharts(charts) {
  try { localStorage.setItem(CUSTOM_KEY, JSON.stringify(charts)) } catch { /* storage unavailable: keep in memory only */ }
}

export function loadSelectedChartId() {
  try { return localStorage.getItem(SELECTED_KEY) } catch { return null }
}

export function saveSelectedChartId(id) {
  try { localStorage.setItem(SELECTED_KEY, id) } catch { /* ignore */ }
}

/** Uploaded charts get an id that cannot collide with a built-in one. */
export function customId(chart) {
  return chart.id.startsWith('custom-') ? chart.id : `custom-${chart.id}`
}
