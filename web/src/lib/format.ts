export const f0 = (n: number) => n.toFixed(0)
export const f1 = (n: number) => n.toFixed(1)
export const f2 = (n: number) => n.toFixed(2)

export function signed(n: number, digits = 0): string {
  return `${n > 0 ? '+' : n < 0 ? '−' : ''}${Math.abs(n).toFixed(digits)}`
}

/** mm:ss.cc — a judge reads a clip in seconds, not in frames. */
export function timecode(seconds: number): string {
  const s = Math.max(0, seconds)
  const m = Math.floor(s / 60)
  const rest = s - m * 60
  return `${m}:${rest.toFixed(2).padStart(5, '0')}`
}

export const SEVERITY_LABEL = {
  good: 'Strength',
  warning: 'Watch',
  serious: 'Flaw',
  critical: 'Risk',
} as const
