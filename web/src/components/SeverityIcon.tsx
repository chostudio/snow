import type { Severity } from '../types'

const COLOR: Record<Severity, string> = {
  good: 'var(--good)',
  warning: 'var(--warning)',
  serious: 'var(--serious)',
  critical: 'var(--critical)',
}

/** Status colour never travels alone — every use pairs this glyph with the
 * written severity label beside it. */
export default function SeverityIcon({ severity, size = 17 }: { severity: Severity; size?: number }) {
  const c = COLOR[severity]
  if (severity === 'good') {
    return (
      <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true">
        <circle cx="8" cy="8" r="7" fill="none" stroke={c} strokeWidth="1.6" />
        <path d="M4.6 8.3 6.9 10.6 11.4 5.7" fill="none" stroke={c} strokeWidth="1.8" strokeLinecap="round" />
      </svg>
    )
  }
  if (severity === 'critical') {
    return (
      <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true">
        <path d="M5.4 1.4h5.2L14.6 5.4v5.2L10.6 14.6H5.4L1.4 10.6V5.4Z" fill="none" stroke={c} strokeWidth="1.6" />
        <path d="M8 4.6v4.2" stroke={c} strokeWidth="1.8" strokeLinecap="round" />
        <circle cx="8" cy="11.4" r="1" fill={c} />
      </svg>
    )
  }
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" aria-hidden="true">
      <path d="M8 1.6 15 14.4H1Z" fill="none" stroke={c} strokeWidth="1.6" strokeLinejoin="round" />
      <path d="M8 6.2v3.6" stroke={c} strokeWidth="1.8" strokeLinecap="round" />
      <circle cx="8" cy="11.9" r="1" fill={c} />
    </svg>
  )
}

export { COLOR as SEVERITY_COLOR }
