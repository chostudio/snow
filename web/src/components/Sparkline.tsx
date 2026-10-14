interface Props {
  values: number[]
  color: string
  width?: number
  height?: number
  /** Marks zero when the series is signed, so a sign flip is visible. */
  zeroLine?: boolean
}

export default function Sparkline({ values, color, width = 148, height = 26, zeroLine }: Props) {
  if (values.length < 2) return null
  const lo = Math.min(...values, zeroLine ? 0 : Infinity)
  const hi = Math.max(...values, zeroLine ? 0 : -Infinity)
  const span = hi - lo || 1
  const x = (i: number) => (i / (values.length - 1)) * (width - 2) + 1
  const y = (v: number) => height - 2 - ((v - lo) / span) * (height - 4)
  const d = values.map((v, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join('')
  return (
    <svg width={width} height={height} aria-hidden="true">
      {zeroLine && lo < 0 && hi > 0 && (
        <line x1="0" y1={y(0)} x2={width} y2={y(0)} stroke="var(--axis)" strokeWidth="1" />
      )}
      <path d={d} fill="none" stroke={color} strokeWidth="1.75" strokeLinejoin="round" strokeLinecap="round" />
      <circle cx={x(values.length - 1)} cy={y(values[values.length - 1])} r="2.5" fill={color} />
    </svg>
  )
}
