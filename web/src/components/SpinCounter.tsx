import { useEffect, useMemo, useRef, useState } from 'react'
import type { Session } from '../types'
import { linear } from '../lib/scale'

type Mode = 'total' | 'air'

interface Props {
  session: Session
  frame: number
  onSeek: (frame: number) => void
}

interface Series {
  /** Spins counted so far across every air, held flat between airs. */
  total: number[]
  /** Spins in the current (or most recent) air only; resets when the next air starts. */
  air: number[]
}

/** Rotation is only counted while the rider is inside a detected air. Outside
 * one the value is held, so the curve climbs during a trick and plateaus while
 * gliding. Counts are unsigned: a backside and a frontside spin both add. */
function buildSeries(session: Session): Series {
  const { f, rotation_deg } = session.telemetry
  const total: number[] = []
  const air: number[] = []
  let banked = 0
  let local = 0
  let anchor = 0
  let current: string | null = null
  for (let i = 0; i < f.length; i++) {
    const ev = session.events.find((e) => f[i] >= e.start_f && f[i] <= e.end_f)
    if (ev) {
      if (ev.id !== current) {
        banked += local
        local = 0
        anchor = rotation_deg[i]
        current = ev.id
      }
      local = Math.abs(rotation_deg[i] - anchor)
    }
    total.push((banked + local) / 360)
    air.push(local / 360)
  }
  return { total, air }
}

const M = { top: 10, right: 16, bottom: 22, left: 40 }
const H = 150

function fmtSpins(v: number): string {
  return v.toFixed(2)
}

export default function SpinCounter({ session, frame, onSeek }: Props) {
  const [mode, setMode] = useState<Mode>('total')
  const wrapRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(880)
  const dragging = useRef(false)

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const series = useMemo(() => buildSeries(session), [session])
  const values = series[mode]
  const { f } = session.telemetry
  const { total_frames, fps } = session.clip

  // Last tracked sample at or before the playhead.
  const cur = useMemo(() => {
    let lo = 0
    let hi = f.length - 1
    let ans = -1
    while (lo <= hi) {
      const mid = (lo + hi) >> 1
      if (f[mid] <= frame) {
        ans = mid
        lo = mid + 1
      } else hi = mid - 1
    }
    return ans
  }, [f, frame])

  const spins = cur < 0 ? 0 : values[cur]
  const activeEvent = session.events.find((e) => frame >= e.start_f && frame <= e.end_f)
  const peak = Math.max(...values, 0)
  const top = Math.max(1, Math.ceil(peak * 2) / 2)
  const step = top <= 2 ? 0.5 : 1

  const x = linear([0, total_frames - 1], [M.left, Math.max(M.left + 40, width - M.right)])
  const y = linear([0, top], [H - M.bottom, M.top])

  const ticks: number[] = []
  for (let v = 0; v <= top + 1e-9; v += step) ticks.push(v)

  const path = f.map((fr, i) => `${i === 0 ? 'M' : 'L'}${x(fr).toFixed(1)},${y(values[i]).toFixed(1)}`).join('')
  const px = x(frame)

  const seekFromEvent = (clientX: number) => {
    const rect = wrapRef.current!.getBoundingClientRect()
    const fr = Math.round(x.invert(clientX - rect.left))
    onSeek(Math.max(0, Math.min(total_frames - 1, fr)))
  }

  return (
    <div className="card">
      <div className="card-head">
        <span className="card-title">Spin counter</span>
        <span className="card-note">counts only while airborne · plateaus while gliding</span>
        <div className="toggles" style={{ marginLeft: 12 }}>
          {([['total', 'Total'], ['air', 'Per air']] as const).map(([k, label]) => (
            <button key={k} className="toggle" aria-pressed={mode === k} onClick={() => setMode(k)}>
              {label}
            </button>
          ))}
        </div>
      </div>
      <div className="card-body">
        <div className="spin-readout">
          <span className="spin-big">{fmtSpins(spins)}</span>
          <span className="spin-unit">spins</span>
          <span className="spin-deg">{(spins * 360).toFixed(0)}°</span>
          <span className="spin-state" data-live={!!activeEvent}>
            {activeEvent ? `counting · air ${activeEvent.id}` : 'gliding · not counting'}
          </span>
        </div>

        <div className="timeline" ref={wrapRef}>
          <svg
            width={width} height={H} role="img"
            aria-label="Spins counted so far against clip time"
            onPointerDown={(e) => {
              dragging.current = true
              e.currentTarget.setPointerCapture(e.pointerId)
              seekFromEvent(e.clientX)
            }}
            onPointerMove={(e) => dragging.current && seekFromEvent(e.clientX)}
            onPointerUp={() => (dragging.current = false)}
            onPointerCancel={() => (dragging.current = false)}
          >
            <defs>
              <clipPath id="spin-played">
                <rect x={0} y={0} width={Math.max(0, px)} height={H} />
              </clipPath>
            </defs>

            {ticks.map((v) => (
              <g key={v}>
                <line x1={M.left} x2={x(total_frames - 1)} y1={y(v)} y2={y(v)} stroke={v === 0 ? 'var(--axis)' : 'var(--grid)'} />
                <text x={M.left - 8} y={y(v) + 3.5} textAnchor="end" fontSize="10" fill="var(--ink-muted)" style={{ fontVariantNumeric: 'tabular-nums' }}>
                  {v}
                </text>
              </g>
            ))}
            <text x={M.left - 8} y={M.top - 1} textAnchor="end" fontSize="9.5" fill="var(--ink-muted)">spins</text>

            {/* The airs: the only stretches where the curve is allowed to move. */}
            {session.events.map((ev) => (
              <rect
                key={ev.id} x={x(ev.start_f)} y={M.top} width={Math.max(2, x(ev.end_f) - x(ev.start_f))}
                height={H - M.bottom - M.top} fill="var(--series-1)" opacity="0.09"
              />
            ))}

            {/* Whole curve faint, the part already played strong, so it visibly climbs with the video. */}
            <path d={path} fill="none" stroke="var(--series-3)" strokeWidth="2" opacity="0.25" strokeLinejoin="round" />
            <path d={path} fill="none" stroke="var(--series-3)" strokeWidth="2.5" strokeLinejoin="round" clipPath="url(#spin-played)" />
            {cur >= 0 && <circle cx={x(f[cur])} cy={y(values[cur])} r="4.5" fill="var(--series-3)" stroke="var(--surface-1)" strokeWidth="2" />}

            {[0, ...Array.from({ length: Math.floor((total_frames - 1) / fps / 2) }, (_, i) => (i + 1) * 2)].map((sec) => (
              <text key={sec} x={x(sec * fps)} y={H - 6} textAnchor="middle" fontSize="10" fill="var(--ink-muted)">{sec}s</text>
            ))}

            <line x1={px} x2={px} y1={M.top} y2={H - M.bottom} stroke="var(--ink)" strokeWidth="1.5" />
          </svg>
        </div>
      </div>
    </div>
  )
}
