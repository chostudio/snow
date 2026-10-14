import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import type { Session, TelemetryKey, TrickEvent } from '../types'
import { linear, niceDomain, segmentedPath, ticksEvery, type Scale } from '../lib/scale'
import { signed, timecode } from '../lib/format'

interface PanelSpec {
  key: TelemetryKey
  label: string
  unit: string
  color: string
  height: number
  signedZero: boolean
  /** Horizontal reference line, e.g. the horizontal spin axis at 90°. */
  ref?: { at: number; label: string }
  digits: number
}

/** Three measures of different scale means three panels, never one chart with
 * two y-axes. They share the x-axis, the playhead and the crosshair, so the
 * stack still reads as a single instrument. */
function panelsFor(session: Session): PanelSpec[] {
  const panels: PanelSpec[] = [
    { key: 'rpm', label: 'Rotation rate', unit: 'RPM', color: 'var(--series-1)', height: 104, signedZero: true, digits: 0 },
    {
      key: 'rotation_deg', label: `Cumulative rotation — ${session.source.rotation_measure}`,
      unit: '°', color: 'var(--series-3)', height: 76, signedZero: true, digits: 0,
    },
    {
      key: 'axis_tilt_deg', label: 'Body tilt from vertical', unit: '°', color: 'var(--series-2)',
      height: 76, signedZero: false, digits: 0, ref: { at: 90, label: 'horizontal' },
    },
  ]
  // Not a measure of the rider — it is the contamination that was taken out,
  // so it wears muted ink rather than a series colour.
  if (session.telemetry.camera_rotation_cum_deg) {
    panels.push({
      key: 'camera_rotation_cum_deg', label: 'Camera pan, already subtracted', unit: '°',
      color: 'var(--ink-muted)', height: 56, signedZero: true, digits: 0,
    })
  }
  return panels
}

const M = { top: 28, right: 16, bottom: 24, left: 50 }
const PANEL_GAP = 22
const CHIP_W = 92

interface Props {
  session: Session
  frame: number
  onSeek: (frame: number) => void
  selectedId: string | null
  onSelect: (event: TrickEvent) => void
}

export default function TelemetryTimeline({ session, frame, onSeek, selectedId, onSelect }: Props) {
  const wrapRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(880)
  const [hover, setHover] = useState<number | null>(null)
  const dragging = useRef(false)

  useEffect(() => {
    const el = wrapRef.current
    if (!el) return
    const ro = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width))
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const { fps, total_frames } = session.clip
  const specs = useMemo(() => panelsFor(session), [session])
  const height = M.top + M.bottom + specs.reduce((a, p) => a + p.height, 0) + PANEL_GAP * (specs.length - 1)

  const layout = useMemo(() => {
    const x = linear([0, total_frames - 1], [M.left, Math.max(M.left + 40, width - M.right)])
    let y0 = M.top
    const panels = specs.map((spec) => {
      const values = (session.telemetry[spec.key] ?? []) as number[]
      const domain = niceDomain(values, spec.signedZero)
      const y: Scale = linear(domain, [y0 + spec.height, y0])
      const top = y0
      y0 += spec.height + PANEL_GAP
      return { spec, y, top, values, domain }
    })
    return { x, panels }
  }, [session, specs, width, total_frames])

  const idx = useMemo(() => {
    const m = new Map<number, number>()
    session.telemetry.f.forEach((f, i) => m.set(f, i))
    return m
  }, [session])

  const contiguous = useCallback(
    (i: number) => i > 0 && session.telemetry.f[i] - session.telemetry.f[i - 1] === 1,
    [session],
  )

  const frameFromEvent = useCallback(
    (clientX: number) => {
      const rect = wrapRef.current!.getBoundingClientRect()
      const f = Math.round(layout.x.invert(clientX - rect.left))
      return Math.max(0, Math.min(total_frames - 1, f))
    },
    [layout, total_frames],
  )

  const onPointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    dragging.current = true
    e.currentTarget.setPointerCapture(e.pointerId)
    const f = frameFromEvent(e.clientX)
    onSeek(f)
    const hit = session.events.find((ev) => f >= ev.start_f && f <= ev.end_f)
    if (hit) onSelect(hit)
  }

  const onPointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const f = frameFromEvent(e.clientX)
    setHover(f)
    if (dragging.current) onSeek(f)
  }

  const endDrag = () => {
    dragging.current = false
  }

  const plotLeft = layout.x(0)
  const plotRight = layout.x(total_frames - 1)
  const plotBottom = height - M.bottom
  const hoverI = hover === null ? undefined : idx.get(hover)
  const tooltipX = hover === null ? 0 : layout.x(hover)
  const flip = tooltipX > width - 190

  return (
    <div className="timeline" ref={wrapRef}>
      <svg
        width={width} height={height} role="img"
        aria-label="Rotation rate, cumulative rotation and spin-axis tilt against clip time"
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onPointerLeave={() => {
          setHover(null)
          endDrag()
        }}
      >
        <defs>
          <pattern id="gap-hatch" width="7" height="7" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <line x1="0" y1="0" x2="0" y2="7" stroke="var(--axis)" strokeWidth="1" />
          </pattern>
        </defs>

        {/* Camera cuts: no rider tracked, so no measurement exists here. */}
        {session.gaps.map(([a, b]) => (
          <g key={`gap-${a}`}>
            <rect
              x={layout.x(a)} y={M.top} width={Math.max(1, layout.x(b) - layout.x(a))}
              height={plotBottom - M.top} fill="url(#gap-hatch)" opacity="0.5"
            />
            {layout.x(b) - layout.x(a) > 54 && (
              <text
                x={(layout.x(a) + layout.x(b)) / 2} y={M.top + 12} textAnchor="middle"
                fontSize="9.5" fill="var(--ink-muted)" letterSpacing="0.08em"
              >
                NO TRACK
              </text>
            )}
          </g>
        ))}

        {/* Air events. */}
        {session.events.map((ev, i) => {
          const x0 = layout.x(ev.start_f)
          const x1 = layout.x(ev.end_f)
          const on = ev.id === selectedId
          // Two airs separated by a camera cut can sit a few pixels apart; drop
          // the second chip rather than stack two unreadable ones. The band is
          // still there to click, and the Airs list names every one.
          const prev = session.events[i - 1]
          const showChip = !prev || layout.x(prev.start_f) + CHIP_W + 4 < x0
          return (
            <g key={ev.id}>
              <rect
                x={x0} y={M.top} width={Math.max(2, x1 - x0)} height={plotBottom - M.top}
                fill="var(--series-1)" opacity={on ? 0.14 : 0.07}
              />
              <line x1={x0} y1={M.top} x2={x0} y2={plotBottom} stroke="var(--series-1)" strokeWidth={on ? 2 : 1} />
              <line x1={x1} y1={M.top} x2={x1} y2={plotBottom} stroke="var(--series-1)" strokeWidth={on ? 2 : 1} opacity="0.7" />
              {showChip && (
                <>
                  <rect
                    x={x0 + 2} y={M.top - 17} rx="3" height="15" fill="var(--surface-1)"
                    stroke="var(--series-1)" strokeOpacity="0.4" width={CHIP_W}
                  />
                  <text x={x0 + 8} y={M.top - 6} fontSize="10.5" fill="var(--series-1)" fontWeight="600">
                    {ev.metrics.nearest_spin_deg}° · {Math.abs(ev.metrics.peak_rpm).toFixed(0)} RPM
                  </text>
                </>
              )}
            </g>
          )
        })}

        {layout.panels.map(({ spec, y, top, values, domain }) => {
          const zero = domain[0] < 0 && domain[1] > 0 ? y(0) : null
          return (
            <g key={spec.key}>
              <line x1={plotLeft} y1={top} x2={plotRight} y2={top} stroke="var(--grid)" strokeWidth="1" />
              {zero !== null && (
                <line x1={plotLeft} y1={zero} x2={plotRight} y2={zero} stroke="var(--axis)" strokeWidth="1" />
              )}
              {spec.ref && spec.ref.at > domain[0] && spec.ref.at < domain[1] && (
                <>
                  <line
                    x1={plotLeft} y1={y(spec.ref.at)} x2={plotRight} y2={y(spec.ref.at)}
                    stroke="var(--axis)" strokeWidth="1" strokeDasharray="3 4"
                  />
                  <text x={plotRight - 2} y={y(spec.ref.at) - 4} textAnchor="end" fontSize="9.5" fill="var(--ink-muted)">
                    {spec.ref.label}
                  </text>
                </>
              )}
              <text x={plotLeft} y={top - 4} fontSize="11" fill="var(--ink-2)" fontWeight="600">
                {spec.label}
                <tspan fill="var(--ink-muted)" fontWeight="400"> · {spec.unit}</tspan>
              </text>
              {[domain[1], domain[0]].map((v, i) => (
                <text
                  key={i} x={M.left - 8} y={y(v) + (i === 0 ? 8 : 0)} textAnchor="end"
                  fontSize="10" fill="var(--ink-muted)" style={{ fontVariantNumeric: 'tabular-nums' }}
                >
                  {v.toFixed(spec.digits)}
                </text>
              ))}
              <path
                d={segmentedPath(session.telemetry.f, values, contiguous, layout.x, y)}
                fill="none" stroke={spec.color} strokeWidth="2" strokeLinejoin="round" strokeLinecap="round"
              />
              {hoverI !== undefined && (
                <circle cx={layout.x(hover!)} cy={y(values[hoverI])} r="4" fill={spec.color} stroke="var(--surface-1)" strokeWidth="2" />
              )}
            </g>
          )
        })}

        {/* x axis */}
        <line x1={plotLeft} y1={plotBottom} x2={plotRight} y2={plotBottom} stroke="var(--axis)" strokeWidth="1" />
        {ticksEvery([0, (total_frames - 1) / fps], 2).map((sec) => (
          <g key={sec}>
            <line x1={layout.x(sec * fps)} y1={plotBottom} x2={layout.x(sec * fps)} y2={plotBottom + 4} stroke="var(--axis)" />
            <text
              x={layout.x(sec * fps)} y={plotBottom + 16} textAnchor="middle" fontSize="10"
              fill="var(--ink-muted)" style={{ fontVariantNumeric: 'tabular-nums' }}
            >
              {sec}s
            </text>
          </g>
        ))}

        {hover !== null && (
          <line x1={layout.x(hover)} y1={M.top} x2={layout.x(hover)} y2={plotBottom} stroke="var(--ink-muted)" strokeWidth="1" strokeDasharray="3 3" />
        )}

        {/* Playhead — the same frame the video and the overlay are showing. */}
        <line x1={layout.x(frame)} y1={M.top} x2={layout.x(frame)} y2={plotBottom + 2} stroke="var(--ink)" strokeWidth="1.5" />
        <polygon
          points={`${layout.x(frame) - 5},${M.top - 26} ${layout.x(frame) + 5},${M.top - 26} ${layout.x(frame)},${M.top - 18}`}
          fill="var(--ink)"
        />
      </svg>

      {hover !== null && (
        <div className="tt" style={{ left: flip ? tooltipX - 170 : tooltipX + 14, top: M.top + 6 }}>
          <div className="tt-head">
            {timecode(hover / fps)} · frame {hover}
          </div>
          {hoverI === undefined ? (
            <div className="tt-row">
              <span className="k">No rider tracked</span>
            </div>
          ) : (
            specs.map((p) => (
              <div className="tt-row" key={p.key}>
                <span className="swatch" style={{ background: p.color }} />
                <span className="k">{p.label.split(' — ')[0]}</span>
                <span className="v">
                  {p.signedZero
                    ? signed((session.telemetry[p.key] as number[])[hoverI], p.digits)
                    : (session.telemetry[p.key] as number[])[hoverI].toFixed(p.digits)}
                  {p.unit === '°' ? '°' : ` ${p.unit}`}
                </span>
              </div>
            ))
          )}
        </div>
      )}
    </div>
  )
}
