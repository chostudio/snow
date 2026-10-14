import { useEffect, useRef } from 'react'
import type { PoseFrame, Session, TrickEvent } from '../types'
import { CORE_JOINTS, GROUP_COLOR, groupOf } from '../lib/pose'
import { signed, timecode } from '../lib/format'

export interface OverlayFlags {
  skeleton: boolean
  box: boolean
  trail: boolean
}

interface Props {
  session: Session
  videoRef: React.MutableRefObject<HTMLVideoElement | null>
  clip: string
  frame: number
  poses: Map<number, PoseFrame>
  links: [number, number][]
  flags: OverlayFlags
  rpm: number | null
  rotation: number | null
  activeEvent: TrickEvent | null
}

const TRAIL_FRAMES = 36

export default function VideoStage({
  session, videoRef, clip, frame, poses, links, flags, rpm, rotation, activeEvent,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const pose = poses.get(frame) ?? null

  useEffect(() => {
    const canvas = canvasRef.current
    const ctx = canvas?.getContext('2d')
    if (!canvas || !ctx) return
    const { width, height } = session.clip
    ctx.clearRect(0, 0, width, height)
    if (!pose) return

    if (flags.trail) {
      ctx.lineWidth = 3
      ctx.strokeStyle = 'rgba(57, 135, 229, 0.55)'
      ctx.beginPath()
      let started = false
      for (let f = frame - TRAIL_FRAMES; f <= frame; f++) {
        const p = poses.get(f)
        if (!p) {
          started = false
          continue
        }
        const [x, y] = p.k2[0]
        if (started) ctx.lineTo(x, y)
        else ctx.moveTo(x, y)
        started = true
      }
      ctx.stroke()
    }

    if (flags.box) {
      const [bx, by, bw, bh] = pose.bbox
      ctx.strokeStyle = 'rgba(240, 239, 236, 0.35)'
      ctx.lineWidth = 2
      ctx.setLineDash([10, 8])
      ctx.strokeRect(bx, by, bw, bh)
      ctx.setLineDash([])
    }

    if (flags.skeleton) {
      ctx.lineCap = 'round'
      for (const [a, b] of links) {
        const ca = pose.conf[a] / 100
        const cb = pose.conf[b] / 100
        ctx.globalAlpha = Math.max(0.15, Math.min(ca, cb))
        ctx.strokeStyle = GROUP_COLOR[groupOf(a)]
        ctx.lineWidth = 4
        ctx.beginPath()
        ctx.moveTo(pose.k2[a][0], pose.k2[a][1])
        ctx.lineTo(pose.k2[b][0], pose.k2[b][1])
        ctx.stroke()
      }
      for (const j of CORE_JOINTS) {
        ctx.globalAlpha = Math.max(0.2, pose.conf[j] / 100)
        ctx.fillStyle = GROUP_COLOR[groupOf(j)]
        ctx.beginPath()
        ctx.arc(pose.k2[j][0], pose.k2[j][1], 4.5, 0, Math.PI * 2)
        ctx.fill()
        // A surface ring keeps joints readable where limbs overlap.
        ctx.strokeStyle = 'rgba(11, 11, 10, 0.8)'
        ctx.lineWidth = 2
        ctx.stroke()
      }
      ctx.globalAlpha = 1
    }
  }, [pose, poses, frame, flags, links, session.clip])

  const t = frame / session.clip.fps
  const inAir = activeEvent !== null

  return (
    <div className="stage">
      <video ref={videoRef} src={clip} preload="auto" playsInline />
      <canvas ref={canvasRef} width={session.clip.width} height={session.clip.height} />
      <div className="hud">
        <div className="hud-block">
          <div className="hud-label">Clip time</div>
          <div className="hud-value">{timecode(t)}</div>
          <div className="hud-label" style={{ marginTop: 4 }}>
            frame {frame} / {session.clip.total_frames - 1}
          </div>
        </div>
        <div className="hud-right">
          {inAir && (
            <div className="hud-block" style={{ borderColor: 'var(--series-1)' }}>
              <div className="hud-label">In air</div>
              <div className="hud-value" style={{ fontSize: 15 }}>
                {activeEvent!.metrics.nearest_spin_deg}°
              </div>
            </div>
          )}
          <div className="hud-block">
            <div className="hud-label">Rotation rate</div>
            <div className="hud-value" style={{ color: rpm === null ? 'var(--ink-muted)' : undefined }}>
              {rpm === null ? '—' : signed(rpm)}
              <span style={{ fontSize: 11, color: 'var(--ink-muted)', marginLeft: 4 }}>RPM</span>
            </div>
          </div>
          <SpinDial rotation={rotation} target={activeEvent?.metrics.nearest_spin_deg ?? null} />
        </div>
      </div>
      {!pose && (
        <div className="notrack">
          <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
            <path d="M7 1.2 13 12.4H1Z" fill="none" stroke="currentColor" strokeWidth="1.6" />
            <path d="M7 5.4v3.1" stroke="currentColor" strokeWidth="1.6" />
            <circle cx="7" cy="10.4" r="0.9" fill="currentColor" />
          </svg>
          No rider tracked — camera cut
        </div>
      )}
    </div>
  )
}

/** Where the rider is inside the rotation, and how much of the named spin is
 * already banked. The needle is the measured body-axis angle, not an estimate. */
function SpinDial({ rotation, target }: { rotation: number | null; target: number | null }) {
  const r = 22
  const angle = ((rotation ?? 0) % 360) * (Math.PI / 180)
  const progress = target && rotation !== null ? Math.min(1, Math.abs(rotation) / target) : 0
  const circ = 2 * Math.PI * r
  return (
    <div className="hud-block" style={{ padding: 8 }}>
      <svg width="58" height="58" viewBox="-29 -29 58 58" aria-hidden="true">
        <circle r={r} fill="none" stroke="var(--grid)" strokeWidth="4" />
        <circle
          r={r} fill="none" stroke="var(--series-1)" strokeWidth="4" strokeLinecap="round"
          strokeDasharray={`${circ * progress} ${circ}`} transform="rotate(-90)"
        />
        <line
          x1="0" y1="0" x2={Math.sin(angle) * (r - 7)} y2={-Math.cos(angle) * (r - 7)}
          stroke="var(--ink)" strokeWidth="2" strokeLinecap="round"
        />
        <circle r="2.5" fill="var(--ink)" />
      </svg>
    </div>
  )
}
